"""FCStd backend.

Reads ``.FCStd`` (FreeCAD documents) without needing the FreeCAD
binary. An FCStd is a zip archive that contains a ``Document.xml``
manifest plus one OpenCASCADE BREP file per shape. The BREPs are
loaded with OCP and tessellated with the same path as :func:`read_brep`.

The reader preserves:

* ``cad.label`` — FreeCAD object label.
* ``cad.color`` — RGBA (4-tuple). Alpha is derived from
  ``ViewObject.Transparency`` (0-100 percent) when present, defaulting
  to ``1.0`` (fully opaque) otherwise.
* Assembly / group hierarchy — ``App::DocumentObjectGroup``,
  ``Assembly``, ``Assembly3`` and A2plus parts that own a ``Group``
  link-list become nested :class:`pyvista.MultiBlock` containers.

Corrupt or unreadable BREP entries are surfaced as
:class:`~pyvista_cad.CadReadError` with the offending FreeCAD object
name in the message — they are never silently swallowed.
"""

import os
from pathlib import Path
import tempfile
from typing import Any
import warnings
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pyvista as pv

from pyvista_cad._errors import CadReadError, CadWarning, OptionalDependencyError

_GROUP_TYPES = frozenset(
    {
        'App::DocumentObjectGroup',
        'App::Part',
        'Assembly::AssemblyObject',
        'Assembly3::AssemblyObject',
        'Asm4::Assembly',
        'A2plus::AssemblyObject',
    }
)


def _require_ocp() -> None:
    try:
        import OCP  # noqa: F401
    except ImportError as exc:
        msg = 'OCP not installed; install pyvista-cad[step]'
        raise OptionalDependencyError(msg) from exc


def _parse_document_xml(xml_bytes: bytes) -> dict[str, dict[str, Any]]:
    """Map FCStd object names to label / color / brep-file / children.

    The returned mapping carries one entry per FreeCAD object, keyed by
    the internal object name. Recognised keys per entry:

    ``name``, ``type``, ``label``, ``brep_file``, ``color`` (RGB 3-tuple),
    ``transparency`` (0.0-1.0, where 0.0 == opaque), ``children`` (list
    of child object names, in tree order).
    """
    root = ET.fromstring(xml_bytes)
    objects: dict[str, dict[str, Any]] = {}
    for obj in root.iter('Object'):
        name = obj.get('name')
        if not name:
            continue
        objects[name] = {
            'name': name,
            'type': obj.get('type', ''),
            'label': name,
            'children': [],
        }

    for obj in root.findall('.//ObjectData/Object'):
        name = obj.get('name')
        if not name or name not in objects:
            continue
        for prop in obj.iter('Property'):
            pname = prop.get('name')
            if pname == 'Label':
                s = prop.find('String')
                if s is not None and s.get('value'):
                    objects[name]['label'] = s.get('value')
            elif pname == 'Shape':
                part = prop.find('Part')
                if part is not None and part.get('file'):
                    objects[name]['brep_file'] = part.get('file')
            elif pname == 'ShapeColor':
                col = prop.find('PropertyColor')
                raw_color = col.get('value') if col is not None else None
                if raw_color:
                    try:
                        rgba = int(raw_color)
                    except ValueError:
                        warnings.warn(
                            f'FreeCAD object {name!r}: could not decode ShapeColor '
                            f'value {raw_color!r}; the part will render without its '
                            'authored color.',
                            CadWarning,
                            stacklevel=2,
                        )
                    else:
                        r = ((rgba >> 24) & 0xFF) / 255.0
                        g = ((rgba >> 16) & 0xFF) / 255.0
                        b = ((rgba >> 8) & 0xFF) / 255.0
                        objects[name]['color'] = (r, g, b)
            elif pname == 'Transparency':
                # FreeCAD ``ViewObject.Transparency`` is stored as an
                # ``App::PropertyInteger`` in percent (0 == opaque,
                # 100 == fully transparent).
                ival = prop.find('Integer')
                raw_pct = ival.get('value') if ival is not None else None
                if raw_pct is not None:
                    try:
                        pct = int(raw_pct)
                    except ValueError:
                        warnings.warn(
                            f'FreeCAD object {name!r}: could not decode '
                            f'Transparency value {raw_pct!r}; the part will '
                            'render fully opaque.',
                            CadWarning,
                            stacklevel=2,
                        )
                    else:
                        objects[name]['transparency'] = max(0.0, min(1.0, pct / 100.0))
            elif pname == 'Group':
                # ``PropertyLinkList`` — children of a group / assembly.
                child_names: list[str] = []
                for link in prop.iter('Link'):
                    cname = link.get('value')
                    if cname:
                        child_names.append(cname)
                if child_names:
                    objects[name]['children'] = child_names

    return objects


def _stamp_part(
    poly: pv.PolyData,
    info: dict[str, Any],
    obj_name: str,
) -> str:
    """Tag a single tessellated part with label / color / source metadata."""
    label = info.get('label', obj_name)
    poly.field_data['cad.label'] = np.array([label])
    poly.field_data['cad.source_format'] = np.array(['fcstd'])
    poly.field_data['cad.backend'] = np.array(['ocp'])
    if 'color' in info:
        r, g, b = info['color']
        alpha = 1.0 - float(info.get('transparency', 0.0))
        poly.field_data['cad.color'] = np.array([[r, g, b, alpha]])
    return label


def _tessellate_brep(
    zf: zipfile.ZipFile,
    brep_name: str,
    tmp: Path,
    obj_name: str,
    *,
    linear_deflection: float,
    angular_deflection: float,
) -> pv.PolyData:
    """Extract ``brep_name`` from ``zf`` and tessellate it.

    Wraps OCCT / OS errors in :class:`CadReadError` carrying the
    FreeCAD object name, so the caller never has to guess which part
    blew up.
    """
    from OCP.BRep import BRep_Builder
    from OCP.BRepTools import BRepTools
    from OCP.TopoDS import TopoDS_Shape

    from pyvista_cad._bridges.topods import from_topods

    target = tmp / Path(brep_name).name
    try:
        with zf.open(brep_name) as src, open(target, 'wb') as dst:
            dst.write(src.read())
    except (OSError, KeyError) as exc:
        msg = f"failed to extract BREP '{brep_name}' for FreeCAD object '{obj_name}': {exc}"
        raise CadReadError(msg) from exc

    shape = TopoDS_Shape()
    builder = BRep_Builder()
    try:
        ok = BRepTools.Read_s(shape, str(target), builder)
    except (RuntimeError, OSError, AttributeError) as exc:
        msg = f"OCP failed to read BREP '{brep_name}' for FreeCAD object '{obj_name}': {exc}"
        raise CadReadError(msg) from exc
    if not ok or shape.IsNull():
        msg = (
            f"OCP could not parse BREP '{brep_name}' for FreeCAD object '{obj_name}' (null shape)"
        )
        raise CadReadError(msg)

    return from_topods(
        shape,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
    )


def _build_tree(
    objects: dict[str, dict[str, Any]],
    zf: zipfile.ZipFile,
    tmp: Path,
    *,
    linear_deflection: float,
    angular_deflection: float,
) -> pv.MultiBlock:
    """Walk the FreeCAD group hierarchy and produce a nested MultiBlock.

    Roots are objects that no other object claims as a child. Group-typed
    objects become nested :class:`pyvista.MultiBlock` containers; leaf
    objects with a BREP file become :class:`pyvista.PolyData` blocks.
    Objects with neither a BREP nor children are skipped.
    """
    # Compute roots (not referenced as anyone's child).
    referenced: set[str] = set()
    for info in objects.values():
        for child in info.get('children', []):
            referenced.add(child)
    roots = [name for name in objects if name not in referenced]

    visiting: set[str] = set()

    def visit(name: str) -> pv.DataSet | None:
        if name in visiting or name not in objects:
            return None
        visiting.add(name)
        try:
            info = objects[name]
            is_group = (
                info.get('type', '') in _GROUP_TYPES or bool(info.get('children'))
            ) and not info.get('brep_file')

            if is_group:
                mb = pv.MultiBlock()
                for child in info.get('children', []):
                    sub = visit(child)
                    if sub is not None:
                        label = (
                            objects.get(child, {}).get('label', child)
                            if child in objects
                            else child
                        )
                        mb.append(sub, name=label)
                if mb.n_blocks == 0:
                    return None
                mb.field_data['cad.label'] = np.array([info.get('label', name)])
                mb.field_data['cad.source_format'] = np.array(['fcstd'])
                return mb

            brep_name = info.get('brep_file')
            if not brep_name or brep_name not in zf.namelist():
                return None
            poly = _tessellate_brep(
                zf,
                brep_name,
                tmp,
                name,
                linear_deflection=linear_deflection,
                angular_deflection=angular_deflection,
            )
            _stamp_part(poly, info, name)
            return poly
        finally:
            visiting.discard(name)

    out = pv.MultiBlock()
    for root_name in roots:
        node = visit(root_name)
        if node is not None:
            label = objects[root_name].get('label', root_name)
            out.append(node, name=label)
    return out


def read_fcstd_internal(
    path: str | os.PathLike[str],
    *,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
) -> pv.MultiBlock:
    _require_ocp()

    p = Path(path)
    if not p.exists():
        msg = f'FCStd file not found: {path}'
        raise CadReadError(msg)

    try:
        with zipfile.ZipFile(p) as zf:
            names = zf.namelist()
            if 'Document.xml' not in names:
                msg = f'not a FreeCAD FCStd archive (no Document.xml): {path}'
                raise CadReadError(msg)
            with zf.open('Document.xml') as fh:
                doc_xml = fh.read()
            objects = _parse_document_xml(doc_xml)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                tree = _build_tree(
                    objects,
                    zf,
                    tmp,
                    linear_deflection=linear_deflection,
                    angular_deflection=angular_deflection,
                )

                # If no hierarchy was emitted (e.g. flat document where
                # every object is its own root with no group wrapper),
                # fall back to a flat one-block-per-part layout so
                # back-compat for existing fixtures holds.
                if tree.n_blocks == 0:
                    flat = pv.MultiBlock()
                    for obj_name, info in objects.items():
                        brep_name = info.get('brep_file')
                        if not brep_name or brep_name not in names:
                            continue
                        poly = _tessellate_brep(
                            zf,
                            brep_name,
                            tmp,
                            obj_name,
                            linear_deflection=linear_deflection,
                            angular_deflection=angular_deflection,
                        )
                        label = _stamp_part(poly, info, obj_name)
                        flat.append(poly, name=label)
                    tree = flat

    except zipfile.BadZipFile as exc:
        msg = f'FCStd is not a valid zip archive: {path}'
        raise CadReadError(msg) from exc

    tree.field_data['cad.source_format'] = np.array(['fcstd'])
    tree.field_data['cad.backend'] = np.array(['ocp'])
    return tree
