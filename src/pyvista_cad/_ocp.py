"""OCP 7.x / 8.x compatibility.

OCP 8 drops the ``_s`` suffix on the ``TopoDS`` downcasts and moves the
``TopTools`` collection typedefs to ``OCP.collections`` template names.
Import this module lazily, like any other OCP import.
"""

from typing import Any

from OCP.TopoDS import TopoDS

try:
    from OCP.collections import (
        IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher as ShapeAncestorMap,
        IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as ShapeIndexedMap,
        Map_TopoDS_Shape_TopTools_ShapeMapHasher as ShapeMap,
    )
except ImportError:  # OCP < 8
    from OCP.TopTools import (
        TopTools_IndexedDataMapOfShapeListOfShape as ShapeAncestorMap,
        TopTools_IndexedMapOfShape as ShapeIndexedMap,
        TopTools_MapOfShape as ShapeMap,
    )


def _cast(name: str) -> Any:
    return getattr(TopoDS, f'{name}_s', None) or getattr(TopoDS, name)


to_edge = _cast('Edge')
to_face = _cast('Face')
to_shell = _cast('Shell')

__all__ = ['ShapeAncestorMap', 'ShapeIndexedMap', 'ShapeMap', 'to_edge', 'to_face', 'to_shell']
