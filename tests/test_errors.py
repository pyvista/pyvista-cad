"""Exhaustive tests for the ``pyvista_cad._errors`` exception hierarchy.

Every error and warning class is instantiated, raised, caught through
its declared base, and matched on message text so the public contract
(a single ``CadError`` / ``CadWarning`` catch point) is pinned.
"""

import re
import warnings

import pytest

from pyvista_cad._errors import (
    CadCacheError,
    CadError,
    CadReadError,
    CadWarning,
    CadWriteError,
    MetadataError,
    OptionalDependencyError,
    TessellationError,
    UnsupportedFormatError,
)

_ERROR_CLASSES = [
    CadError,
    CadReadError,
    CadWriteError,
    UnsupportedFormatError,
    OptionalDependencyError,
    TessellationError,
    CadCacheError,
    MetadataError,
]


@pytest.mark.parametrize('cls', _ERROR_CLASSES)
def test_every_error_subclasses_caderror(cls: type[CadError]) -> None:
    assert issubclass(cls, CadError)
    assert issubclass(cls, Exception)


@pytest.mark.parametrize('cls', _ERROR_CLASSES)
def test_every_error_raises_and_formats(cls: type[CadError]) -> None:
    msg = f'boom from {cls.__name__}'
    with pytest.raises(cls, match=r'boom from'):
        raise cls(msg)
    # Caught through the package-wide base.
    with pytest.raises(CadError, match=re.escape(msg)):
        raise cls(msg)
    assert str(cls(msg)) == msg


def test_optional_dependency_error_is_also_importerror() -> None:
    """The optional-dependency error must satisfy ``except ImportError`` too."""
    assert issubclass(OptionalDependencyError, ImportError)
    hint = 'pip install pyvista-cad[step]'
    with pytest.raises(ImportError, match='pip install pyvista-cad'):
        raise OptionalDependencyError(hint)
    with pytest.raises(CadError):
        raise OptionalDependencyError(hint)


def test_specific_types_do_not_cross_match() -> None:
    """A read error is not a write error and vice versa."""
    assert not issubclass(CadReadError, CadWriteError)
    assert not issubclass(CadWriteError, CadReadError)
    read_msg = 'read failed'
    write_msg = 'write failed'
    with pytest.raises(CadReadError):
        raise CadReadError(read_msg)
    with pytest.raises(CadError):  # noqa: PT012
        try:
            raise CadWriteError(write_msg)
        except CadReadError:  # pragma: no cover - must not match
            pytest.fail('CadWriteError matched CadReadError')


def test_metadata_error_message_match() -> None:
    msg = "unknown units 'parsec'"
    with pytest.raises(MetadataError, match=r"unknown units 'parsec'"):
        raise MetadataError(msg)


def test_cad_warning_is_userwarning() -> None:
    assert issubclass(CadWarning, UserWarning)
    with pytest.warns(CadWarning, match='color dropped'):
        warnings.warn('color dropped on STEP write', CadWarning, stacklevel=1)
    # Also catchable as a plain UserWarning (stdlib filters / pytest.warns).
    with pytest.warns(UserWarning, match='units degraded'):
        warnings.warn('units degraded', CadWarning, stacklevel=1)


def test_cad_warning_is_a_warning_not_a_caderror() -> None:
    assert issubclass(CadWarning, Warning)
    assert not issubclass(CadWarning, CadError)
