"""Error types raised by pyvista-cad."""


class CadError(Exception):
    """Base class for all pyvista-cad errors.

    All errors raised by pyvista-cad inherit from this class so that
    downstream code can catch any CAD-related failure with a single
    ``except`` clause.

    Examples
    --------
    >>> from pyvista_cad import CadError
    >>> issubclass(CadError, Exception)
    True

    """


class CadReadError(CadError):
    """Raised when a CAD file cannot be read.

    Wraps backend-specific read failures so that callers can catch a
    single, package-defined exception type instead of every backend's
    bespoke error class.

    Examples
    --------
    >>> from pyvista_cad import CadError, CadReadError
    >>> issubclass(CadReadError, CadError)
    True

    """


class CadWriteError(CadError):
    """Raised when a CAD file cannot be written.

    Wraps backend-specific write failures so that callers can catch a
    single, package-defined exception type instead of every backend's
    bespoke error class.

    Examples
    --------
    >>> from pyvista_cad import CadError, CadWriteError
    >>> issubclass(CadWriteError, CadError)
    True

    """


class UnsupportedFormatError(CadError):
    """Raised when a requested format is not supported.

    Either because the format is outside the v1 matrix or because the
    optional extra that provides the backend is not installed.

    Examples
    --------
    >>> from pyvista_cad import CadError, UnsupportedFormatError
    >>> issubclass(UnsupportedFormatError, CadError)
    True

    """


class OptionalDependencyError(CadError, ImportError):
    """Raised when an optional backend is needed but not installed.

    Carries a pip install hint in the error message so the user knows
    which extra to install.

    Examples
    --------
    >>> from pyvista_cad import CadError, OptionalDependencyError
    >>> issubclass(OptionalDependencyError, CadError)
    True

    """


class TessellationError(CadError):
    """Raised when OCCT tessellation fails for a given shape.

    Examples
    --------
    >>> from pyvista_cad import CadError, TessellationError
    >>> issubclass(TessellationError, CadError)
    True

    """


class CadCacheError(CadError):
    """Raised when a cached originating ``TopoDS_Shape`` is corrupt.

    A mesh that carries a B-rep cache entry but whose entry fails to
    deserialize or validate raises this rather than masquerading as a
    mesh with no CAD origin. A genuinely absent backend or absent cache
    entry does not raise this.

    Examples
    --------
    >>> from pyvista_cad import CadCacheError, CadError
    >>> issubclass(CadCacheError, CadError)
    True

    """


class MetadataError(CadError):
    """Raised when a ``cad.*`` metadata value fails validation.

    Examples
    --------
    >>> from pyvista_cad import CadError, MetadataError
    >>> issubclass(MetadataError, CadError)
    True

    """


class CadWarning(UserWarning):
    """Category for pyvista-cad data or feature-loss warnings.

    Emitted when a backend reads or writes a valid file but a piece of
    authored data (color, material, geometry, text, units) cannot be
    carried across and is dropped or degraded. Subclasses
    :class:`UserWarning` so ``pytest.warns(UserWarning)`` and the stdlib
    warning filters still match.

    Examples
    --------
    >>> from pyvista_cad import CadWarning
    >>> issubclass(CadWarning, UserWarning)
    True

    """
