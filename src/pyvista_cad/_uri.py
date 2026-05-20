"""URI gate for reader entry points.

PyVista's :func:`pyvista.read` auto-downloads remote URIs to a temp file
and retries — but only when the custom reader raises
:class:`pyvista.core.utilities.reader_registry.LocalFileRequiredError`.
Every reader registered through ``pyvista.readers`` / ``pv.register_reader``
in this package must call :func:`require_local_path` before touching the
filesystem so that ``http(s)://`` (and other scheme) inputs round-trip
through ``_download_uri`` instead of dying with ``FileNotFoundError``.
"""

import os

from pyvista.core.utilities.reader_registry import (
    LocalFileRequiredError,
    has_scheme,
)


def require_local_path(path: 'str | os.PathLike[str]') -> None:
    """Raise ``LocalFileRequiredError`` if *path* is a URI."""
    if isinstance(path, str) and has_scheme(path):
        raise LocalFileRequiredError(path)
