"""CAD reader functions.

Each reader registers itself with PyVista's reader registry via
``@pv.register_reader('.ext')``. Importing :mod:`pyvista_cad` imports
this package, which triggers registration.
"""
