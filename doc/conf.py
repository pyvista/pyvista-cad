"""Sphinx + Sphinx Gallery configuration for pyvista-cad."""

import datetime
from importlib.metadata import version as get_version
import os
from pathlib import Path

import pyvista
from pyvista.plotting.utilities.sphinx_gallery import DynamicScraper

pyvista.set_error_output_file('errors.txt')
pyvista.OFF_SCREEN = True
pyvista.set_plot_theme('document')
pyvista.global_theme.window_size = [1024, 768]
pyvista.BUILDING_GALLERY = True
os.environ['PYVISTA_BUILDING_GALLERY'] = 'true'

# Headless rendering is provided by the environment, not here: CI uses
# pyvista/setup-headless-display-action (xvfb), and OFF_SCREEN is set
# above. This mirrors how the other pyvista-org doc builds are wired.

project = 'pyvista-cad'
year = datetime.datetime.now(datetime.UTC).year
copyright = f'2026-{year}, The PyVista Developers'  # noqa: A001
author = 'The PyVista Developers'
try:
    release = get_version('pyvista-cad').removesuffix('+dirty')
except Exception:
    release = '0.0.0'
version = release

src_dir = Path(__file__).absolute().parent
root_dir = src_dir.parent
package_dir = root_dir / 'src' / 'pyvista_cad'

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinx_gallery.gen_gallery',
    'numpydoc',
    'notfound.extension',
    'pyvista.ext.plot_directive',
    'pyvista.ext.viewer_directive',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

autosummary_generate = True
autodoc_typehints = 'description'
add_module_names = False

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

html_theme = 'sphinx_book_theme'
html_title = f'pyvista-cad {version}'
html_logo = '_static/pyvista_logo.svg'
html_favicon = '_static/pyvista_logo.svg'

html_static_path = ['_static']
html_css_files = ['theme_overrides.css']

html_context = {
    'github_url': 'https://github.com',
    'github_user': 'pyvista',
    'github_repo': 'pyvista-cad',
    'github_version': 'main',
    'doc_path': 'doc',
}

html_theme_options = {
    'home_page_in_toc': True,
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/pyvista/pyvista-cad',
            'icon': 'fa-brands fa-github',
        },
        {
            'name': 'PyPI',
            'url': 'https://pypi.org/project/pyvista-cad/',
            'icon': 'fa-brands fa-python',
        },
    ],
    'navigation_with_keys': False,
    'path_to_docs': 'doc',
    'repository_branch': 'main',
    'repository_url': 'https://github.com/pyvista/pyvista-cad',
    'show_prev_next': True,
    'show_toc_level': 3,
    'toc_title': 'On this page',
    'use_download_button': True,
    'use_edit_page_button': True,
    'use_fullscreen_button': True,
    'use_issues_button': True,
    'use_repository_button': True,
    'use_source_button': False,
}

myst_enable_extensions = [
    'amsmath',
    'attrs_block',
    'attrs_inline',
    'colon_fence',
    'deflist',
    'dollarmath',
    'fieldlist',
    'html_admonition',
    'html_image',
    'replacements',
    'smartquotes',
    'strikethrough',
    'substitution',
    'tasklist',
]
myst_heading_anchors = 3

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pyvista': ('https://docs.pyvista.org/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

sphinx_gallery_conf = {
    'backreferences_dir': 'api/_backrefs',
    'doc_module': ('pyvista_cad', 'pyvista'),
    'examples_dirs': ['../examples/'],
    'gallery_dirs': ['examples'],
    'filename_pattern': r'\.py',
    'within_subsection_order': 'FileNameSortKey',
    'first_notebook_cell': (
        '%matplotlib inline\nfrom pyvista import set_plot_theme\nset_plot_theme("document")\n'
    ),
    'image_scrapers': (DynamicScraper(), 'matplotlib'),
    'remove_config_comments': True,
    'reset_modules_order': 'both',
    'download_all_examples': False,
    'reference_url': {
        'pyvista_cad': None,
    },
    'thumbnail_size': (640, 480),
    'show_signature': False,
    'capture_repr': ('_repr_html_', '__repr__'),
}

copybutton_prompt_text = r'>>> |\.\.\. |\$ '
copybutton_prompt_is_regexp = True

linkcheck_ignore = [
    r'https://github.com/pyvista/pyvista-cad/issues/\d+',
]
linkcheck_timeout = 15
