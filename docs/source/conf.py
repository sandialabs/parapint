import os
import sys
sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------

project = 'Parapint'
copyright = '2020, National Technology & Engineering Solutions of Sandia, LLC (NTESS)'
author = 'Michael Bynum and Carl Laird and Bethany Nicholson and Denis Ridzal'
release = '0.1.0.dev'

# -- General configuration ---------------------------------------------------

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'numpydoc']

autodoc_member_order = 'bysource'
numpydoc_class_members_toctree = False
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
add_module_names = False

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

latex_documents = [('index', 'parapint.tex', 'Parapint', 'Michael Bynum \\and Carl Laird \\and Bethany Nicholson \\and Denis Ridzal', 'report', False)]
