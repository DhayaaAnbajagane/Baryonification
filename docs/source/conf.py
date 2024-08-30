# Configuration file for the Sphinx documentation builder.

import os
import subprocess

def run_apidoc(_):
    source_dir = os.path.abspath('.')
    package_dir = os.path.abspath('../your_python_package')
    output_dir = os.path.join(source_dir, 'source')
    cmd_path = 'sphinx-apidoc'
    subprocess.check_call([cmd_path, '-o', output_dir, package_dir, '--force'])

def setup(app):
    app.connect('builder-inited', run_apidoc)
    
# -- Project information

project = 'Baryonification'
copyright = '2024, Anbajagane'
author = 'Dhayaa Anbajagane'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',  # optional: for Google-style and NumPy-style docstrings
    'sphinx.ext.viewcode',  # optional: to include links to the source code
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'