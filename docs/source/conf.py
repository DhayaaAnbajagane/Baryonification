# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

print(os.path.abspath('../../'))
print(os.listdir(os.path.abspath('../../')))

def run_apidoc(_):
    import os
    import sys
    import subprocess

    cmd_path = 'sphinx-apidoc'
    if hasattr(sys, 'real_prefix'):  # Check to see if we are in a virtualenv
        # If we are, assemble the path manually
        cmd_path = os.path.abspath(os.path.join(sys.prefix, 'bin', 'sphinx-apidoc'))
    subprocess.check_call([cmd_path, '--force', '--ext-autodoc', '--ext-intersphinx', 
                           '-e', '-o', './source', '../../', '../../*setup*'])
    
    subprocess.check_call(['cat', './source/latest.Profiles.BaryonCorrection.rst'])

    subprocess.check_call(['cat', './source/latest.utils.rst'])

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