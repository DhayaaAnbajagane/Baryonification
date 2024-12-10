# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../../'))

def run_apidoc(_):
    import os
    import sys
    import subprocess

    subprocess.check_call(['cp', '-r', '../../../latest', '../../../BaryonForge'])
    
    cmd_path = 'sphinx-apidoc'
    if hasattr(sys, 'real_prefix'):  # Check to see if we are in a virtualenv
        # If we are, assemble the path manually
        cmd_path = os.path.abspath(os.path.join(sys.prefix, 'bin', 'sphinx-apidoc'))
    
    for module in ['Profiles', 'Runners', 'utils']:
        subprocess.check_call([cmd_path, '--force', '--ext-autodoc', '--ext-intersphinx', 
                            '-e', '-o', '../../docs/source', '../../../BaryonForge/BaryonForge/' + module, 
                            '../../../BaryonForge/*setup*'])
        
    subprocess.check_call([cmd_path, '--force', '--ext-autodoc', '--ext-intersphinx', 
                        '-e', '-o', '../../docs/source', '../../../BaryonForge/BaryonForge/', '../../../BaryonForge/*setup*'])


def setup(app):
    app.connect('builder-inited', run_apidoc)

# -- Project information

project = 'BaryonForge'
copyright = '2024, Anbajagane'
author = 'Dhayaa Anbajagane'

release = '1.0'
version = '1.0.0'

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

autodoc_member_order = "bysource"

templates_path = ['_templates']

# -- Options for HTML output
html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'