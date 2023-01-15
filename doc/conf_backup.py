# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
import pydata_sphinx_theme

sys.path.append("scripts")
from gallery_directive import GalleryDirective

#sys.path.insert(0, os.path.abspath('../../'))
#sys.path.append(os.path.abspath('sphinxext'))

# -- Project information -----------------------------------------------------

project = u'pygpc'
copyright = u'2022, Konstantin Weise'
author = u'Konstantin Weise'

# The short X.Y version
version = u'0.3'

# The full version, including alpha/beta/rc tags
release = u'0.3.3'


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '5.3'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# extensions = [
#     'sphinx.ext.autodoc',
#     'sphinx.ext.mathjax',
#     'sphinx.ext.viewcode',
#     'sphinx.ext.githubpages',
#     'sphinx.ext.napoleon',
#     # 'sphinx.ext.imgmath',
#     'matplotlib.sphinxext.plot_directive',
#     'sphinx_gallery.gen_gallery',
#     'sphinx.ext.intersphinx',
#     'sphinx.ext.autosectionlabel'
# ]

extensions = [
    "sphinx.ext.autodoc",
    'sphinx.ext.mathjax',
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinxext.rediraffe",
    "sphinx_design",
    "sphinx_copybutton",
    # For extension examples and demos
    "ablog",
    "jupyter_sphinx",
    "matplotlib.sphinxext.plot_directive",
    "myst_nb",
    # "nbsphinx",  # Uncomment and comment-out MyST-NB for local testing purposes.
    "numpydoc",
    "sphinx_togglebutton",
    'sphinx.ext.autosectionlabel',
    'sphinxcontrib.napoleon'
]

# configuration of sphinx gallery
sphinx_gallery_conf = {
    'examples_dirs': ['examples/introduction', 'examples/gpc', 'examples/algorithms',
                      'examples/features', 'examples/examples', 'examples/sampling'],   # path to your example scripts
    'gallery_dirs': ['auto_introduction', 'auto_gpc', 'auto_algorithms', 'auto_features', 'auto_examples', 'auto_sampling'],
    'default_thumb_file': 'examples/images/pygpc_logo_square.png',
    'remove_config_comments': True # path to where to save gallery generated output

} #'default_thumb_file': '../../../pckg/media/pygpc_logo_git.png',

# sphinx_gallery_conf = {
#     'gallery_dirs': ['auto_introduction', 'auto_gpc', 'auto_algorithms', 'auto_features', 'auto_examples', 'auto_sampling'],
#     'default_thumb_file': 'examples/images/pygpc_logo_square.png',
#     'remove_config_comments': True # path to where to save gallery generated output
#
# } #'default_thumb_file': '../../../pckg/media/pygpc_logo_git.png',

# 'IPython.sphinxext.ipython_directive',
# 'IPython.sphinxext.ipython_console_highlighting',
# 'sphinx.ext.intersphinx',

# # Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# -- Internationalization ------------------------------------------------
# specifying the natural language populates some key tags
language = "en"

# ReadTheDocs has its own way of generating sitemaps, etc.
if not os.environ.get("READTHEDOCS"):
    extensions += ["sphinx_sitemap"]

    # -- Sitemap -------------------------------------------------------------
    html_baseurl = os.environ.get("SITEMAP_URL_BASE", "http://127.0.0.1:8000/")
    sitemap_locales = [None]
    sitemap_url_scheme = "{link}"

autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
exclude_patterns = ["_build"]

# -- Extension options -------------------------------------------------------

# This allows us to use ::: to denote directives, useful for admonitions
myst_enable_extensions = ["colon_fence", "linkify", "substitution"]


# html_sidebars = {
#     "**": ["sidebar-nav-bs", "sidebar-ethical-ads"]
# }

# Define the json_url for our version switcher.
json_url = "https://pydata-sphinx-theme.readthedocs.io/en/latest/_static/switcher.json"

# Define the version we use for matching in the version switcher.
version_match = os.environ.get("READTHEDOCS_VERSION")
# If READTHEDOCS_VERSION doesn't exist, we're not on RTD
# If it is an integer, we're in a PR build and the version isn't correct.
if not version_match or version_match.isdigit():
    # For local development, infer the version to match from the package.
    release = pydata_sphinx_theme.__version__
    if "dev" in release or "rc" in release:
        version_match = "latest"
        # We want to keep the relative reference if we are in dev mode
        # but we want the whole url if we are effectively in a released version
        json_url = "_static/switcher.json"
    else:
        version_match = "v" + release

# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme' # 'alabaster'# 'scipy' #'sphinx_rtd_theme'
html_theme_path = ['_theme']
html_static_path = ["_static"]
html_sourcelink_suffix = ""

html_theme_options = {
"github_url": "https://github.com/pygpc-polynomial-chaos/pygpc",
"twitter_url": "https://twitter.com/k_weise_",
"header_links_before_dropdown": 4,
"icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pygpc/",
            "icon": "fa-solid fa-box",
        },
        {
            "name": "pygpc",
            "url": "https://github.com/pygpc-polynomial-chaos/pygpc",
            "icon": "_static/logo.png",
            "type": "local",
            "attributes": {"target": "_blank"},
        },
    ],
    "logo": {
        "text": "pygpc",
        "image_dark": "logo.png",
        "alt_text": "pygpc",
    },
    "use_edit_page_button": True,
    "show_toc_level": 1,
    "navbar_align": "left",  # [left, content, right] For testing that the navbar items align properly
    "navbar_center": ["version-switcher", "navbar-nav"],
    "announcement": "https://raw.githubusercontent.com/pydata/pydata-sphinx-theme/main/docs/_templates/custom-template.html",
    # "show_nav_level": 2,
    # "navbar_start": ["navbar-logo"],
    # "navbar_end": ["theme-switcher", "navbar-icon-links"],
    # "navbar_persistent": ["search-button"],
    # "primary_sidebar_end": ["custom-template.html", "sidebar-ethical-ads.html"],
    # "footer_items": ["copyright", "sphinx-version"],
    # "secondary_sidebar_items": ["page-toc.html"],  # Remove the source buttons
    "switcher": {
        "json_url": json_url,
        "version_match": version_match,
    },
}

html_sidebars = {
    "community/index": [
        "sidebar-nav-bs",
        "custom-template",
    ],  # This ensures we test for custom sidebars
    "examples/no-sidebar": [],  # Test what page looks like with no sidebar items
    "examples/persistent-search-field": ["search-field"],
    # Blog sidebars
    # ref: https://ablog.readthedocs.io/manual/ablog-configuration-options/#blog-sidebars
    "examples/blog/*": [
        "postcard.html",
        "recentposts.html",
        "tagcloud.html",
        "categories.html",
        "authors.html",
        "languages.html",
        "locations.html",
        "archives.html",
    ],
}

myst_heading_anchors = 2
myst_substitutions = {"rtd": "[Read the Docs](https://readthedocs.org/)"}

html_context = {
    "github_user": "pygpc",
    "github_repo": "pygpc",
    "github_version": "main",
    "doc_path": "docs",
}

rediraffe_redirects = {
    "contributing.rst": "community/index.rst",
}

# ABlog configuration
# blog_path = "examples/blog/index"
# blog_authors = {
#     "pydata": ("PyData", "https://pydata.org"),
#     "jupyter": ("Jupyter", "https://jupyter.org"),
# }


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]
todo_include_todos = True


def setup(app):
    # Add the gallery directive
    app.add_directive("gallery-grid", GalleryDirective)

# old theme options:
# html_theme_options = {
#     "edit_link": "true",
#     "sidebar": "right",
#     "pygpc_logo": "true",
#     "rootlinks": [],
#     "body_max_width": "1500"
# }

# pngmath_latex_preamble = r"""
# \usepackage{color}
# \definecolor{textgray}{RGB}{51,51,51}
# \color{textgray}
# """
# pngmath_use_preview = True
# pngmath_dvipng_args = ['-gamma 1.5', '-D 96', '-bg Transparent']
# html_short_title = 'pygpc'
#
# # Output file base name for HTML help builder.
# htmlhelp_basename = 'pygpc_doc'
#
# # The name of an image file (relative to this directory) to place at the top
# # of the sidebar.
#
# # The name of an image file (relative to this directory) to use as a favicon of
# # the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# # pixels large.
# #html_favicon = None
#
# # -- Options for HTMLHelp output ---------------------------------------------
#
#
# # -- Options for LaTeX output ------------------------------------------------
#
# latex_elements = {
#     # The paper size ('letterpaper' or 'a4paper').
#     #
#     # 'papersize': 'a4paper',
#
#     # The font size ('10pt', '11pt' or '12pt').
#     #
#     # 'pointsize': '10pt',
#
#     # Additional stuff for the LaTeX preamble.
#     #
#     # 'preamble': '',
#
#     # Latex figure (float) alignment
#     #
#     # 'figure_align': 'htbp',
#     'extraclassoptions': 'openany,oneside',
#     'babel': '\\usepackage[shorthands=off]{babel}'
# }
#
# # Grouping the document tree into LaTeX files. List of tuples
# # (source start file, target name, title,
# #  author, documentclass [howto, manual, or own class]).
#
# # latex_documents = [
#     # (master_doc, 'pygpc.tex', u'pygpc Documentation',
#      # u'Konstantin Weise, Benjamin Kalloch, Lucas Possner', 'manual'),
# # ]
#
# latex_documents = [
#     (master_doc, 'pygpc.tex', u'Documentation of the pygpc package',
#      u'Konstantin Weise', 'manual'),
# ]
#
#
# # -- Options for manual page output ------------------------------------------
#
# # One entry per manual page. List of tuples
# # (source start file, name, description, authors, manual section).
#
# # man_pages = [
#     # (master_doc, 'pygpc', u'pygpc Documentation',
#      # [author], 1)
# # ]
#
# man_pages = [
#     (master_doc, 'pygpc', u'Documentation of the pygpc package',
#      [author], 1)
# ]
#
# # -- Options for Texinfo output ----------------------------------------------
#
# # Grouping the document tree into Texinfo files. List of tuples
# # (source start file, target name, title, author,
# #  dir menu entry, description, category)
# texinfo_documents = [
#     (master_doc, 'pygpc', u'Documentation of the pygpc package',
#      author, 'pygpc', 'One line description of project.',
#      'Miscellaneous'),
# ]
#
#
# # -- Options for Epub output -------------------------------------------------
#
# # Bibliographic Dublin Core info.
# epub_title = project
#
# # The unique identifier of the text. This can be a ISBN number
# # or the project homepage.
# #
# # epub_identifier = ''
#
# # A unique identification for the text.
# #
# # epub_uid = ''
#
# # A list of files that should not be packed into the epub file.
# epub_exclude_files = ['search.html']
#
#
# # -- Extension configuration -------------------------------------------------
#
# # -- Options for intersphinx extension ---------------------------------------
#
# # Example configuration for intersphinx: refer to the Python standard library.
# intersphinx_mapping = {'algorithms': ('/examples/algorithms/', None)}
