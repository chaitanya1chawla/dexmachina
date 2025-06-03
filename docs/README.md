# Installation 
This doc page template is built on top of the [pyroki](https://github.com/chungmin99/pyroki/tree/main/docs) repo.
To install the build package:
```
cd docs
pip install -r requirements.txt
```
Each doc page is locally edited in markdown files, which are parsed (with `myst-parser`) and built into doc page with [Sphinx](https://www.sphinx-doc.org/en/master/usage/installation.html). To update the latest html, run `make html` in this directory. 