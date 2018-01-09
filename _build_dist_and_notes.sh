# Build, upload to PyPI, and install the NebulaBayes package
# Used https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/

# The MANIFEST.in file tells Distutils what files to include in the source distribution
# It incldues the grid data files and documentation.
# The "package_data" argument to setup specifies the data required to be installed
# WITH the package itself.

# Run the build for the distribution:
# Compensate for a bug in "wheel" by removing "build"
# Also clean all the build working
rm -rf build dist src/NebulaBayes.egg-info
python setup.py sdist bdist_wheel


version="0.9.4"  # Need to change this manually!
# Now there are dirs called "build" and dist"
# The "dist" dir contains a wheel that can be installed:
# pip install dist/NebulaBayes-${version}-py2.py3-none-any.whl  # python 2
# pip uninstall NebulaBayes
# /Applications/anaconda/bin/pip install dist/NebulaBayes-${version}-py2.py3-none-any.whl  # python 3
# /Applications/anaconda/bin/pip uninstall NebulaBayes

# Upload to PyPI testing site (https://testpypi.python.org/)
# (note that I've added a ~/.pypirc file)
# (use python 3 twine because a newer SSL is needed):
/Applications/anaconda/bin/twine upload -r test dist/NebulaBayes-${version}*
# We can install from the test PyPI site with
# pip install -i https://testpypi.python.org/pypi NebulaBayes  # <- doesn't work due to old SSL
# /Applications/anaconda/bin/pip install -i https://testpypi.python.org/pypi NebulaBayes  # Works!

# Upload to real PyPI site (https://pypi.python.org/)
/Applications/anaconda/bin/twine upload -r pypi dist/NebulaBayes-${version}*
# Can install with
# pip install NebulaBayes  # python 2
# /Applications/anaconda/bin/pip install NebulaBayes  # python 3
