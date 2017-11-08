# Build and install the NebulaBayes package
# I mostly used https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/

# The MANIFEST.in file tells Distutils what files to include in the source distribution
# It incldues the grid data files and documentation.
# The "package_data" argument to setup specifies the data required to be installed
# WITH the package itself.

# Run the build for the distribution:
# Compensate for a bug in "wheel" by removing "build"
# Also clean all the build working
rm -rf build dist src/NebulaBayes.egg-info
python setup.py sdist bdist_wheel


# Now there are dirs called "build" and dist"
# The "dist" dir contains a wheel we can install:
# pip install dist/NebulaBayes-0.88-py2.py3-none-any.whl  # python 2
# pip uninstall NebulaBayes
# /Applications/anaconda/bin/pip install dist/NebulaBayes-0.88-py2.py3-none-any.whl  # python 3
# /Applications/anaconda/bin/pip uninstall NebulaBayes

# Upload to PyPI testing site (https://testpypi.python.org/)
# (need to update version in command below; use python 3 twine because we need
# a newer SSL):
/Applications/anaconda/bin/twine upload -r test dist/NebulaBayes-0.9*
# We can install from the test PyPI site with
# pip install -i https://testpypi.python.org/pypi NebulaBayes  # <- doesn't work due to old SSL
# /Applications/anaconda/bin/pip -i https://testpypi.python.org/pypi NebulaBayes  # Works!


