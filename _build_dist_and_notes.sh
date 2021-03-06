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
# Extract the version string from _version.py
version="$(grep __version__ src/NebulaBayes/_version.py | cut -d \" -f2)"

# Now there are dirs called "build" and dist"
# The "dist" dir contains a wheel that can be installed:
# pip install dist/NebulaBayes-${version}-py2.py3-none-any.whl  # python 2
# pip uninstall NebulaBayes
# /Applications/anaconda/bin/pip install dist/NebulaBayes-${version}-py2.py3-none-any.whl  # python 3
# /Applications/anaconda/bin/pip uninstall NebulaBayes
# Install from Github:
# python3 -m pip install git+https://github.com/ADThomas-astro/NebulaBayes.git@f81ccfb6907d81551d2a6f53407231ca817bc0f5

# Upload to PyPI testing site (https://testpypi.python.org/)
# (note that I've added a ~/.pypirc file)
twine upload -r test dist/NebulaBayes-${version}*
# We can install from the test PyPI site with
# pip install -i https://testpypi.python.org/pypi NebulaBayes

# Upload to real PyPI site (https://pypi.python.org/)
twine upload -r pypi dist/NebulaBayes-${version}*
# Can install with
# pip install NebulaBayes
