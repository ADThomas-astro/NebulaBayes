
# Some notes
# The MANIFEST.in file tells Distutils what files to include in the source distribution
# It incldues the grid data files and documentation.
# 




rm -rf build  # Compensate for a bug in "wheel"; clean the build dir
python setup.py sdist bdist_wheel


# Now there are dirs called "build" and dist"
# pip install dist/NebulaBayes-0.88-py2.py3-none-any.whl