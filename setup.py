import os
from setuptools import setup


###################################################################

NAME = "NebulaBayes"
KEYWORDS = ["astronomy", "Bayesian statistics"]
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    # "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Science/Research",
    "Natural Language :: English",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Astronomy",
]
INSTALL_REQUIRES = [  # Required packages to install NebulaBayes
    "astropy >= 1.1.0",  # Need Table -> DataFrame conversion feature
                         # (https://github.com/astropy/astropy/pull/3504)
    "matplotlib",
    "numpy",
    "pandas",
    "scipy",
]

###################################################################

HERE = os.path.abspath(os.path.dirname(__file__))
VERSION_FILE = os.path.join(HERE, "src", "NebulaBayes", "_version.py")
__version__ = None  # Value replaced on next line; this keeps linter happy
exec(open(VERSION_FILE).read())  # Defines __version__
README_FILE = os.path.join(HERE, "README.txt")
LONG_DESCRIPTION = open(README_FILE).read()


if __name__ == "__main__":
    setup(
        name=NAME,
        description="Compare observed emission line fluxes to predictions",
        license="MIT",
        # url=find_meta("uri"),
        version=__version__,
        author="Adam D. Thomas",
        author_email="adam.thomas@anu.edu.au",
        # maintainer="Adam D. Thomas",
        # maintainer_email="adam.thomas@anu.edu.au",
        keywords=KEYWORDS,
        long_description=LONG_DESCRIPTION,
        packages=["NebulaBayes"],
        package_dir={"": "src"},
        package_data={"NebulaBayes": [
                        "grids/*", "docs/*", "tests/*.py",
                        "tests/run_tests.sh",
                        "tests/test_outputs/test_outputs_go_here"]},
                        # Last file is a hack to include the test_outputs dir,
                        # but no other files in this dir
        include_package_data=True,
        zip_safe=False,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
    )
