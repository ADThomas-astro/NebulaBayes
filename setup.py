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
INSTALL_REQUIRES = [  # Required packages to use NebulaBayes
    "astropy",
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



if __name__ == "__main__":
    setup(
        name=NAME,
        description="Compare",
        license="MIT",
        # url=find_meta("uri"),
        version=__version__,
        author="Adam D. Thomas",
        author_email="adam.thomas@anu.edu.au",
        # maintainer="Adam D. Thomas",
        # maintainer_email="adam.thomas@anu.edu.au",
        keywords=KEYWORDS,
        long_description="FILL ME IN",
        packages=["NebulaBayes"],
        package_dir={"": "src"},
        package_data={"NebulaBayes": ["grids/*", "docs/*"]},
        include_package_data=True,
        zip_safe=False,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
    )
