from setuptools import setup


# The text of the README file
with open('README.md') as f:
    rm = f.read()

# This call to setup() does all the work
setup(
    name="boaf",
    version="0.0.1",
    description="Birds Of A Feather - Clustering in Python",
    long_description=rm,
    long_description_content_type="text/markdown",
    url="https://github.com/TimothyRogers/BOAF.git",
    author="Tim Rogers",
    author_email="tim.rogers@sheffield.ac.uk",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9+",
    ],
    packages=['boaf'],
    package_dir={'':'src'},
    include_package_data=False,
    install_requires=[
        "numpy"
    ],
)