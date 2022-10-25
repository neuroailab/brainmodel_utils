from setuptools import setup, find_packages
import os

if os.path.exists("requirements.txt"):
    with open("requirements.txt", "r") as fb:
        requirements = fb.readlines()
else:
    requirements = [
        "numpy==1.18.5",
        "xarray==0.15.1",
        "scikit-learn>=0.24.2",
        "pandas==0.25.3",
        "scipy==1.5.2",
        "h5py==2.10.0",
        "black>=19.10b0",
    ]

print(find_packages())
setup(
    name="brainmodel_utils",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.6",
    # metadata to display on PyPI
    description="Brain-Model Utilities",
    # could also include long_description, download_url, etc.
)
