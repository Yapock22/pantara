from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pantara",
    version="0.7.4",
    description="Matching-pursuit symbolic regression for physical laws",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Yapock22",
    url="https://github.com/Yapock22/pantara",
    license="MIT",
    packages=find_packages(),
    package_data={"pantara": ["model.pt"]},
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "torch>=1.12",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
