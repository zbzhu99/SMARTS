import pathlib

import setuptools

setuptools.setup(
    name="dreamerv2",
    version="2.1.1.a",
    description="Mastering Atari with Discrete World Models",
    url="http://github.com/danijar/dreamerv2",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    packages=["dreamerv2"],
    package_data={"dreamerv2": ["configs.yaml"]},
    entry_points={"console_scripts": ["dreamerv2=dreamerv2.train:main"]},
    install_requires=[
        "gym",
        "ruamel.yaml",
        "tensorflow==2.4.0",
        "tensorflow_probability==0.12.2",
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
