import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="alegnn", # Replace with your own username
    version="0.4.0",
    author="Damian Owerko",
    author_email="owerko@seas.upenn.edu",
    description="A PyTorch library for graph neural networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/damowerko/graph-neural-networks",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)