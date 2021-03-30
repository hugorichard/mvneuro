from setuptools import setup, find_packages

setup(
    name="mvneuro",
    description="A multiview package for neuroscience",
    version="0.0.1",
    keywords="",
    packages=find_packages(),
    python_requires=">=3",
    install_requires=['numpy', 'sklearn', 'fastsrm', 'multiviewica']
)
