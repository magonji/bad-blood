"""
Setup script for Bad Blood FTIR Spectral Analysis Tool.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    here = os.path.abspath(os.path.dirname(__file__))
    readme_path = os.path.join(here, 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Bad Blood - FTIR Spectral Analysis Tool"

# Read requirements from requirements.txt
def read_requirements():
    here = os.path.abspath(os.path.dirname(__file__))
    requirements_path = os.path.join(here, 'requirements.txt')
    requirements = []
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return requirements

setup(
    name="bad-blood",
    version="2.0.0",
    author="Mario González-Jiménez",
    author_email="mario.gonzalez-jimenez@glasgow.ac.uk",
    description="A comprehensive tool for processing FTIR spectral data from .mzz files",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/mariogj/bad-blood",  # Update with your actual repository URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'bad-blood=bad_blood_pkg.bad_blood:main',
        ],
    },
    keywords="ftir spectroscopy infrared analysis chemistry physics research data-processing",
    project_urls={
        "Bug Reports": "https://github.com/mariogj/bad-blood/issues",
        "Source": "https://github.com/mariogj/bad-blood",
        "Documentation": "https://github.com/mariogj/bad-blood/wiki",
    },
    include_package_data=True,
    zip_safe=False,
)