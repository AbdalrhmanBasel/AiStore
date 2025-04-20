from setuptools import setup, find_packages

def read_file(file_path):
    """
    This function reads the contents of files, such 
    as README.md file. It ensures that a file is opened with correct
    encoding "utf-8".
    """
    with open(file_path, encoding="utf-8") as file:
        return file.read()
    
long_description = read_file("README.md")

setup(
    name="ais_gnn_recommender",
    version="0.1",
    description="Graph Neural Network-based recommendation system for eCommerce platforms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Abdalrhman Morida",
    author_email="abdalrhman30x@gmail.com",
    url="https://github.com/yourusername/gnn_recommender",
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.8.0", 
        "torch-geometric>=2.0.0", 
        "pandas>=1.2.0", 
        "scikit-learn>=0.24.0", 
        "matplotlib>=3.3.0", 
        "numpy>=1.20.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.50.0",
        "scipy>=1.6.0",   
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
    ],
    python_requires='>=3.7',
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/gnn_recommender/issues",
        "Documentation": "https://gnn_recommender.readthedocs.io",
        "Source Code": "https://github.com/yourusername/gnn_recommender",
    },
    zip_safe=False,
)
