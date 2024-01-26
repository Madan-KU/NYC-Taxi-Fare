import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

PKG_NAME = "NYCTaxiFarePredictor"
USER_NAME = ""
PROJECT_NAME = "NYC-Taxi-Fare-Predictor"

setuptools.setup(
    name=f"{PKG_NAME}",
    version="0.0.1",
    # author=USER_NAME,
    author_email="",
    description="NYC Taxi Fare Predictor",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url=f"https://github.com/{USER_NAME}/{PROJECT_NAME}",
    # project_urls={
    #     "Bug Tracker": f"https://github.com/{USER_NAME}/{PROJECT_NAME}/issues",
    # },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "mlflow",
        "fastapi",
        "feast",
        "flask",
        "dvc",
        "pandas",
        "numpy",
        "scikit-learn",
        "pyarrow",
        "redis",
    ],
)