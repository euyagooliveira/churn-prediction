from setuptools import find_packages, setup

setup(
    name="churn_prediction",
    version="0.1.0",
    description="End-to-end customer churn prediction pipeline",
    author="Yago Oliveira",
    author_email="euyagooliveira@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "scikit-learn>=1.4",
        "xgboost>=2.0",
        "lightgbm>=4.0",
        "imbalanced-learn>=0.12",
        "shap>=0.45",
        "matplotlib>=3.7",
        "seaborn>=0.13",
    ],
    entry_points={
        "console_scripts": [
            "churn-train=models.train:main",
        ],
    },
)
