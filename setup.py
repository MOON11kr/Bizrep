from setuptools import setup

setup(
    name="bizrep",
    version="0.1",
    author="Karishma M Patel",
    author_email="karishma2p@gmail.com",
    description="A Python library for predictive analytics and business forecasting.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MOON11kr/Bizrep",
    py_modules=["bizrep"],  # Specify the single Python file
    install_requires=[
        "pandas",
        "scikit-learn",
        "prophet",
        "statsmodels",
        "reportlab",
        "openpyxl",
        "plotly",
        "dash",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
