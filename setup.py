from pathlib import Path
from setuptools import setup, find_packages

parent_dir = Path(__file__).resolve().parent

setup(
    name="smart-compliance",
    version="0.1.0",
    author="Clarice Bouwer, Amit Malik, Vighnesh Gaya",
    description="Smart Compliance",
    long_description=parent_dir.joinpath(
        "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://smart-compliance.streamlit.app/",
    license="MIT",
    packages=find_packages(
        exclude=("smart_compliance/tests")
    ),
    data_files=[
        ("", ["requirements.txt"]),
    ],
    include_package_data=True,
    zip_safe=False,
    install_requires=parent_dir.joinpath("requirements.txt")
    .read_text(encoding="utf-8")
    .splitlines(),
    python_requires=">=3.6",
)
