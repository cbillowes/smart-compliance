from setuptools import find_packages
from setuptools import setup

with open("../requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='smart-compliance',
      version="0.0.1",
      description="Smart Compliance",
      license="MIT",
      author="Clarice Bouwer, Amit Malik, Vighnesh Gaya",
      url="https://smart-compliance.streamlit.app/",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      include_package_data=True,
      zip_safe=False)
