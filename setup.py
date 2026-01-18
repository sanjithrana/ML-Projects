#this setup file is responsible in creating my ml application as a package and we can also install this packge in our projects

from setuptools import setup,find_packages
from typing import List

hypen_e = "-e ."
def get_requirements(file_path:str)->List[str]:
    """this fun returns the list of req"""

    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if hypen_e in requirements:
            requirements.remove(hypen_e)

    return requirements

setup(
    name="ML project",
    version="0.0.1",
    author = "sanjith",
    author_email="chilupurisanjith18@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements("requirements.txt")

)