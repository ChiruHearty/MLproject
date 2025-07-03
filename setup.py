from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'  # Editable install marker to remove from requirements

def get_requirements(file_path: str) -> List[str]:
    '''
    Reads requirements.txt and returns list of dependencies,
    cleaning newline chars and removing '-e .' if present.
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Strip newline characters
        requirements = [req.replace("\n", "") for req in requirements]

        # Remove editable installs marker if present
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

# Main setup function to configure the package
setup(
    name='mlproject',             # Package name
    version='0.0.1',              # Initial version
    author='chiru',               # Author name
    author_email='chirustar84412@gmail.com',  # Author email
    packages=find_packages(),     # Automatically find all packages/subpackages
    install_requires=get_requirements('requirements.txt')  # Dependencies
)

# setup.py Packaging Script
# Purpose:
# Defines a Python package setup script that reads dependencies from requirements.txt and configures metadata for packaging and distribution.
#
# Key Points:
# - Removes editable install entry -e . if present.
# - Automatically discovers packages using find_packages().
# - Specifies author info, version, and dependencies.
# - Useful for pip-installable or deployable ML projects.
#  