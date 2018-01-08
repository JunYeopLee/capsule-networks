from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['pillow', 'scipy']

setup(
    name='capsNet',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='tensorflow implementation of capsule network'
)