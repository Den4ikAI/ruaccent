
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='ruaccent',
    version='1.5.8',
    author='Denis Petrov',
    author_email='arduino4b@gmail.com',
    description='A Russian text accentuation tool',
    license='Apache License 2.0',
    url='https://github.com/Den4ikAI/ruaccent',  
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Operating System :: MacOS',
    ],
    package_data={
        '': ['*.md', '*.txt'],
    },
    include_package_data=True,
)
