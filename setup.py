from setuptools import setup, find_packages

setup(
    name='ruaccent',
    version='1.0.0',
    author='Denis Petrov',
    author_email='arduino4b@gmail.com',
    description='A Russian text accentuation tool',
    license='MIT', 
    url='https://github.com/Den4ikAI/ruaccent',  
    packages=find_packages(),
    install_requires=[
        'huggingface_hub',
        'torch==1.13.1',
        'transformers',
        'sentencepiece'
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Operating System :: MacOS',
    ],
)
