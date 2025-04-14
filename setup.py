from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pycolmap',
    version='0.1.0',
    author='SperidLabs',
    author_email='contact@speridlabs.com',
    description='Utilities for working with COLMAP reconstructions',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/speridlabs/pycolmap',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.18.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',  # Add pytest for development testing
        ]
    },
)
