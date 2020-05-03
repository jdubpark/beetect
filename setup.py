from setuptools import setup, find_packages

setup(
    name='beetect',
    version='1.0',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'Pillow',
        'opencv-python',
        'imgaug',
        'torch',
        'torchvision',
        'pandas',
        'numpy',
        'matplotlib',
        'requests'
    ],
)
