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
        'pandas',
        'numpy',
        'matplotlib',
        'Cython',
        'pycocotools',
        # says imgaug 0.4.0 is incompatible BUT requires imgaug 0.4.0
        # or else it throws import error for imgaug.albumentations
        'albumentations',
        'torch',
        'torchvision',
        'tensorflow',
        'tensorboard',
    ],
)
