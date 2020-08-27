from setuptools import setup, find_packages

setup(
    name='beetect',
    version='1.0',
    packages=['beetect'],
    python_requires='>=3.6',
    install_requires=[
        'Pillow',
        'opencv-python',
        'imgaug',
        'pandas',
        'easydict',
        'numpy==1.18.0', # tf 2.3.0
        'scipy==1.4.1', # tf 2.3.0
        'matplotlib',
        'Cython',
        'pycocotools',
        'tqdm',
        # says imgaug 0.4.0 is incompatible BUT requires imgaug 0.4.0
        # or else it throws import error for imgaug.albumentations
        'albumentations',
        'torch',
        'torchvision',
        'tensorflow==2.3.0',
        'tensorflow_addons',
        'tensorboard>=2.2.1',
        # 'horovod', # manual installation required, see master README
        'gdown',
    ],
)
