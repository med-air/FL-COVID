import numpy as np
import setuptools
from setuptools.extension import Extension

extensions = [
    Extension(
        'fl_covid.utils.compute_overlap',
        ['fl_covid/utils/compute_overlap.pyx'],
        include_dirs=[np.get_include()]
    ),
    Extension(
        'fl_covid.bin.image_formatter',
        ['fl_covid/bin/image_formatter.pyx'],
        include_dirs=[np.get_include()]
    ),
]

setuptools.setup(
    name='fl_covid-detector',
    version='0.1.0',
    description='covid object detection.',
    packages=setuptools.find_packages(),
    install_requires=['tensorflow', 'scipy', 'cython', 'Pillow', 'opencv-python', 'progressbar2', 'tqdm'],
    entry_points={
        'console_scripts': [
            'detector-train=fl_covid.bin.train_fed:main',
            'detector-convert-model=fl_covid.bin.convert_model:main',
        ],
    },
    ext_modules=extensions,
    setup_requires=["cython>=0.28"]
)
