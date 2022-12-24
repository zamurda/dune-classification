from setuptools import setup

setup(
    name='projectpkg',
    version='0.0.1',
    packages=['dctools', 'csvtools'],
    install_requires=[
        'numpy>=1.24.0',
        'pandas>=1.5.0',
        'matplotlib>=3.6'
    ],
    python_requires='>=3.10'
)