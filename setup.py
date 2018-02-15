from setuptools import setup

setup(
    name='tftables',
    version='1.1.2',
    url='https://github.com/ghcollin/tftables',
    description='Interface for reading HDF5 files into Tensorflow.',
    long_description=open("README.rst").read(),
    keywords='tensorflow HDF5',
    license='MIT',
    author='ghcollin',
    author_email='',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    py_modules=['tftables'],
    install_requires=['multitables', 'numpy!=1.10.1', 'tensorflow']
)