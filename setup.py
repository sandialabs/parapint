from setuptools import setup, find_packages


setup(name='parapint',
      version='0.2.0',
      packages=find_packages(),
      description='Parapint: Parallel NLP algorithms',
      long_description=open('README.md').read(),
      long_description_content_type="text/markdown",
      author='Parapint Developers',
      maintainer_email='mlbynum@sandia.gov',
      license='Revised BSD',
      url='https://github.com/parapint/parapint',
      install_requires=['numpy>=1.13.0', 'scipy', 'pyomo>5.7.1', 'pyutilib', 'mpi4py'],
      include_package_data=True,
      python_requires='>=3.6',
      classifiers=["Programming Language :: Python :: 3",
                   "Programming Language :: Python :: 3.6",
                   "Programming Language :: Python :: 3.7",
                   "Programming Language :: Python :: 3.8",
                   "License :: OSI Approved :: BSD License",
                   "Operating System :: OS Independent"]
      )
