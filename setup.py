from setuptools import setup, find_packages


setup(name='parapint',
      version='0.1.0.dev',
      packages=find_packages(),
      description='Parapint: Parallel NLP algorithms',
      long_description=open('README.md').read(),
      license='Revised BSD',
      url='https://github.com/parapint/parapint',
      install_requires=['numpy>=1.13.0', 'scipy', 'pyomo', 'mpi4py'],
      zip_safe=False,
      python_requires='>=3.7'
      )
