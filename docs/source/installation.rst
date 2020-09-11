Installation
============

Parapint can be installed by cloning the parapint repository from
https://github.com/parapint/parapint ::

  git clone https://github.com/parapint/parapint.git
  cd parapint/
  python setup.py install

Requirements
-------------

Parapint requires Python (at least version 3.7) and the following packages:

* Numpy (version 1.13.0 or greater)
* Scipy
* Pyomo (Parapint currently only works with the master branch of Pyomo)

Pyomo should be installed from source and used to build Pynumero extensions::

  pip install numpy
  pip install scipy
  git clone https://github.com/pyomo/pyomo.git
  cd pyomo/
  python setup.py install
  cd pyomo/contrib/pynumero/
  python build.py -DBUILD_ASL=ON -DBUILD_MA27=ON -DIPOPT_DIR=<path/to/ipopt/build/>

Pymumps also needs to be installed in order to use MUMPS::

  conda install pymumps
