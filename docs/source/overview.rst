Overview
========

Parapint is a package for parallel solution of dynamic optimization
problems. Parapint currently includes a Schur-Complement decomposition
algorithm based on [Word2014]_. Parapint utilizes Pynumero
`BlockVector` and `BlockMatrix` classes (which in turn utilize Numpy
arrays and Scipy sparse matrices) for efficient block-based linear
algebra operations such as block-matrix, block-vector dot
products. These classes enable convenient construction of
block-structured KKT systems. Parapint also utilizes Pynumero
interfaces to efficient numerical routines in C, C++, and Fortran,
including the AMPL Solver Library (ASL), MUMPS, and the MA27 routines
from the Harwell Subroutine Library (HSL).

Parapint is designed with three primary modules:

  * The algorithms. The algorithms drive the solution process and
    perform high level operations such as the fraction-to-the boundary
    rule or inertia correction for the interior point algorithm. The
    interior point algorithm is designed to work with any
    :py:class:`~parapint.interfaces.interface.BaseInteriorPointInterface`
    and any
    :py:class:`~parapint.linalg.base_linear_solver_interface.LinearSolverInterface`
    as long as the interface and the linear solver are compatible.
  * The interfaces. All interfaces should inherit from
    :py:class:`~parapint.interfaces.interface.BaseInteriorPointInterface`
    and implement all abstract methods. These are the methods required
    by the interior point algorithm. The interfaces are designed to work
    with a subset of linear solvers. The table below outlines which
    interfaces work with which linear solvers.
  * The linear solvers. All linear solvers should inherit from
    :py:class:`~parapint.linalg.base_linear_solver_interface.LinearSolverInterface`
    and implement all abstract methods. These are the methods required
    by the interior point algorithm. The linear solvers are designed to
    work with certain interface classes. The table below outlines which
    linear solvers work with which interfaces.

.. _table-class-compatability:
.. table:: Compatible linear solvers and interfaces

   ======================================================================================================= ==================================================================================================================
   Linear Solver                                                                                           Compatible Interface Class
   ======================================================================================================= ==================================================================================================================
   :class:`~parapint.linalg.ma27_interface.InteriorPointMA27Interface`                                     :class:`~parapint.interfaces.interface.InteriorPointInterface`
   :class:`~parapint.linalg.mumps_interface.MumpsInterface`                                                :class:`~parapint.interfaces.interface.InteriorPointInterface`
   :class:`~parapint.linalg.scipy_interface.ScipyInterface`                                                :class:`~parapint.interfaces.interface.InteriorPointInterface`
   :class:`~parapint.linalg.schur_complement.explicit_schur_complement.SchurComplementLinearSolver`        :class:`~parapint.interfaces.schur_complement.sc_ip_interface.DynamicSchurComplementInteriorPointInterface`
   :class:`~parapint.linalg.schur_complement.mpi_explicit_schur_complement.MPISchurComplementLinearSolver` :class:`~parapint.interfaces.schur_complement.mpi_sc_ip_interface.MPIDynamicSchurComplementInteriorPointInterface`
   ======================================================================================================= ==================================================================================================================

.. [Word2014] Word, D. P., Kang, J., Akesson, J., &
              Laird, C. D. (2014). Efficient parallel solution of
              large-scale nonlinear dynamic optimization
              problems. Computational Optimization and Applications,
              59(3), 667-688.
