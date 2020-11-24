from abc import ABC, abstractmethod
import logging


class LinearSolverInterface(ABC):
    """
    This is the base class for linear solvers that work with the
    interior point algorithm. Derived classes must implement the
    following abstract methods:

      * do_symbolic_factorization
      * do_numeric_factorization
      * do_back_solve
      * get_inertia
    """
    @classmethod
    def getLoggerName(cls):
        return 'linear_solver'

    @classmethod
    def getLogger(cls):
        name = 'algorithms.' + cls.getLoggerName()
        return logging.getLogger(name)

    @abstractmethod
    def do_symbolic_factorization(self, matrix, raise_on_error=True, timer=None):
        """
        Perform symbolic factorization with the nonzero structure of the matrix.
        """
        pass

    @abstractmethod
    def do_numeric_factorization(self, matrix, raise_on_error=True, timer=None):
        """
        Factorize the matrix. Can only be called after do_symbolic_factorization.
        """
        pass

    def increase_memory_allocation(self, factor):
        raise NotImplementedError('Should be implemented by base class.')

    @abstractmethod
    def do_back_solve(self, rhs):
        """
        Solve the linear system matrix * x = rhs for x. Can only be called
        after do_numeric_factorization.
        """
        pass

    @abstractmethod
    def get_inertia(self):
        """
        Get the inertia of the factorized matrix. Can only be called 
        after do_numeric_factorization.
        """
        pass
