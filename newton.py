'''newton.py

Implementation of a Newton-Raphson root-finder.

'''

import numpy as np
import functions as F
import warnings   # import class of warnings

class Newton(object):
    """Newton objects have a solve() method for finding roots of f(x)
    using Newton's method. x and f can both be vector-valued.

    """
    
    def __init__(self, f, tol=1.e-6, maxiter=20, dx=1.e-6, Df=None, max_radius=None):
        """Parameters:
        
        f: the function whose roots we seek. Can be scalar- or
        vector-valued. If the latter, must return an object of the
        same "shape" as its input x

        tol, maxiter: iterate until ||f(x)|| < tol or until you
        perform maxiter iterations, whichever happens first

        dx: step size for computing approximate Jacobian

        Df: the analytic Jacobians if specified by the user 

        max_radius: a radius r around the initial guess x0 with
        which the computed root must lie if specified by the user

        """
        self._f = f
        self._tol = tol
        self._maxiter = maxiter
        self._dx = dx
        self._Df = Df
        self._max_radius = max_radius

    def solve(self, x0):
        """Determine a solution of f(x) = 0, using Newton's method, starting
        from initial guess x0. x0 must be scalar or 1D-array-like. If
        x0 is scalar, return a scalar result; if x0 is 1D, return a 1D
        numpy array.

        """
        # NOTE: no need to check whether x0 is scalar or vector. All
        # the functions/methods invoked inside solve() return "the
        # right thing" when x0 is scalar.
        x = x0
        
        for i in range(self._maxiter):
            fx = self._f(x)

            # raise exception when root x is out of maximum radius bound to x0 with iteration
            if  (not self._max_radius is None) and (np.linalg.norm(x-x0) > self._max_radius):
                raise Exception("Root x is out of maximum radius bound to x0 with iteration !")

            # linalg.norm works fine on scalar inputs
            if np.linalg.norm(fx) < self._tol:
                return x
            x = self.step(x, fx)
        return x

        # raise exception when reaching maxiter with no correct root
        niter = i + 1
        fx = self._f(x)
        if (niter == self._maxiter) and (np.linalg.norm(fx) >= self._tol):
            raise Exception("Reaching maxiter with no correct root !")
        
        

    def step(self, x, fx=None):
        """Take a single step of a Newton method, starting from x. If the
        argument fx is provided, assumes fx = f(x).

        """
        if fx is None:
            fx = self._f(x)

        if self._Df is None:
            Df_x = F.approximateJacobian(self._f, x, self._dx)
        else:
            Df_x = self._Df(x)

        # linalg.solve(A,B) returns the matrix solution to AX = B, so
        # it gives (A^{-1}) B. np.matrix() promotes scalars to 1x1
        # matrices.
        h = np.linalg.solve(np.matrix(Df_x), np.matrix(fx))
        # Suppose x was a scalar. At this point, h is a 1x1 matrix. If
        # we want to return a scalar value for our next guess, we need
        # to re-scalarize h before combining it with our previous
        # x. The function np.asscalar() will act on a numpy array or
        # matrix that has only a single data element inside and return
        # that element as a scalar.

        # raise warnings when h is too big, might be abnormal
        thres = 1.e6
        if np.linalg.norm(h) > thres:
            warnings.warn("One step within iteration is too big, check x0 or f(x) for recalculation is recommended !")
        
        if np.isscalar(x):
            h = np.asscalar(h)

        return x - h
        # instead of the original one: return x + h
