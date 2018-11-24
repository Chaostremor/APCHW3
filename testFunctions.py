#!/usr/bin/env python3

import unittest
import numpy as np
import numpy.testing as npt

import functions as F

class TestFunctions(unittest.TestCase):
    # passed after fixing bugs in newton.py and functions.py 
    def test_ApproxJacobian1(self):
        slope = 3.0
        # Yes, you can define a function inside a function/method. And
        # it has scope only within the method within which it's
        # defined (unless you return it to the outside world, which
        # you can do in Python with no need for anything like C's
        # malloc() or C++'s new() )
        def f(x):
            return slope * x + 5.0

        x0 = 2.0
        dx = 1.e-3
        Df_x = F.approximateJacobian(f, x0, dx)
        # self.assertEqual(Df_x.shape, (1,1)
        # If x and f are scalar-valued, Df_x should be, too
        self.assertTrue(np.isscalar(Df_x))
        self.assertAlmostEqual(Df_x, slope)

    # passed after fixing bugs in newton.py and functions.py 
    def test_ApproxJacobian2(self):
        # numpy matrices can also be initialized with strings. The
        # semicolon separates rows; spaces (or commas) delimit entries
        # within a row.
        A = np.matrix("1.0 2.0; 3.0 4.0")

        def f(x):
            # The * operator for numpy matrices is overloaded to mean
            # matrix-multiplication, rather than elementwise
            # multiplication as it does for numpy arrays
            return A * x

        # The vector-valued function f defined above is the following:
        # if we let u = f(x), then
        #
        # u1 = x1 + 2 x2
        # u2 = 3 x1 + 4 x2
        #
        # The Jacobian of this function is constant and exactly equal
        # to the matrix A. approximateJacobian should thus return
        # something pretty close to A.

        x0 = np.matrix("5.0; 6.0")
        dx = 1.e-6
        Df_x = F.approximateJacobian(f, x0, dx)

        # Make sure approximateJA
        self.assertEqual(Df_x.shape, (2,2))
        # numpy arrays and matrices vectorize comparisons. So if a & b
        # are arrays, the expression a==b will itself be an array of
        # booleans. But an array of booleans does not itself evaluate
        # to a clean boolean (this is an exception to the general
        # Python rule that "every object can be interpreted as a
        # boolean"), so normal assert statements will break. We need
        # array-specific assert statements found in numpy.testing
        npt.assert_array_almost_equal(Df_x, A)

    def test_ApproxJacobian3(self):
        f = lambda x : x**2 - 7*x + 10
        #g = F.Polynomial([10, -7, 1])
        dx = 1.e-10
        for x0 in np.linspace(-2,2,11):
            Df_x = F.approximateJacobian(f, x0, dx)
            self.assertTrue(np.isscalar(Df_x))
            self.assertAlmostEqual(Df_x, (2*x0-7), places=4)

    def test_JacobianMethod(self):
        f = F.Polynomial([-15, 23, -9, 1])
        dx = 1.e-6
        method = "next"
        x0 = 2.5
        #print(x0)
        Df_x = F.approximateJacobian(f, x0, dx, method)
        self.assertTrue(np.isscalar(Df_x))
        self.assertAlmostEqual(Df_x, (3*x0**2-18*x0+23), places=5)

        method = "former"
        Df_x = F.approximateJacobian(f, x0, dx, method)
        self.assertTrue(np.isscalar(Df_x))
        self.assertAlmostEqual(Df_x, (3*x0**2-18*x0+23), places=5)

        method = "middle"
        Df_x = F.approximateJacobian(f, x0, dx, method)
        self.assertTrue(np.isscalar(Df_x))
        self.assertAlmostEqual(Df_x, (3*x0**2-18*x0+23), places=7)

    # passed after fixing bugs in newton.py and functions.py 
    def test_Polynomial(self):
        # p(x) = x^2 + 5x + 4
        p = F.Polynomial([4, 5, 1])
        # linspace(a, b, N) produces N equally spaced values from a to
        # b, including both a & b.
        for x in np.linspace(-2,2,11):
            self.assertAlmostEqual(p(x), 4 + 5*x + x**2)
        
        # It would have been more Pythonic to feed the whole array x into p
        # (the way the Polynomial class is written, it's agnostic
        # about whether x is a scalar or an array) and compare with an
        # npt assertion, as follows:
        #
        # p = F.Polynomial([4, 5, 1])
        # npt.assert_array_almost_equal(p(x), 4 + 5*x + x**2)
        #
        # But I though this would be more legible.
        

if __name__ == '__main__':
    unittest.main()



