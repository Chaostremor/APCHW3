#!/usr/bin/env python3

import unittest
import numpy as np

import newton

class TestNewton(unittest.TestCase):
    # passed after fixing bugs in newton.py and functions.py     
    def testLinear(self):
        # Just so you see it at least once, this is the lambda keyword
        # in Python, which allows you to create anonymous functions
        # "on the fly". As I commented in testFunctions.py, you can
        # define regular functions inside other
        # functions/methods. lambda expressions are just syntactic
        # sugar for that.  In other words, the line below is
        # *completely equivalent* under the hood to:
        #
        # def f(x):
        #     return 3.0*x + 6.0
        #
        # No difference.
        f = lambda x : 3.0*x + 6.0

        # Setting maxiter to 2 b/c we're guessing the actual root
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        x = solver.solve(2.0)
        # Equality should be exact if we supply *the* root, ergo
        # assertEqual rather than assertAlmostEqual
        self.assertEqual(x, -2.0)
        
    def testQuard1(self):
        f = lambda x : x**2 - 7*x + 10
        solver = newton.Newton(f, tol=1.e-15, maxiter=10)
        x = solver.solve(100)
#        y = solver.solve(4)
#        print(x)
#        print(y)
#        self.assertAlmostEqual(x, 2.0)
#        self.assertAlmostEqual(y, 5.0)
        self.assertTrue((np.isclose(x, 2.0)) or (np.isclose(x, 5.0)))


if __name__ == "__main__":
    unittest.main()

    
