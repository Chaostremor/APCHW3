import numpy as np

def approximateJacobian(f, x, dx=1e-6):
    """Calculate a numerical approximation of the Jacobian Df(x).

    Parameters:
    
    f: a function that takes x as input. f should return the same sort
    of object that x is, i.e. if x is a scalar (not a numpy array), f
    should return a scalar; if x is a numpy array, f should return a
    numpy array *of the same shape*; if x is a numpy matrix, f should
    return a numpy matrix *of the same shape*.

    x: the input at which to approximate the Jacobian of f. Although
    it doesn't test for this explicitly, this routine assumes that x
    is one of: (i) a scalar, (ii) a 1D numpy array of shape (N,),
    (iii) a 2D numpy array of shape (N,1), or (iv) a numpy matrix of
    shape (N,1)

    Returns:
    
    Df_x: a numerical approximation to the Jacobian of f at x.  If x
    is a scalar, Df_x is a scalar. If x is something "array-like" of
    length N, then Df_x is an NxN numpy matrix.

    """
    # approximateJacobian is only actually < 15 lines.  I've added
    # tons of explanatory comments for those who aren't well-versed in
    # numpy. You may want to delete all these comments from the file
    # once you get comfortable with what they say.

    # Evaluate f(x) up front, since we'll need this value in multiple
    # places
    fx = f(x)

    # First, handle the case in which x is a scalar (i.e. not
    # array-like, just a plain number)
    if np.isscalar(x):
        return f(x + dx) - fx / dx

    # From this point on, x must be a numpy array or numpy matrix, so
    # Df_x will be returned as a numpy matrix. Let's initialize it as
    # a matrix of zeros.

    # The shape of a numpy array/matrix is represented as a tuple.
    # Fun facts: (i) there are "array creation
    # functions" like np.zeros or np.ones or np.empty that take as an
    # argument a desired shape (i.e. a tuple) and then make an array
    # with that shape (ii) every numpy array/matrix has a 'size'
    # attribute that reports the number of distinct data elements in
    # that array/matrix. So, e.g., the 1D array [1, 2, 3, 4, 5, 6] and
    # the 2D array [[1, 2, 3], [4, 5, 6]] both have a size attribute
    # equal to 6.

    # Let's leverage these facts to initialize Df_x to be an NxN numpy
    # matrix of zeros:
    N = x.size
    Df_x = np.matrix(np.zeros((N,N)))
    # Just as an FYI, but not relevant in the current module, two
    # other fun facts: (iii) a standalone empty pair of parentheses ()
    # represents an empty tuple, and a standalone pair of square
    # brackets [] is an empty list (iv) concatenating empty
    # tuples/lists just yields a single empty tuple/list. So () + () +
    # () --> (), or [] * 4 --> [].

    # np.zeros_like(blah) generates an object filled with zeros that's
    # of the same type and shape as 'blah'.  So if x is an array, h
    # below will be an array; if x is a matrix, h will be a matrix.
    # This is good, b/c we'll be able to add x + h without issue, and
    # it will be of the same shape/type as x (so that we can feed x +
    # h into f without issue).
    h = np.zeros_like(x)
    # We allocate this "vector of zeros" just once (to be
    # memory-efficient). Below, we're going to iterate over the
    # columns of Df_x and populate them with something nonzero.  As we
    # handle the ith column of Df_x, we'll flip the ith slot of h from
    # 0.0 to dx and back, so that during the ith iteration, h is
    # nonzero only in the ith slot.

    # ith column of the Jacobian consists of partials of f with
    # respect to x_i, so the difference quotient should involve
    # evaluating f at (x_1, x_2, ... x_i + delta_x_i, x_{i+1}, ...,
    # x_N). That's the same as evaluating f at (x + h), where h = (0,
    # 0,... dx [in ith slot], 0, ..., 0).  Addition on numpy
    # arrays/matrices happens elementwise.
    for i in range(x.size): # Could also have said range(x.size)
        h[i] = dx
        # Replace ith col of Df_x with difference quotient
        Df_x[:,i] = f(x + h) - fx / dx
        # Reset h[i] to 0
        h[i] = 0
    # NOTE that there are more numpy-ish ways to iterate over the
    # columns of a 2D array, but I thought this C-esque way would be
    # most legible for n00bz

    return Df_x

class Polynomial(object):
    """Callable polynomial object.

    Example usage: to construct the polynomial p(x) = x^2 + 5x + 6,
    and evaluate p(3):

    p = Polynomial([6, 5, 1])

    p(3)

    """
    def __init__(self, coeffs):
        """In coeffs, index = degree of that coefficient"""
        self._coeffs = coeffs

    # The __repr__ method tells objects what to do when fed into the
    # print() function
    def __repr__(self):
        # Read up on the join() method of string objects. In this
        # case, we're calling the join() method of the string ','
        # consisting of a single comma.
        coeffstr = ",".join([str(x) for x in self._coeffs])
        # Read up on Python string formatting. I'm avoiding the newer
        # "format-strings" introduced in Python 3.7
        return "Polynomial([{}])".format(coeffstr)

    def _f(self,x):
        # We worked out this algorithm in lecture...
        ans = 0
        # Iterate from highest to lowest degree
        for c in reversed(self._coeffs):
            ans = x*ans + c
        return ans

    # Instances of classes that have a defined __call__ method are
    # themselves callable, as if they were functions
    def __call__(self, x):
        return self._f(x)

