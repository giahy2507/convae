__author__ = 'HyNguyen'

import numpy as np
import theano.tensor as T
import theano

if __name__ == "__main__":
    A = np.array([[1,2,3], [2,3,4], [5,5,5]])
    B = np.array([[3,3,3], [1,2,3], [9,0,5]])

    X = T.dmatrix("X")
    Y = T.dmatrix("Y")
    X_Y = X - Y
    SQR = T.sqr(X_Y)
    SUM = T.sum(SQR,axis=1)

    showfunction = theano.function(inputs=[X,Y], outputs=[X_Y,SQR,SUM])

    x1, x2, x3 = showfunction(A,B)
    print x1, "\n" , x2
    print x3
    print np.square(A)
