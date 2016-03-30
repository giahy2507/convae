__author__ = 'HyNguyen'

import numpy as np
import theano.tensor as T
import theano
import cPickle
from summaryobject import *

if __name__ == "__main__":
    with open("data/vietnamesemds.pickle", mode="rb") as f:
        clusters = cPickle.load(f)

    print(clusters)
