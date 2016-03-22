__author__ = 'HyNguyen'

import time
import numpy as np

if __name__ == "__main__":
    start = time.time()
    A = np.load("data_processed.npy")
    end = time.time()
    print "load data ",end - start



