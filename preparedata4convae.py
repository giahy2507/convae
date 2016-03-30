__author__ = 'HyNguyen'

from vector.wordvectors import WordVectors
import time
import numpy as np


if __name__ == "__main__":
    w2v = WordVectors.load("vector/100")
    print len(w2v.word_index.keys())
    fi = open("data/mini_corpus.txt", mode="r")
    data_matrix = []
    start = time.clock()
    lines = fi.readlines()
    end = time.clock()
    print "Readlines: ", end - start
    counter = 0
    for line in lines:
        counter +=1
        if counter % 100 == 0:
            print("Process sentence line: ", counter)
        sentence_matrix = w2v.cae_prepare_data(line, min_length=10, max_length=100)
        if sentence_matrix is not None:
            data_matrix.append(sentence_matrix)
    data_matrix = np.array(data_matrix)
    np.save("data_processed",data_matrix)
    print data_matrix.shape
    print "Number null word: ", w2v.count_null_word
    print "Number exist word: ", w2v.count_exist_word
    fi.close()
