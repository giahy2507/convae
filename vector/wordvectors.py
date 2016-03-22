__author__ = 'HyNguyen'

import numpy as np
import time
class WordVectors(object):
    def __init__(self, embsize, embed_matrix, word_index):
        self.embsize = embsize
        self.embed_matrix = embed_matrix
        self.word_index = word_index
        self.word_list = word_index.keys()
        self.count_null_word = 0
        self.count_exist_word = 0

    @classmethod
    def load(cls, filename):
        fi = open(filename,mode="r")
        dict_size, embsize = fi.readline().split()
        dict_size, embsize = int(dict_size), int(embsize)
        embed_matrix = np.zeros((dict_size+1,embsize),dtype=float)
        word_index = {"UNK":0}
        counter = 1
        for i in range(1,dict_size+1,1):
            counter +=1
            if counter % 10000 == 0:
                print("Process wordvector line: ", counter)
            elements = fi.readline().split()
            word = elements[0]
            vector = np.array(elements[1:]).reshape((1,embsize))
            word_index[word] = i
            embed_matrix[i] = vector
        fi.close()
        embed_matrix[0] = np.mean(embed_matrix[1:],axis=0)
        return WordVectors(embsize,embed_matrix,word_index)

    def wordvector(self, word):
        if word in self.word_list:
            self.count_exist_word +=1
            return self.embed_matrix[self.word_index[word]]
        else:
            #Null word
            self.count_null_word +=1
            return self.embed_matrix[0]

    def cae_prepare_data(self, sentence, min_length,  max_length):
        sentence = sentence.replace("\n","")
        elements = sentence.split()
        if len(elements) > max_length or len(elements) < min_length:
            print sentence, " " ,len(elements)
            return None
        sentence_matrix = np.array([self.wordvector(word) for word in elements])
        if sentence_matrix.shape[0] < max_length:
            sentence_matrix = np.concatenate((sentence_matrix,np.zeros((max_length-sentence_matrix.shape[0],self.embsize))))
        else:
            print(sentence)
            return None
        return sentence_matrix

if __name__ == "__main__":

    w2v = WordVectors.load("100")
    print len(w2v.word_index.keys())
    fi = open("../datatrain.txt", mode="r")
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
        sentence_matrix = w2v.cae_prepare_data(line,min_length=10,max_length=100)
        if sentence_matrix != None:
            data_matrix.append(sentence_matrix)
    data_matrix = np.array(data_matrix)
    np.save("data_processed",data_matrix)
    print data_matrix.shape
    print "Number null word: ", w2v.count_null_word
    print "Number exist word: ", w2v.count_exist_word
    fi.close()





