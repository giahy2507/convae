__author__ = 'HyNguyen'
import numpy as np
import time
from gensim.models import word2vec
import pickle
import copy
import os

class WordVectors(object):
    def __init__(self, embsize, embed_matrix, word_index):
        self.embsize = embsize
        self.embed_matrix = embed_matrix
        self.word_index = word_index
        self.word_list = word_index.keys()
        self.count_null_word = 0
        self.count_exist_word = 0

    def add_wordvector_from_w2vmodel(self, word2vec, words):
        for word in words:
            try:
                vector = word2vec[word]
                if word in self.word_index.keys():
                    continue
                else:
                    self.word_index[word] = len(self.word_index.keys())
                    self.embed_matrix = np.concatenate((self.embed_matrix,vector.reshape(1,300)))
                    # print("hy")
                    # print(self.embed_matrix.shape)
                self.count_exist_word +=1
            except:
                self.count_null_word +=1
                continue

    def save_pickle(self, filename):
        with open(filename, mode="wb") as f:
            pickle.dump(self,f)

    @classmethod
    def load_pickle(cls, filename):
        if os.path.isfile(filename):
            with open(filename, mode="rb") as f:
                return pickle.load(f)
        else:
            print("no file")

    def save_text_format(self, filename):
        with open(filename, mode= "w") as f:
            if self.embed_matrix.shape[0] != len(self.word_index.keys()):
                print("co gi do sai sai")

            f.write(str(self.embed_matrix.shape[0]) + " " + str(self.embsize)+ "\n")
            print(self.embed_matrix.shape)
            for key in self.word_index.keys():
                index  = self.word_index[key]
                vector = self.embed_matrix[index].reshape(300)
                listnum = map(str, vector.tolist())
                f.write(key + " " + " ".join(listnum) + "\n")

    @classmethod
    def load(cls, filename):
        fi = open(filename,mode="r")
        dict_size, embsize = fi.readline().split()
        dict_size, embsize = int(dict_size), int(embsize)
        embed_matrix = np.zeros((dict_size+1,embsize),dtype=np.float32)
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
        embed_matrix[0] = np.mean(embed_matrix[1:],axis=0,dtype=np.float32)
        return WordVectors(embsize,embed_matrix,word_index)

    def wordvector(self, word):
        if word in self.word_list:
            self.count_exist_word +=1
            return self.embed_matrix[self.word_index[word]]
        else:
            #Null word
            self.count_null_word +=1
            return self.embed_matrix[0]

    def get_vector_addtion(self, words):
        result_vec = copy.deepcopy(self.wordvector(words[0]))
        for i in range(1,len(words)):
            result_vec += self.wordvector(words[i])
        return result_vec

    def cae_prepare_data_from_string(self, sentence, min_length=10,  max_length=100):
        sentence = sentence.replace("\n","")
        elements = sentence.split()
        sentence_matrix = np.array([self.wordvector(word) for word in elements])
        padding = np.zeros((5,self.embsize),dtype=float)
        if sentence_matrix.shape[0] < max_length and sentence_matrix.shape[0] > min_length:
            sentence_matrix = np.concatenate((sentence_matrix,np.zeros((max_length-sentence_matrix.shape[0],self.embsize))))
        else:
            print(sentence)
            return None
        sentence_matrix_final = np.concatenate((padding,sentence_matrix,padding))
        return sentence_matrix_final

    def cae_prepare_data_from_words(self, words, min_length=10, max_length=100):
        sentence_matrix = np.array([self.wordvector(word) for word in words])
        padding = np.zeros((5,self.embsize),dtype=np.float32)
        if sentence_matrix.shape[0] <= max_length and sentence_matrix.shape[0] >= min_length:
            sentence_matrix = np.concatenate((sentence_matrix,np.zeros((max_length-sentence_matrix.shape[0],self.embsize))))
        else:
            # print(" ".join(words))
            return None
        sentence_matrix_final = np.concatenate((padding,sentence_matrix,padding))
        return sentence_matrix_final

if __name__ == "__main__":

    wordvector = WordVectors.load("../model/wordvector.txt")

    # w2v = word2vec.Word2Vec.load_word2vec_format("/Users/HyNguyen/Documents/Research/Data/GoogleNews-vectors-negative300.bin",binary=True)
    #
    # for key in wordvector.word_index.keys():
    #     if key == "UNK":
    #         continue
    #     A = wordvector.wordvector(key).reshape(300)
    #     B = w2v[key].reshape(300)
    #     # print A.shape
    #     # print A.dtype
    #     # print B.shape
    #     # print B.dtype
    #
    #     if np.array_equal(A,B) is False:
    #         print(key)



