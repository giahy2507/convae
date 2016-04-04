__author__ = 'HyNguyen'

from vector.wordvectors import WordVectors
import time
import numpy as np

from gensim.models import word2vec
from nltk.corpus import brown
from nltk.corpus import treebank
import nltk
import xml.etree.ElementTree as ET
import os

from nltk.corpus import stopwords
import re

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":


    # Load Word2Vec from Google
    w2v = word2vec.Word2Vec.load_word2vec_format("/Users/HyNguyen/Documents/Research/Data/GoogleNews-vectors-negative300.bin",binary=True)

    # stops = set(stopwords.words("english"))
    # Create object WordVectors

    wordvectors = WordVectors(300,{},{})

    # Penn Tree Bank
    treebank_sents = treebank.sents()
    for i in range(len(treebank_sents)):
        senttmp = " ".join(treebank_sents[i])
        words = nltk.word_tokenize(senttmp)
        wordvectors.add_wordvector_from_w2vmodel(w2v,words)
    print("Finish penn tree bank corpus, Wordvector size: ", len(wordvectors.embed_matrix.keys()))



    # Brown
    brown_sents = brown.sents()
    for i in range(len(brown_sents)):
        senttmp = " ".join(brown_sents[i])
        words = nltk.word_tokenize(senttmp)
        wordvectors.add_wordvector_from_w2vmodel(w2v,words)
    print("Finish brow corpus, Wordvector size: ", len(wordvectors.embed_matrix.keys()))


    # DUC data
    folder_path = "/Users/HyNguyen/Documents/Research/Data/DUC20042005/duc2004/DUC2004_Summarization_Documents/duc2004_testdata/tasks1and2/duc2004_tasks1and2_docs/docs"
    clusters_name = os.listdir(folder_path)
    for cluster_name in clusters_name:
        if cluster_name[0] == ".":
            # except file .DStore in my macbook
            continue
        files_name = os.listdir(folder_path + "/" + cluster_name)
        for file_name in files_name:
            if file_name[0] == ".":
                # except file .DStore in my macbook
                continue
            file_path = folder_path + "/" + cluster_name +"/"+ file_name
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                text_tag = root._children[3]
                if text_tag.tag == "TEXT":
                    text = text_tag.text.replace("\n", "")
                sentences = nltk.tokenize.sent_tokenize(text)
                for sentence in sentences:
                    words = nltk.word_tokenize(sentence)
                    wordvectors.add_wordvector_from_w2vmodel(w2v,words)
            except:
                print "exception: ", file_name
                continue
        print("Finish cluster name:", cluster_name," , Wordvector size: ", len(wordvectors.embed_matrix.keys()))

    wordvectors.save("model/wordvector.pickle")
    """

    # prepare data for vietnamese convae

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
    """
