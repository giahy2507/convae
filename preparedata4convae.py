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
import matplotlib.pyplot as plt



def statistic_freq():

    wordvectors = WordVectors.load("model/wordvector.txt")

    freq_array = [0] * 500

    # Penn Tree Bank
    treebank_sents = treebank.sents()
    for i in range(len(treebank_sents)):
        senttmp = " ".join(treebank_sents[i])
        words = nltk.word_tokenize(senttmp)
        freq_array[len(words)] +=1

    # Brown
    brown_sents = brown.sents()
    for i in range(len(brown_sents)):
        senttmp = " ".join(brown_sents[i])
        words = nltk.word_tokenize(senttmp)
        freq_array[len(words)] +=1

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
                    freq_array[len(words)] +=1
            except:
                print "exception parse XML: ", file_name
                continue
        print("Finish cluster name:", cluster_name," , Wordvector size: ", str(wordvectors.embed_matrix.shape[0]))

    plt.plot(range(200), freq_array[:200], color='red', marker='.')
    plt.show()

def collect_data_from_ptb_brow_duc2004():

    start_collect = time.time()
    samples = []
    # Penn Tree Bank
    treebank_sents = treebank.sents()
    for i in range(len(treebank_sents)):
        senttmp = " ".join(treebank_sents[i])
        words = nltk.word_tokenize(senttmp)
        samples.append(words)
    print("Finish collecting training data from Penn Tree Bank")

    # Brown
    brown_sents = brown.sents()
    for i in range(len(brown_sents)):
        senttmp = " ".join(brown_sents[i])
        words = nltk.word_tokenize(senttmp)
        samples.append(words)
    print("Finish collecting training data from Brown")

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
                    samples.append(words)
            except:
                print "exception parse XML: ", file_name
                continue
    print("Finish collecting training data from DUC2004")
    print("length of samples", len(samples))
    end_collect = time.time()
    print("Total time for collecting training data: " + str(end_collect - start_collect))
    return samples

if __name__ == "__main__":

    wordvectors = WordVectors.load("model/wordvector.txt")
    train_data =  collect_data_from_ptb_brow_duc2004()
    final_array = []
    for i, words in enumerate(train_data):
        words_array = wordvectors.cae_prepare_data_from_words(words, 10, 100)
        final_array.append(words_array)
        if i == 69:
            break
    final_array = np.array(final_array)
    print(final_array.shape)




