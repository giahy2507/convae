__author__ = 'HyNguyen'
from vector.wordvectors import WordVectors
import time
import numpy as np

from nltk.corpus import brown
from nltk.corpus import treebank
import nltk
import xml.etree.ElementTree as ET
import os

import sys

from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def collect_data_from_ptb_brow_duc2004():

    start_collect = time.time()
    samples = []
    # Penn Tree Bank
    treebank_sents = treebank.sents()
    for i in range(len(treebank_sents)):
        senttmp = " ".join(treebank_sents[i])
        words = nltk.word_tokenize(senttmp)
        samples.append(words)

    sys.stdout.write("Finish collecting training data from Penn Tree Bank")
    sys.stdout.flush()

    # Brown
    brown_sents = brown.sents()
    for i in range(len(brown_sents)):
        senttmp = " ".join(brown_sents[i])
        words = nltk.word_tokenize(senttmp)
        samples.append(words)
    sys.stdout.write("Finish collecting training data from Brown")
    sys.stdout.flush()

    # DUC data
    folder_path = "/Users/HyNguyen/Documents/Research/Data/duc2004/DUC2004_Summarization_Documents/duc2004_testdata/tasks1and2/duc2004_tasks1and2_docs/docs"
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
    sys.stdout.write("Finish collecting training data from DUC2004")
    sys.stdout.flush()
    sys.stdout.write("length of samples" + str(len(samples)))
    sys.stdout.flush()
    end_collect = time.time()
    sys.stdout.write("Total time for collecting training data: " + str(end_collect - start_collect))
    sys.stdout.flush()
    return samples


if __name__ == "__main__":
    data_scatters = []
    start_total = 0
    if rank == 0:
        start_total = time.time()
        wordvectors = WordVectors.load("model/wordvector.txt")
        print("Finished read wordvectors ...")
        traindata = collect_data_from_ptb_brow_duc2004()
        size_sample = int(len(traindata)/size)
        for i in range(size):
            if i* size_sample + size_sample > len(traindata):
                data_scatters.append(traindata[i*size_sample:])
            else:
                data_scatters.append(traindata[i*size_sample : i*size_sample+size_sample])
    else:
        wordvectors = None
        data_scatter = None

    wordvectors = comm.bcast(wordvectors, root = 0)
    print("Process:", rank, "broadcasted wordvectors ...")
    data_scatter = comm.scatter(data_scatters,root=0)
    print("Process:", rank, "Data scatter length: ", len(data_scatter))
    # print("Process:", rank, "Data scatter [0]: ", data_scatter[0])
    # print("Process:", rank, "Data scatter [-1]: ", data_scatter[-1])

    #work with data_scatter
    final_array = []
    for i, words in enumerate(data_scatter):
        if i != 0 and i% 1000 == 0:
            print("Process:", rank, "Preparedata line ", i)
        words_array = wordvectors.cae_prepare_data_from_words(words, 10, 100)
        if words_array is not None:
            final_array.append(words_array)
    final_array = np.array(final_array)
    print("Process:", rank, "Data final array shape: ", final_array.shape)

    data_matrix_gather = comm.gather(final_array, root=0)

    if rank == 0:
        # gather and save
        print("data gather")
        data_matrix_final = data_matrix_gather[0]
        for i in range(1,len(data_matrix_gather)):
            data_matrix_final = np.concatenate((data_matrix_final,data_matrix_gather[i]))
        print("Process:", rank, "data_matrix_final.shape: ", data_matrix_final.shape)
        end_total = time.time()
        print("Process:", rank, "Total time: ", end_total - start_total, "s")
        np.save("data/data.convae", data_matrix_final)
        print("Process:", rank, "Save to data/data.convae.np ")

