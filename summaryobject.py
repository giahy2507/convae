__author__ = 'HyNguyen'
import os
import codecs
import numpy as np
import codecs
from vector.wordvectors import WordVectors
from convae import ConvolutionAutoEncoder

class Cluster(object):
    def __init__(self, cluster_id ,list_documents, list_references):
        self.list_documents = list_documents
        self.list_references = list_references
        self.length_documents = len(list_documents)
        self.length_references = len(list_references)
        self.cluster_id = cluster_id
        self.my_summarys = []

    @classmethod
    def load_from_folder(cls, cluster_id , cluster_path):
        if os.path.exists(cluster_path):
            files_name = os.listdir(cluster_path)
            list_documents = []
            list_references = []
            for file_name in files_name:
                file_prefix = file_name.find('.body.tok.txt')
                sentences = []
                document_id = ""
                if file_prefix > 0 :
                    document_id = file_name[:file_prefix]
                    file = codecs.open(cluster_path + '/' + file_name)
                    for line in file.readlines():
                        # remove name of authors
                        if len(line) < 50:
                            continue
                        sentences.append(Sentence(line))
                    list_documents.append(Document(sentences,document_id))
                    file.close()
                elif file_name.find(".ref") != -1 and file_name.find(".tok.txt") != -1:
                    fi = codecs.open(cluster_path + '/' + file_name)
                    lines = fi.readlines()
                    sentences = [Sentence(line,None) for line in lines]
                    fi.close()
                    document_id = "ref"
                    list_references.append(Document(sentences,document_id))
            return Cluster(cluster_id,list_documents, list_references)
        else:
            return None

class Document(object):
    def __init__(self,  list_sentences , document_id = -1,):
        self.list_sentences = list_sentences
        self.document_id = document_id
        self.length = len(list_sentences)
        self.word_count = sum([sentence.length for sentence in list_sentences if isinstance(sentence, Sentence)])

class Sentence(object):
    def __init__(self, content, vector = None):
        self.content = content
        self.vector = vector
        self.length = content.count(" ")
        self.sentece_id = -1

import numpy as np
import time
import cPickle

if __name__ == "__main__":

    clusterpath = "data/vietnamesemds/cluster_1/"
    vectormodel = "model/word2vec/100"
    vietnamesemds_path = "data/vietnamesemds/"

    start = time.time()
    w2v = WordVectors.load("vector/100")
    end = time.time()

    convae = ConvolutionAutoEncoder.rebuild_for_testing(mini_batch_size=1,filemodel="model/CAE.model")
    clusters = [None]* 201

    counter = 1
    for cluster_id in os.listdir(vietnamesemds_path):
        _, id = cluster_id.split("_")
        cluster = Cluster.load_from_folder(cluster_id, vietnamesemds_path + cluster_id + "/")
        print ("Cluster ", counter)
        counter+=1
        for document in cluster.list_documents:
            for sentence in document.list_sentences:
                sentence_matrix = w2v.cae_prepare_data(sentence.content)
                if sentence_matrix is None:
                    sentence.vector = None
                    continue
                sentence_vector = convae.get_vector_function(sentence_matrix)
                sentence.vector = sentence_vector.T
        clusters[int(id)] = cluster

    with open("data/vietnamesemds.pikcle", mode="wb") as f:
        cPickle.dump(clusters, f)
