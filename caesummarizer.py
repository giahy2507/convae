__author__ = 'HyNguyen'
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
from summaryobject import *
from summary import summary as smr
from vector.wordvectors import WordVectors
from convae import ConvolutionAutoEncoder
import cPickle
import os

class CAESummarizer(object):
    def __init__(self, cae_model, word_vector_model, mode = 0):
        self.cae = cae_model
        self.wordvector = word_vector_model
        self.mode = mode

    @classmethod
    def create_my_summarizer(cls, cae_model_path , word_vector_model_path = "vector/100", mode = 0):
        word_vectors = WordVectors.load(word_vector_model_path)
        convae = ConvolutionAutoEncoder.rebuild_for_testing(mini_batch_size=1, filemodel=cae_model_path)
        return CAESummarizer(convae, word_vectors, mode)

    @classmethod
    def summary(self, cluster, max_word, mode="sub_cosine"):
        """
        ----------
        Params
            cluster:
            cluster:
        :return:
        """
        summary_sentences = []
        V = []
        P = []
        L = np.array([])
        k = 0
        if cluster !=None:
            for document in cluster.list_documents:
                p = []
                for sentence in document.list_sentences:
                    if sentence.vector is None:
                        continue
                    p.append(k)
                    sentence.sentece_id = k
                    k = k + 1
                    V.append(sentence.vector)
                    L = np.append(L,sentence.length)
                P.append(p)
            alpha = 0.7
            galma = 0.3
            n = len(V)
            numberofWord = max_word
            mode = mode
            summarize = smr.do_summarize(V, n, P, L, alpha, galma, numberofWord, mode)
            print (summarize)
            word_count = 0
            for document in cluster.list_documents:
                for sentence in document.list_sentences:
                    if sentence.sentece_id in summarize:
                        word_count += sentence.length
                        if word_count > max_word:
                            word_count -= sentence.length
                            continue
                        cluster.my_summarys.append(sentence.content)
        return cluster.my_summarys


def generate_system(clusters , path_to_model , path_to_system, mode="sub_cosine"):
    groups = [85,130,180,220,270,340]
    counter = 0
    for group in groups:
        path_to_group_model = path_to_model + "/" + str(group)
        for file_name in os.listdir(path_to_group_model):
            counter +=1
            cluster_id, _, _, _ = file_name.split(".")
            _ , cluster_id = cluster_id.split("_")
            number_count = 0
            if file_name.find("ref1") != -1:
                number_count = clusters[int(cluster_id)].list_references[0].word_count
            elif file_name.find("ref2") != -1:
                number_count = clusters[int(cluster_id)].list_references[1].word_count
            summary_text = CAESummarizer.summary(clusters[int(cluster_id)], number_count, mode=mode)
            path_to_system_group = path_to_system + "/" + str(group)
            if not os.path.exists(path_to_system_group):
                os.makedirs(path_to_system_group)
            print "summary ", counter , "\""+file_name+"\"", "couting_word: ",number_count
            fo = open(path_to_system_group + "/" +file_name,'w')
            fo.writelines(summary_text)
            fo.close()
        print "finished group ", group

def create_summary_format_vn():
    print("create summary format")
    # vietnamesemds_path = "data/vietnamesemds/"
    # caesummarizer = CAESummarizer.create_my_summarizer("model/CAE.model","vector/100")
    #
    # clusters = [None]* 201
    # counter = 1
    # for cluster_id in os.listdir(vietnamesemds_path):
    #     _, id = cluster_id.split("_")
    #     cluster = Cluster.load_from_folder(cluster_id, vietnamesemds_path + cluster_id + "/")
    #     print ("Cluster ", counter)
    #     counter+=1
    #     for document in cluster.list_documents:
    #         for sentence in document.list_sentences:
    #             sentence_matrix = caesummarizer.wordvector.cae_prepare_data(sentence.content)
    #             if sentence_matrix is None:
    #                 sentence.vector = None
    #                 continue
    #             sentence_vector = caesummarizer.cae.get_vector_function(sentence_matrix)
    #             sentence.vector = sentence_vector.T
    #     clusters[int(id)] = cluster
    #
    # with open("data/vietnamesemds.pikcle", mode="wb") as f:
    #     cPickle.dump(clusters, f)
    with open("data/vietnamesemds.pickle", mode="rb") as f:
        clusters = cPickle.load(f)

    generate_system(clusters, "data/VietnameseMDS-grouped/model", "data/VietnameseMDS-grouped/system", mode="sub_euclid")

# modeList = {"sub_cosine":0, "sub_euclid":1,"mmr_cosine":2,"mmr_euclid":3,"kmean_simple":4,
#                 "mmr_kmean_cosine":5,"mmr_kmean_euclid":6,"mmr_pagerank_cosine":7,
#                 "mmr_pagerank_euclid":8}


def create_summary_format_duc2004(ducpath, wordvectors_path, summary_path):
    wordvectors = WordVectors.load(wordvectors_path)
    clusters = []
    for cluster_id in os.listdir(ducpath):
        if cluster_id[0] == ".":
            continue
        cluster = Cluster.load_from_folder_duc(cluster_id,ducpath+ "/"+cluster_id,wordvectors)
        summary = CAESummarizer.summary(cluster,100)
        file_summary = summary_path + "/" + cluster_id[:-1].upper()+".M.100.T.1"
        with open(file_summary, mode="w") as f:
            for line in summary:
                f.write(line + "\n")
        clusters.append(cluster)
        print("Finish loading cluster_id: ", cluster_id)
    return clusters

def create_summary_format_opinosis(opinosis_path, wordvectors_path, summary_path):
    wordvectors = WordVectors.load(wordvectors_path)
    clusters = []
    for cluster_id in os.listdir(opinosis_path):
        if cluster_id[0] == ".":
            continue
        cluster = Cluster.load_from_opinosis(cluster_id,opinosis_path+"/"+cluster_id, wordvectors)
        summary = CAESummarizer.summary(cluster,25,"kmean_simple")
        if len(summary) == 0:
            print("ttdt")
        cluster_id,_,_ = cluster_id.split(".")
        folder_summary = summary_path+"/"+cluster_id
        if not os.path.isdir(folder_summary):
            os.makedirs(folder_summary)
        file_summary = folder_summary+"/"+cluster_id+".1.txt"
        with open(file_summary, mode="w") as f:
            for line in summary:
                f.write(line + "\n")
        clusters.append(cluster)
        print("Finish loading cluster_id: ", folder_summary)
    return clusters


if __name__ == "__main__":

    # ducpath = "/Users/HyNguyen/Documents/Research/Data/duc2004/DUC2004_Summarization_Documents/duc2004_testdata/tasks1and2/duc2004_tasks1and2_docs/docs"
    # wordvectors_path = "model/wordvector.txt"
    # summary_path = "data/peer"
    # clusters = create_summary_format_duc2004(ducpath, wordvectors_path, summary_path)
    # with open("data/duc.sumobj.pickle", mode="wb") as f:
    #     cPickle.dump(clusters,f)

    opinosis_path = "/Users/HyNguyen/Documents/Research/Data/OpinosisDataset1.0_0/topics"
    wordvectors_path = "model/wordvector.txt"
    summary_path = "data/peer"
    clusters = create_summary_format_opinosis(opinosis_path,wordvectors_path,summary_path)



