__author__ = 'HyNguyen'

import numpy as np
import os
import xml.etree.ElementTree as ET
import nltk

if __name__ == "__main__":

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
                    # wordvectors.add_wordvector_from_w2vmodel(w2v,words)
            except:
                print "exception: ", cluster_name ,file_name
                continue
        # print("Finish cluster name:", cluster_name," , Wordvector size: ", str(wordvectors.embed_matrix.shape[0]))

