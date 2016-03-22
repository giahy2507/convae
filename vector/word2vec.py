__author__ = 'Giahy'

from gensim.models import word2vec
from gensim import utils
from numpy import array, average
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Word2VecModel:
    def __init__(self, w2vmodel):
        self.embsize = w2vmodel.vector_size
        self.model = w2vmodel
        new_dict = {}
        for key in self.model.vocab:
            new_dict[key.encode("UTF-8")] = self.model.vocab[key]
        self.model.vocab = new_dict
        # Null word = mean (all word vector)
        self.null_word =  average(self.model.syn0, axis=0)
        # count for statistic
        self.count_except = 0
        self.count_try = 0

    def save(self, filepath):
        try:
            self.model.save_word2vec_format(filepath)
            return "Successful Saving to " + filepath
        except:
            return "Failed Saving !"

    @classmethod
    def load(cls, filepath):
        try:
            if os.path.isfile(filepath):
                model = word2vec.Word2Vec.load_word2vec_format(filepath)
                return Word2VecModel(model)
            else:
                return None
        except:
            return None

    @classmethod
    def train(cls, filepath, window=5, min_count=5, workers= 4, embsize = 100):
        sentences = word2vec.LineSentence(filepath)
        model = word2vec.Word2Vec(sentences, size=embsize, window=window, min_count=min_count, workers=workers)
        return Word2VecModel(model)

    def getWordEmbeddingFromString(self, word_str):
        lower_str = word_str.lower().encode('UTF-8')
        try:
            a = array(self.model[lower_str]).T
            self.count_try +=1
            return a
        except:
            b = array(self.null_word)
            self.count_except +=1
            return b

    def parseInstanceFromSentence(self, sentence_str):
        words_str = utils.to_unicode(sentence_str).split(' ')
        return array([self.getWordEmbeddingFromString(word_str) for word_str in words_str]).T

    def parseFromCorpusSentence(self, corpus_sent):
        return array([self.getWordEmbeddingFromString(word_i) for word_i in corpus_sent]).T

if __name__ == '__main__':
    w2vmodel = Word2VecModel.train("../data/viettreebank/corpus_lower.txt",embsize=500)
    w2vmodel.save("500.txt")


