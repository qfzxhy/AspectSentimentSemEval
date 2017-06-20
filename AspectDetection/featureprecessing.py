from AspectDetection import NameListGenerator as nlg
from AspectDetection import headword_generator as hwg
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from AspectDetection import wordclasses_load as wll

class_name = 200
class LoadFeature(object):
    def __init__(self,docs,raw_sents_labels,threshold1 = 0,threshold2 = 4):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.docs = docs
        self.raw_sents_labels = raw_sents_labels
        self.threshold1 = threshold1
        self.threshold2 = threshold2


    def get_word_tfidf(self,test_corpus):
        tfidf_clf = self.vectorizer.fit(self.docs)
        tf_idf = tfidf_clf.transform(test_corpus)
        return tf_idf.todense()

    def get_word_feature(self,test_corpus):
        tfidf_clf = self.vectorizer.fit(self.docs)
        tf_idf = tfidf_clf.transform(test_corpus).todense()
        compare_matrix = np.zeros(tf_idf.shape,dtype=float)
        feature_matrix = np.less(compare_matrix,tf_idf).astype(int)
        return feature_matrix

    def get_head_word_feature(self,test_corpus):
        headword_corpus = hwg.headword_corpus_generate(corpus=self.docs)
        tfidf_clf = self.vectorizer.fit(headword_corpus)
        tf_idf = tfidf_clf.transform(test_corpus).todense()
        compare_matrix = np.zeros(tf_idf.shape, dtype=float)
        feature_matrix = np.less(compare_matrix, tf_idf).astype(int)
        return feature_matrix

    def get_bigram_feature(self,test_corpus):
        bigram_map = {}
        bigram_count_map = {}
        for doc in self.docs:
            words = doc.split(' ')
            for id, word in enumerate(words):
                if id == 0:
                    continue
                if (words[id - 1] + word) not in bigram_count_map:
                    bigram_count_map[words[id - 1] + word] = 1
                else:
                    bigram_count_map[words[id - 1] + word] += 1

        for doc in self.docs:
            words = doc.split(' ')
            for id, word in enumerate(words):
                if id == 0:
                    continue
                if (words[id - 1] + word) not in bigram_map and bigram_count_map[words[id - 1] + word] > 3:
                    bigram_map[words[id - 1] + word] = len(bigram_map)
        feature_dim = len(bigram_map)
        feature_matrix = np.zeros((len(test_corpus), feature_dim))
        for j, sent in enumerate(test_corpus):
            words = sent.split(' ')
            #words = word_tokenize(words)
            feature = [0 for i in range(feature_dim)]
            for i in range(1,len(words)):
                if (words[i-1] + words[i]) in bigram_map:
                    feature[bigram_map[words[i-1] + words[i]]] = 1
            feature_matrix[j, :] = np.array(feature)
        return feature_matrix

    def get_namelist1_feature(self,test_corpus):
        name_list1,name_list2 = nlg.get_namelist(self.docs, self.raw_sents_labels, threshold1=self.threshold1,
                                                            threshold2=self.threshold2)
        feature_dim = len(name_list1)
        feature_matrix = np.zeros((len(test_corpus),feature_dim))
        for sent_id,sent in enumerate(test_corpus):
            feature = np.zeros(feature_dim)
            for i,item in enumerate(name_list1):
                if item[0] in sent:
                    feature[i] = 1
            feature_matrix[sent_id,:] = feature
        return feature_matrix
    def get_namelist2_feature(self,test_corpus):
        name_list1, name_list2 = nlg.get_namelist(self.docs, self.raw_sents_labels, threshold1=self.threshold1,
                                                  threshold2=self.threshold2)
        feature_dim = len(name_list2)
        feature_matrix = np.zeros((len(test_corpus),feature_dim))
        for sent_id,sent in enumerate(test_corpus):
            feature = np.zeros(feature_dim)
            for i,item in enumerate(name_list2):
                if item[0] in sent:
                    feature[i] = 1
            feature_matrix[sent_id,:] = feature
        return feature_matrix
    #word2vec cluster feature classes - 200
    def get_cluster_feature(self,test_corpus):
        word_class_map = wll.load_classes()
        feature_dim = class_name
        feature_matrix = np.zeros((len(test_corpus),feature_dim))
        for sent_id,sent in enumerate(test_corpus):
            feature = np.zeros(feature_dim)
            words = sent.split(' ')
            for word in words:
                if word in word_class_map:
                    feature[word_class_map[word]] = 1
            feature_matrix[sent_id, :] = feature
        return feature_matrix

    def get_all_feature(self,test_corpus):
        #return self.get_word_feature(test_corpus)
        matrix = np.column_stack((self.get_word_feature(test_corpus),self.get_bigram_feature(test_corpus)))
        #matrix = self.get_word_feature(test_corpus)
        matrix = np.column_stack((matrix,self.get_namelist1_feature(test_corpus)))
        matrix = np.column_stack((matrix,self.get_namelist2_feature(test_corpus)))
        #matrix = np.column_stack((matrix,self.get_bigram_feature(test_corpus)))
        matrix = np.column_stack((matrix,self.get_cluster_feature(test_corpus)))
        #matrix = np.column_stack((matrix,self.get_head_word_feature(test_corpus)))
        return matrix


if __name__ == '__main__':
    #test
    docs = ['a good man','his name is qf','is qf']
    load_feature = LoadFeature(docs)
    sents = ['a good woman','is qf his name']
    # feature_matrix = load_feature.get_word_feature(sents)
    # print(feature_matrix)
    # feature_matrix = load_feature.get_word_tfidf(sents)
    # print(feature_matrix)
    feature_matrix = load_feature.get_all_feature(sents)
    print(feature_matrix)




