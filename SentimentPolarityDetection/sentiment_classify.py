# -*- coding: utf-8 -*-
'''
by qianf
2017-3-24
aspect detect ---------fasttext
8 aspect
'''
#baseline
#特特征：上下文{词+词性}
#model：svm or fasttext or other
from sklearn.cross_validation import KFold
from nltk.tokenize import word_tokenize
import codecs
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.svm import SVC,LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
# import fasttext
from SentimentPolarityDetection import  data_loader
category = ['positive','netural','negative']
paras = {
    'window':6
}
def loadfile(file_path=None):
    if file_path==None:
        return
    load = data_loader.Load(file_path)
    return load.datas,load.labels

def feature_process(filepath):
    #left right
    docs,docs_labels = loadfile(file_path=filepath)
    docs_features_sentiment_label = []
    docs_sentiment_labels = []
    for doc_id,doc_labels in enumerate(docs_labels):
        sent = docs[doc_id]
        tokens = word_tokenize(sent)
        for doc_label in doc_labels:
            doc_features = []
            if doc_label['target'] != "NULL":
                from_id = int(doc_label['from'])#cong 0开始
                to_id = int(doc_label['to'])
                left_sent = sent[:from_id].strip().split(' ')
                left_margin = len(left_sent) - paras['window'] - 1
                if left_margin < 0:
                    left_margin = 0
                for i in range(left_margin,len(left_sent)):
                    doc_features.append(left_sent[i])
                doc_features.append(sent[from_id:to_id])
                right_sent = sent[to_id:].strip().split(' ')
                right_margin = paras['window']
                if right_margin >= len(right_sent):
                    right_margin = len(right_sent) - 1
                for i in range(0,right_margin+1):
                    doc_features.append(right_sent[i])
            else:
                for word in tokens:
                    doc_features.append(word)
        docs_features_sentiment_label.append((' '.join(doc_features),doc_label['polarity']))
    return docs_features_sentiment_label

def generate_text_for_fasttext(traindata,testdata):
    traindata_path = './fasttext_model/traindata.txt'
    testdata_path = './fasttext_model/testdata.txt'
    filewriter1 = codecs.open(traindata_path,'w','utf-8')
    for data in traindata:
        sentence = data[0] + "\t__label__" + data[1] + "\n"
        filewriter1.write(sentence)
    filewriter1.flush()
    filewriter1.close()
    #test
    filewriter2 = codecs.open(testdata_path, 'w', 'utf-8')
    for data in testdata:
        sentence = data[0] + "\t__label__" + data[1] + "\n"
        filewriter2.write(sentence)
    filewriter2.flush()
    filewriter2.close()
    return traindata_path,testdata_path

# def train(train_filepath,test_filepath):
#     traindocs_features_sentiment_label = feature_process(train_filepath)
#     testdocs_features_sentiment_label = feature_process(test_filepath)
#     train_path,test_path = generate_text_for_fasttext(traindocs_features_sentiment_label,testdocs_features_sentiment_label)
#     classifier = fasttext.supervised(train_path, "./fasttext_model/news_fasttext.model", label_prefix="__label__")
#     print(test_path)
#    # texts = ['example very long text 1', 'example very longtext 2']
#     labels = classifier.test(test_path)
#     print(labels.precision)




def tfidf(train_docs,test_docs):
    vectorizer = TfidfVectorizer(stop_words='english')
    clf = vectorizer.fit(train_docs)
    return clf.transform(test_docs)

def train_other_model(train_filepath,test_filepath):
    traindocs_features_sentiment_label = feature_process(train_filepath)
    testdocs_features_sentiment_label = feature_process(test_filepath)
    train_docs = [x[0] for x in traindocs_features_sentiment_label]
    test_docs = [x[0] for x in testdocs_features_sentiment_label]
    train_labels = [x[1] for x in traindocs_features_sentiment_label]
    test_labels = [x[1] for x in testdocs_features_sentiment_label]
    trainX = tfidf(train_docs,train_docs)
    testX = tfidf(train_docs,test_docs)
    trainy = []
    for label in train_labels:
        if label == 'positive':
            trainy.append(1)
        elif label == 'netural':
            trainy.append(0)
        else:
            trainy.append(-1)
    trainy = np.array(trainy)
    testy = []
    for label in test_labels:
        if label == 'positive':
            testy.append(1)
        elif label == 'netural':
            testy.append(0)
        else:
            testy.append(-1)
    testy = np.array(testy)
    clf = LinearSVC()
    #clf = SGDClassifier(alpha=.0001, n_iter=50, penalty='l1', loss='log')
    #clf = MultinomialNB(alpha=0.01)
    clf.fit(trainX,trainy)
    labels = clf.predict(testX)
    print(sum(labels == testy)*1.0/len(testy))

if __name__ == '__main__':
    traindata_path = '../restaurant2015/ABSA-15_Restaurants_Train_Final.xml'
    testdata_path = '../restaurant2015/ABSA15_Restaurants_Test.xml'
    train_other_model(traindata_path,testdata_path)


