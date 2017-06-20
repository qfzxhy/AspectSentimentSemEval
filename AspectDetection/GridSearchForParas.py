from AspectDetection import featureprecessing as fp
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier

import restaurant2015 as rst
from AspectDetection import train as tn

traindata_path = './restaurant2015/ABSA-15_Restaurants_Train_Final.xml'
def gridsearch_threshold():
    thresholds1 = [0,1,2,3,4,5]
    thresholds2 = range(0,10,2)
    for i in thresholds1:
        for j in thresholds2:
            yield i,j

paras = {
    'solver':'sgd',
    'activation':'relu',
    'alpha':0.001,
    'random_state':1,
    'hidden_layer_sizes':(8),
    'learning_rate_init':0.9,
    'threshold':0.12

}
def cross_validating():
    load = rst.Load(traindata_path)
    docs = load.datas
    docs_labels = load.labels
    # key paras value:fscore
    paras_fscore_map = {}
    for i in range(5):
        #random_state = 0 : 每次随机种子不一样
        train_docs,eval_docs,train_docs_labels,eval_docs_labels = train_test_split(docs,docs_labels,test_size=0.2,random_state=0)

        trainY = tn.raw_label_process(train_docs_labels)

        evalY = tn.raw_label_process(eval_docs_labels)

        paras_group = []
        for i1,i2 in gridsearch_threshold():
            feature_loader = fp.LoadFeature(train_docs,train_docs_labels,i1,i2)
            trainX = feature_loader.get_all_feature(train_docs)
            print(trainX.shape)
            evalX = feature_loader.get_all_feature(eval_docs)
            paras_str = str(i1)+','+str(i2)
            print('begin'+paras_str)
            f = 0.0
            clfs = []
            for category, trainX, y in tn.getdata(trainX,trainY):
                #print('train the classfier in model selection MODEL:' + category)
                clf = MLPClassifier(solver=paras['solver'],
                                    hidden_layer_sizes=paras['hidden_layer_sizes'],
                                    alpha=paras['alpha'],
                                    learning_rate_init=paras['learning_rate_init'],
                                    random_state=paras['random_state'],
                                    verbose=False
                                    )
                clf.fit(trainX, y)
                clfs.append(clf)
            predict_prob = np.zeros((len(eval_docs), tn.category_num))
            for col, clf in enumerate(clfs):
                prob = clf.predict_proba(evalX)[:, 1]
                predict_prob[:, col] = prob
            clfs.clear()
            #根据predict_label和threshold来决定预测label
            threshod_list = [paras['threshold'] for j in range(tn.category_num)]
            threshod_vec = np.array(threshod_list)
            for row in range(len(eval_docs)):
                predict_prob[row,:] = np.less_equal(threshod_vec,predict_prob[row,:])
            predict_prob.astype(int)
            #get f score     predict_prob and  testY
            fscore = tn.get_fscore(evalY,predict_prob)
            if i == 0:
                paras_fscore_map[paras_str] = fscore
            else:
                paras_fscore_map[paras_str] = (paras_fscore_map[paras_str] * (i-1)+fscore)/i
            # print((fscore,[i1,i2,i3,i4,i5]))
            # paras_group.append((fscore,[i1,i2,i3,i4,i5]))
    paras_fscore_map=sorted(paras_fscore_map.items(),key = lambda x : x[1])
    for item in list(paras_fscore_map.items()):
        print(item)
    #print("best paras:"+str(max(paras_group)))
#

if __name__ == '__main__':
    cross_validating()