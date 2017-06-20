#aspect detection
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier

import restaurant2015 as rst
from AspectDetection import featureprecessing as fp

#get tfidf
category_num = 13
traindata_path = '../restaurant2015/ABSA-15_Restaurants_Train_Final.xml'
testdata_path = '../restaurant2015/ABSA15_Restaurants_Test.xml'
output_path = 'G:/Master2D/semeval2016/restaurant2015/baseevalvalid1/BaseEvalValid1/predict.xml'

categorys = ['DRINKS#QUALITY',
            'FOOD#GENERAL',
            'LOCATION#GENERAL',
            'FOOD#QUALITY',
            'FOOD#PRICES',
            'DRINKS#STYLE_OPTIONS',
            'FOOD#STYLE_OPTIONS',
            'RESTAURANT#GENERAL',
            'DRINKS#PRICES',
            'RESTAURANT#PRICES',
            'AMBIENCE#GENERAL',
            'RESTAURANT#MISCELLANEOUS',
            'SERVICE#GENERAL']
category_map = {
    'DRINKS#QUALITY':1,
    'FOOD#GENERAL':2,
    'LOCATION#GENERAL':3,
    'FOOD#QUALITY':4,
    'FOOD#PRICES':5,
    'DRINKS#STYLE_OPTIONS':6,
    'FOOD#STYLE_OPTIONS':7,
    'RESTAURANT#GENERAL':8,
    'DRINKS#PRICES':9,
    'RESTAURANT#PRICES':10,
    'AMBIENCE#GENERAL':11,
    'RESTAURANT#MISCELLANEOUS':12,
    'SERVICE#GENERAL':13
}

# raw label : food#quality -> num label : [0,0,1,0,0,...,]
def raw_label_process(docs_raw_str_labels):
    labels = []
    for each_doc_raw_str_labels in docs_raw_str_labels:
        label = [0 for i in range(category_num)]
        for each_doc_raw_str_label_dic in each_doc_raw_str_labels:
            # print(each_doc_raw_str_label_dic)
            label[category_map[each_doc_raw_str_label_dic['category']] - 1] = 1
            # print(label)
        labels.append(label)
    Y = np.array(labels)
    return Y

paras = {
    'solver':'sgd',
    'activation':'relu',
    'alpha':0.01,
    'random_state':1,
    'hidden_layer_sizes':(8),
    'learning_rate_init':0.9,
    'threshold':0.15

}
def getdata(trainX,trainY):
    #
    for category_id,category in enumerate(categorys):
        vec = np.zeros(13)
        vec[category_id] = 1
        y = np.matmul(trainY,vec)
        yield category,trainX,y

def train():
    print('load train datas and labels')
    load = rst.Load(traindata_path)
    train_docs = load.datas
    train_docs_labels = load.labels
    feature_loader = fp.LoadFeature(train_docs,train_docs_labels)
    print('begin train')
    clfs = []
    trainX = feature_loader.get_all_feature(train_docs)
    #tfidf_matrix 是稀疏矩阵,转化为正常矩阵

    trainY = raw_label_process(train_docs_labels)
    #trainX, trainY = data_feature_precess(train_docs, train_all_docs_raw_str_labels,train_docs)
    for category,trainX,y in getdata(trainX,trainY):
        print('train the classfier:'+category)
        clf = MLPClassifier(solver=paras['solver'],
                            hidden_layer_sizes=paras['hidden_layer_sizes'],
                            alpha=paras['alpha'],
                            learning_rate_init=paras['learning_rate_init'],
                            random_state=paras['random_state'],
                            verbose = False
                            )
        #clf = SVC(kernel='linear', probability=True)
        clf.fit(trainX,y)
        clfs.append(clf)
    return clfs

def test(clfs = None):
    if clfs == None:
        return
    print('begin test')
    #test_loader
    load_test = rst.Load(testdata_path)
    #测试数据，原始数据
    test_docs = load_test.datas
    #测试数据label,原始label。like:<Opinion target="food" category="FOOD#QUALITY" polarity="negative" from="4" to="8"/>
    test_docs_labels = load_test.labels
    #同上
    load_train = rst.Load(traindata_path)
    #加载特征
    feature_loader = fp.LoadFeature(load_train.datas,load_train.labels)
    testX = feature_loader.get_all_feature(test_docs)
    #测试数据label转化为矩阵形式
    testY = raw_label_process(test_docs_labels)
    #预测概率初始化
    predict_prob = np.zeros((len(test_docs), category_num))
    #预测
    for col, clf in enumerate(clfs):
        prob = clf.predict_proba(testX)[:, 1]
        predict_prob[:, col] = prob

    # 根据threshold决定每一句预测的label(字符串形式)
    labels = []
    for docid, test_doc in enumerate(test_docs):
        # up_threshold_ids : a tuple
        up_threshold_ids = np.where(predict_prob[docid, :] >= paras['threshold'])
        ids = up_threshold_ids[0]
        # for id in up_threshold_ids:
        if len(ids) == 0:
            labels.append(None)
            continue
        label = []
        for id in ids:
            label.append(categorys[id])
        labels.append(label)
    #生成预测结果xml,利用评估工具A.jar来评测
    rst.generate_xml(labels, input_path=testdata_path, output_path=output_path)

    #利用自己的评测函数来评测，二者评测结果差不多
    threshod_list = [paras['threshold'] for j in range(category_num)]
    threshod_vec = np.array(threshod_list)
    for row in range(len(test_docs)):
        predict_prob[row, :] = np.less_equal(threshod_vec, predict_prob[row, :])
    predict_prob =predict_prob.astype(int)
    # get f score     predict_prob and  testY
    fscore = get_fscore(testY, predict_prob)
    print('fscore:'+str(fscore))


def paras_generate():
    tuned_parameters = {
        'learning_rate_init': [0.1, 0.5, 0.9],
        'hidden_layer_sizes': [2, 4, 6, 8, 10],
        'solver': ['sgd', 'lbfgs'],
        'alpha': [0.0001, 0.001, 0.01],
        'threshold': [0.1,0.14,0.2,0.24]
    }
    for i1 in (tuned_parameters['learning_rate_init']):
        for i2 in (tuned_parameters['hidden_layer_sizes']):
            for i3 in (tuned_parameters['solver']):
                for i4 in (tuned_parameters['alpha']):
                    for i5 in tuned_parameters['threshold']:
                        yield i1,i2,i3,i4,i5

# tune paras for multi-label classify that use the whole label
def model_selection2():
    load = rst.Load(traindata_path)
    docs = load.datas
    docs_labels = load.labels
    #key paras value:fscore
    paras_fscore_map = {}
    # 5 - cross validate
    for i in range(5):
        #random_state = 0 : 每次随机种子不一样
        train_docs,eval_docs,train_docs_labels,eval_docs_labels = train_test_split(docs,docs_labels,test_size=0.2,random_state=0)
        feature_loader = fp.LoadFeature(train_docs)
        trainX = feature_loader.get_all_feature(train_docs)
        trainY = raw_label_process(train_docs_labels)
        evalX = feature_loader.get_all_feature(eval_docs)
        evalY = raw_label_process(eval_docs_labels)

        paras_group = []
        for i1,i2,i3,i4,i5 in paras_generate():
            paras_str = str(i1)+','+str(i2)+','+str(i3)+','+str(i4)+","+str(i5)
            print('begin'+paras_str)
            f = 0.0
            clfs = []
            for category, trainX, y in getdata(trainX,trainY):
                #print('train the classfier in model selection MODEL:' + category)
                clf = MLPClassifier(solver=i3,
                                             hidden_layer_sizes=i2,
                                             alpha=i4,
                                             learning_rate_init=i1,
                                             random_state=paras['random_state']
                                    )
                clf.fit(trainX, y)
                clfs.append(clf)
            predict_prob = np.zeros((len(eval_docs), category_num))
            for col, clf in enumerate(clfs):
                prob = clf.predict_proba(evalX)[:, 1]
                predict_prob[:, col] = prob
            clfs.clear()
            #根据predict_label和threshold来决定预测label
            threshod_list = [i5 for j in range(category_num)]
            threshod_vec = np.array(threshod_list)
            for row in range(len(eval_docs)):
                predict_prob[row,:] = np.less_equal(threshod_vec,predict_prob[row,:])
            predict_prob.astype(int)
            #get f score     predict_prob and  testY
            fscore = get_fscore(evalY,predict_prob)
            if i == 0:
                paras_fscore_map[paras_str] = fscore
            else:
                paras_fscore_map[paras_str] = (paras_fscore_map[paras_str] * (i-1)+fscore)/i
            # print((fscore,[i1,i2,i3,i4,i5]))
            # paras_group.append((fscore,[i1,i2,i3,i4,i5]))
    sorted(paras_fscore_map.items(),lambda x,y : x[1] > y[1])
    for item in paras_fscore_map.items():
        print(item)
    #print("best paras:"+str(max(paras_group)))

#get F1
def get_fscore(gold_matrix,predict_matrix):

    predict = np.sum(predict_matrix == 1)
    gold = np.sum(gold_matrix == 1)
    common = 0
    predict_list = predict_matrix
    gold_list = gold_matrix

    for i in range(len(predict_list)):
        for j in range(len(predict_list[i])):
            if predict_list[i][j] == 1 and gold_list[i][j] == 1:
                common += 1
    print('predict:'+str(predict)+'gold:'+str(gold)+'common:'+str(common))
    precision = common * 1.0 / predict
    recall = common * 1.0 / gold
    f = 2 * precision * recall / (precision + recall)
    print('precision:'+str(precision)+'recall:'+str(recall)+"f:"+str(f))
    return f

#tune the paras for each binary classify
#has error this method


#model_selection2()
# #model_selection()

if __name__ == '__main__':

    clfs = train()
    test(clfs)

# def train():
#     #13model