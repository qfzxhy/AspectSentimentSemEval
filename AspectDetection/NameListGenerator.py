import restaurant2015 as rst
import operator
def get_namelist(docs,raw_sents_labels,threshold1=2,threshold2=3):
    name_list1 = {}
    name_list2 = {}
    all_name_list1 = {}
    all_name_list2 = {}

    for raw_sent_labels in raw_sents_labels:
        for raw_sent_label in raw_sent_labels:
            opinion_target = raw_sent_label['target']
            if opinion_target == 'NULL':
                continue
            if opinion_target not in name_list1:
                name_list1[opinion_target] = 1
                all_name_list1[opinion_target] = 0
            else:
                name_list1[opinion_target] += 1
            words = opinion_target.split(' ')
            for word in words:
                if word not in name_list2:
                    name_list2[word] = 1
                    all_name_list2[word] = 0
                else:
                    name_list2[word] += 1
    # for doc in docs:
    #     for key in all_name_list1.keys():
    #         all_name_list1[key] += doc.count(key)
    #     for key in all_name_list2.keys():
    #         all_name_list2[key] += doc.count(key)
    # print(name_list1)
    # for key in name_list1:
    #     name_list1[key] = name_list1[key] * 1.0 / all_name_list1[key]
    #name_list1 = sorted(name_list1.items(), key=lambda x: x[1], reverse=True)
    #name_list2 = sorted(name_list2.items(), key=lambda x: x[1], reverse=True)
    #name_list1 =
    #all_name_list1 = sorted(all_name_list1.items(),key = lambda x:x[1],reverse = True)
    # name_list1 = qselect(list(name_list1.items()),threshold1)
    # name_list2 = qselect(list(name_list2.items()),threshold2)
    for key in list(name_list1.keys()):
        if name_list1[key] < threshold1:
            del name_list1[key]
    for key in list(name_list2.keys()):
        if name_list2[key] < threshold2:
            del name_list2[key]
    return list(name_list1.items()),list(name_list2.items())
    #print(all_name_list1)
    #


def qselect(A, k):
    if len(A) < k: return A
    pivot = A[-1]
    right = [pivot] + [x for x in A[:-1] if x[1] >= pivot[1]]
    rlen = len(right)
    if rlen == k:
        return right
    if rlen > k:
        return qselect(right, k)
    else:
        left = [x for x in A[:-1] if x[1] < pivot[1]]
        return qselect(left, k - rlen) + right

if __name__ == '__main__':
    # docs = [
    #     'Judging from previous posts this used to be a good place, but not any longer.',
    #     'The food was lousy - too sweet or too salty and the portions tiny.',
    #
    # ]
    # #<Opinion target="place" category="RESTAURANT#GENERAL" polarity="negative" from="51" to="56"/>
    # #<Opinion target="food" category="FOOD#QUALITY" polarity="negative" from="4" to="8"/>
    # raw_sent_labels = [
    #     [{'Opinion target':'good place'}],
    #     [{'Opinion target':'good food'}]
    # ]
    traindata_path = './restaurant2015/ABSA-15_Restaurants_Train_Final.xml'
    load = rst.Load(traindata_path)
    get_namelist(load.datas,load.labels)