import codecs
import os
import sys
import importlib
importlib.reload(sys)

path = "G:/Master2D/semeval2016/archive/yelp_dataset_challenge_round9/yelp_review_datasets/"
path2 = "G:/Master2D/semeval2016/archive/yelp_dataset_challenge_round9/yelp_review_datasets_1/"
def get_files():
    filename_list = os.listdir(path)
    return filename_list



#
#    # reader.close()
# def process_line(line):
#     line = line[1:-1]
#     print(line)
#     strs = line.split(",")
#     text = ""
#     for str in strs:
#         if "\"text\":" in str:
#             text = str[7:]
#             print(text)
#             if text[0] == "\"" or text[0] == "\'":
#                 text = text[1:]
#             if text[-1] == '\"' or text[-1] == "\'":
#                 text = text[:-1]
#     return text

def process_line(line):
    line = line.strip('\n')
    #print(line)
    if line[0] != '{' or line[-1] != '}':
        return ''
    dic = eval(line)
    #print(dic)
    if 'text' in dic:
        return dic['text']
    return ''
def process():
    for filename in get_files():
        print(filename)
        list = []

        reader = codecs.open(path + filename, 'r', 'utf-8',errors='ignore')

        for line in reader.readlines():
            text = process_line(line)
            #print(text)
            if len(text) > 0:
                list.append(text)
        reader.close()
        writer = codecs.open(path2+filename,'w','utf-8')
        for line in list:
            writer.write(line+"\n")
        writer.flush()
        writer.close()
process()
# # text = process_line("{\"a\":1,\"text\":\"wo ss\'}")
# # print(text)
# s = '{"a":1,"text":"wo ss"}'
# dic = eval(s)
# print(dic['text'])
