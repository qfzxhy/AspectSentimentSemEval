from nltk.tokenize import word_tokenize
import codecs
import os
#path = "G:/Master2D/semeval2016/archive/yelp_dataset_challenge_round9/test/"
from nlp_tools.stanford_parser import parser
path = "G:/Master2D/semeval2016/archive/yelp_dataset_challenge_round9/test/"
path3 = "G:/Master2D/semeval2016/archive/yelp_dataset_challenge_round9/yelp_review_processed/"
def get_files():
    filename_list = os.listdir(path)
    return filename_list

def get_tokens(parse_result,sent):

    tokens = []
    for x in parse_result.tokens:
        r = list(x)
        tokens.append(sent[int(r[0]):int(r[1])])
    return tokens


def tokenize_process(file,stanford_parser):

    reader = codecs.open(path + file,'r','utf-8',errors='ignore')
    list = []
    for line in reader.readlines():
        #print(line)
        if line.strip('\n') == "":
            continue
        else:
            # words = word_tokenize(line)
            if len(line.split()) < 5:
                continue
            lines = line.split('.')
            for l in lines:
                parse_result = stanford_parser.parseToStanfordDependencies(line)
                tokens = get_tokens(parse_result, l)
                new_line = " ".join(tokens)
                print new_line
                list.append(new_line)
    reader.close()
    writer = codecs.open(path3 + file,'w','utf-8')
    for line in list:
        if line.strip() == "":
            continue
        else:
            writer.write(line+"\n")
    writer.flush()
    writer.close()

if __name__ =='__main__':
    stanford_parser = parser.Parser()
    for file in get_files():
        print(file)
        tokenize_process(file,stanford_parser)
    # print "Then there's the other 1\/2 of the store ".split()