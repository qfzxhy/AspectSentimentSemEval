from nlp_tools.stanford_parser import parser
from nltk.tokenize import word_tokenize
def headword_corpus_generate(corpus):
    corpus_headword = []
    stanford_parser = parser.Parser()
    for sent in corpus:
        corpus_headword.append(headword_generate(sent,stanford_parser))
    return corpus_headword

def headword_generate(sent,stanford_parser):
    headwords = dependency_relation_generate(sent,stanford_parser)

    return ' '.join(headwords)

def dependency_relation_generate(sent,stanford_parser):
    re = stanford_parser.parseToStanfordDependencies(sent)
    dependencies = re.dependencies
    headwords = set()

    for dep in dependencies:
        l1 = list(dep[1])
        l2 = list(dep[2])
        headwords.add(sent[l1[0]:l1[1]])
        #deps.append(((l2[0],l2[1]),(l1[0],l1[1])))
    #sorted(deps,key = lambda x : x[0][0])
    return(headwords)
import restaurant2015 as rst
# x = headword_generate('Small, bright, and clean, BZ Grill stands apart from the usual run of gyro joints, both in china and out.',parser.Parser())
# print(x)
# testdata_path = './restaurant2015/ABSA15_Restaurants_Test.xml'
# load = rst.Load(testdata_path)
# headword_corpus_generate(load.datas)
