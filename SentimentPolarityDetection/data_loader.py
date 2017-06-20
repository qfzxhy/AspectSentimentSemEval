from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import codecs
def parse_xml(datas,labels,file_path):
    DOMTree = parse(file_path)
    Data = DOMTree.documentElement
    sentence_list = Data.getElementsByTagName("sentence")
    for sentence in sentence_list:
        # if sentence.hasAttribute('id'):
        #     print(sentence.getAttribute('id'))
        #print(sentence)
        text_node = sentence.getElementsByTagName("text")[0]
        #print(text.childNodes[0].data)
        text_str = text_node.childNodes[0].data
        text = ' '.join(word_tokenize(text_str))

        datas.append(text)
        opinions = sentence.getElementsByTagName("Opinions")
        label = []
        if len(opinions) > 0:
            opinions = opinions[0].getElementsByTagName("Opinion")
            for opinion in opinions:
                dic = {}
                dic['target'] = opinion.getAttribute('target')
                dic['category'] = opinion.getAttribute('category')
                dic['polarity'] = opinion.getAttribute('polarity')
                dic['from'] = opinion.getAttribute('from')
                dic['to'] = opinion.getAttribute('to')
                label.append(dic)
        labels.append(label)

class Load(object):
    def __init__(self,file):
        self.datas = []
        self.labels = []
        parse_xml(self.datas,self.labels,file)

# Load()
if __name__ == '__main__':
    testdata_path = '../restaurant2015/ABSA15_Restaurants_Test.xml'
    load = Load(testdata_path)