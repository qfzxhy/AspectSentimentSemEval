from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import codecs
def generate_xml(labels=None,input_path=None,output_path=None):
    if input_path == None or output_path == None:
        print('file can not none')
        return
    reader = codecs.open(input_path,'r',encoding='utf-8')
    writer = codecs.open(output_path,'w',encoding='utf-8')
    sentence_id = 0
    lines = reader.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if '<text>' in line and '</text>' in line:
            writer.write(line)
            if labels[sentence_id] != None:
                writer.write('                <Opinions>\n')
                for text_label in labels[sentence_id]:
                    writer.write('                    <Opinion target="NULL" category="'+text_label+'" polarity="NULL" from="0" to="0"/>\n')
                writer.write('                </Opinions>\n')
            sentence_id = sentence_id + 1
            while '</sentence>' not in lines[i]:
                i = i + 1

        else:
            writer.write(line)
            #print(line)
            i = i + 1






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
                # print('{'+opinion.getAttribute('target')+','
                #       +opinion.getAttribute('category')+','
                #       +opinion.getAttribute('polarity')+','
                #       +opinion.getAttribute('from')+','
                #       +opinion.getAttribute('to')+'}')
                label.append(dic)
        # else:
        #     print(opinions)
        labels.append(label)

class Load(object):
    def __init__(self,file):
        self.datas = []
        self.labels = []
        #self.map = {}
        parse_xml(self.datas,self.labels,file)
        # print(len(self.datas))
        # print(len(self.labels))

# Load()
if __name__ == '__main__':
    testdata_path = '../restaurant2015/ABSA15_Restaurants_Test.xml'
    load = Load(testdata_path)
#predict:675gold:775common:457
# precision:0.677037037037recall:0.589677419355f:0.630344827586
# fscore:0.630344827586

#precision:0.71009771987recall:0.562580645161f:0.627789776818
#fscore:0.627789776818
#
#