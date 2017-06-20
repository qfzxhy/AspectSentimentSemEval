import codecs
#import nltk
# from nltk.tokenize import word_tokenize
# # nltk.download('punkt')
# sent = "Why all the bad reviews??  Came here one day after work with a friend since I had just found out about the place. It's a combination of two of our favorite dishes, sisig and spicy seafood in a bag...how bad can it be right?? "
# print(sent)
# print(" ".join(word_tokenize(sent)))
classes_file = "../restaurant2015/classes3.txt"
def load_classes():
    word_class_map = {}
    reader = codecs.open(classes_file,'r','utf-8',errors='ignore')
    lines = reader.readlines()
    for line in lines:
        word_class = line.split(' ')
        word_class_map[word_class[0]] = int(word_class[1])
    reader.close()
    return word_class_map
# word_class_map = load_classes()
# print(word_class_map.items())