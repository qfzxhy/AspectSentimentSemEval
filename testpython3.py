# d = {'a':1, 'b':0, 'c':1, 'd':4}
# for key in list(d.keys()):
#     if d[key] < 2:
#         del d[key]
# print(d)
import numpy as np


import os

from nlp_tools.stanford_parser import parser
stanford_parser = parser.Parser()
sent = "great hot dogs.."
x = stanford_parser.parseToStanfordDependencies("great hot dogs..")
tokens = []

for t in x.tokens:
    print(sent[int(list(t)[0]):int(list(t)[1])])

print(list(x.posTags))
print(list(x.tokens))
# x = stanford_parser.parseToStanfordDependencies("The pastas are incredible, the risottos (particularly the sepia) are fantastic and the braised rabbit is amazing.")
#
# print('postag:'+str(x.posTags))
# print('sent:'+str(x.sentence))
# print('toke:'+str(x.tokens))
# print('tokensToPosTags:'+str(x.tokensToPosTags))
# print("dependencies"+str(x.dependencies))
# print("depToGov"+str(x.depToGov))
# print(x)

# for y in x.dependencies:
#     print(y)
#
# for u in x.dependencies:
#     print(u[1])
#     l = list(u[1])
#     print(l)
#     print(x.sentence[l[0]:l[1]])
#print(x.tokens)





# X = [[1, 2, 3, 4],
#      [2, 2, 3, 4],
#      [3, 2, 3, 4],
#      [4, 2, 3, 4],
#      [5, 2, 3, 4],
#      [6, 2, 3, 4],
#      [7, 2, 3, 4],
#      [8, 2, 3, 4],
#      [9, 2, 3, 4],
#      [10, 2, 3, 4],]
# X = np.array(X)
# y = np.array([1,2,3,4,5,6,7,8,9,10])
# for i in range(5):
#     trainX,evalX,trainy,evaly = train_test_split(X,y)
#     print("--------------------")
#     print(trainX)
#     print(trainy)
sent = 'Pick up the tire pallet.'
