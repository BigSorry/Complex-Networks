import numpy as np
import pandas as pd
import networkx as nx
import time
import math
import matplotlib.pyplot as plt
from collections import defaultdict

campaigns = pd.read_csv('./campaigns.csv', delimiter=';', header=None)
friends = pd.read_csv('./friends.csv', nrows=5000000, delimiter=';', header=None)
transactions = pd.read_csv('./transactions.csv', delimiter=';', header=None)

graphFriends = nx.Graph()
graphTrans = nx.Graph()

friendsData = friends.loc[:, 0:2]
friendsData.loc[:,0] = friendsData.loc[:,0].apply(lambda x: int(time.mktime(time.strptime(x,"%Y-%m-%d %H:%M:%S"))))
friendsData = friendsData.values

transactionsData = transactions.loc[:,0:2]
transactionsData.loc[:,0] =  transactionsData.loc[:,0].apply(lambda x : int(time.mktime(time.strptime(x,"%Y-%m-%d %H:%M:%S"))))
transactionsDataP = transactionsData
transactionsData = transactionsData.values

# graphFriends.add_edges_from(friendsData.loc[:, 1:2].values)
graphTrans.add_edges_from(transactions.loc[:,1:2].values)
print("starting")

transDegreeSet = defaultdict(np.int32)
for [t, node1, node2] in transactionsData:
    if node1 in transDegreeSet:
        transDegreeSet[node1] += 1
    else:
        transDegreeSet[node1] = 1
    if node2 in transDegreeSet:
        transDegreeSet[node2] += 1
    else:
        transDegreeSet[node2] = 1

friendDegreeSet = defaultdict(set)
edgesFriends = defaultdict(set)

for [date,node1,node2] in friendsData:
    key = (min(node1,node2), max(node1,node2))
    edgesFriends[key].add(date)
    if node1 in transDegreeSet:
        friendDegreeSet[node1].add(node2)
    if node2 in transDegreeSet:
        friendDegreeSet[node2].add(node1)

edgesTrans = np.array([(min(node1,node2), max(node1,node2)) for [node1, node2] in graphTrans.edges.keys()])
# Friends dups
intersect = []
difference = []
for [date,node1,node2] in transactionsData:
    key = (min(node1,node2), max(node1,node2))
    if key in edgesFriends:
        boolIntersect = 0
        for dateFriend in edgesFriends[key]:
            if date > dateFriend:
                intersect.append((date, min(node1,node2), max(node1,node2)))
                boolIntersect = 1
                break
        if boolIntersect == 0:
            difference.append((date, min(node1,node2), max(node1,node2)))
    else:
        difference.append((date, min(node1,node2), max(node1,node2)))

intersect = np.unique(intersect, axis = 0)

print("Prob is {0}".format(np.array(difference).shape[0] / transactionsData.shape[0]))
print("Prob is {0}".format((transactionsData.shape[0] - intersect.shape[0]) / transactionsData.shape[0]))

friendDegree = [[x, len(friendDegreeSet[x])] for x in friendDegreeSet.keys()]
friendDegree = sorted(friendDegree, key=lambda d: d[1], reverse=True)
transDegree = [[x, transDegreeSet[x]] for x in transDegreeSet.keys()]
transDegree = sorted(transDegree, key=lambda d: d[1], reverse=True)


f = [0.05 * x for x in range(1, 11)]


r_TFf = [float(len(np.intersect1d(friendDegree[:int((len(friendDegree) * i))], transDegree[:int((len(transDegree) * i))])) / float(len(friendDegree[:int((len(friendDegree) * i))]))) for i in f]
r_FTf = [float(len(np.intersect1d(transDegree[:int((len(transDegree) * i))], friendDegree[:int((len(friendDegree) * i))])) / float(len(transDegree[:int((len(transDegree) * i))]))) for i in f]

plt.figure(1)
plt.plot(f, r_TFf, 'red', label="$r_{TF}(f)$")
plt.plot(f, r_FTf, 'blue', label="$r_{FT}(f)$")
plt.legend()


transDegree = sorted(transDegree, key=lambda d: d[0])
friendDegree = sorted(friendDegree, key=lambda d: d[0])

c = {key: [value] for [key, value] in transDegree}
for [id, y] in friendDegree:
    if id in c and len(c[id]) == 1:
        c[id].append(y)
c = [c[key] for key in c if len(c[key]) == 2]


def mean2(x):
    y = np.sum(x) / np.size(x)
    return y


def corr2(a,b):
    amean = mean2(a)
    bmean = mean2(b)

    a = a - amean
    b = b - bmean

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum())
    return r


x = np.array(c)[:, 0]
y = np.array(c)[:, 1]
cor = corr2(x,y)
print(cor)

plt.show()
print('Done')
