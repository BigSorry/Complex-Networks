import numpy as np
import pandas as pd
import networkx as nx
import time
from collections import  defaultdict

campaigns = pd.read_csv('./campaigns.csv', delimiter=';', header = None)
friends = pd.read_csv('./friends.csv',   nrows= 177850, delimiter=';', header=None)
transactions = pd.read_csv('./transactions.csv', skiprows= 2711, nrows= 5000, delimiter=';', header = None)

graphFriends = nx.Graph()
graphTrans = nx.Graph()

friendsData = friends.loc[:, 0:2]
friendsData.loc[:,0] =  friendsData.loc[:,0].apply(lambda x : int(time.mktime(time.strptime(x,"%Y-%m-%d %H:%M:%S"))))
friendsData = friendsData.values

transactionsData = transactions.loc[:,0:2]
transactionsData.loc[:,0] =  transactionsData.loc[:,0].apply(lambda x : int(time.mktime(time.strptime(x,"%Y-%m-%d %H:%M:%S"))))
transactionsDataP = transactionsData
transactionsData = transactionsData.values

# graphFriends.add_edges_from(friendsData.loc[:, 1:2].values)
# graphTrans.add_edges_from(transactionsData.loc[:,1:2].values)

edgesFriends = defaultdict(set)
for [date,node1,node2] in friendsData:
    key = (min(node1,node2), max(node1,node2))
    edgesFriends[key].add(date)

# edgesTrans = np.array([val for val in graphTrans.edges.keys()])
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
print('Done')