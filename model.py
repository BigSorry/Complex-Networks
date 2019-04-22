import pandas as pd
import networkx as nx
import time
import math
import sys
import matplotlib.pyplot as plt
from random import random
import numpy as np

day = 60*60*24

print("campaign")
campaigns = pd.read_csv('./campaigns.csv', delimiter=';', header=None)
campaignData = campaigns.loc[:, 0:3]
campaignData = campaignData.loc[campaignData[0] == 4]
campaignData.loc[:, 1] = campaignData.loc[:, 1].apply(lambda x: int(time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S"))))
campaignData = campaignData.values


allFriends = set()
beginTime = sys.maxsize
for [camp, t, node1, node2] in campaignData:
    beginTime = min(beginTime, t)
    allFriends.add(node1)
    allFriends.add(node2)

print("friends")
friends = pd.read_csv('./friends.csv', delimiter=';', header=None)
friendsData = friends.loc[:, 0:2]
friendsData = friendsData.loc[(friendsData[1].isin(allFriends) | friendsData[2].isin(allFriends))]
friendsData.loc[:, 0] = friendsData.loc[:, 0].apply(lambda x: int(time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S"))))
friendsData = friendsData.values
print(len(friendsData))


campaigns = None
friends = None

graphCamp = nx.DiGraph()
for [camp, t, node1, node2] in campaignData:
    tstamp = math.ceil((t-beginTime+1)/day)
    graphCamp.add_edge(node1, node2, t=tstamp)
nx.write_graphml(graphCamp, 'graphCampaign.graphml')

old_link_set = {x: set() for x in nx.nodes(graphCamp)}
for [camp, t, node1, node2] in campaignData:
    tstamp = math.ceil((t-beginTime+1)/day)
    old_link_set[node1].add((node2, tstamp))
    old_link_set[node2].add((node1, tstamp))


infec = [set() for x in range(0, 240*60)]
for [camp, t, node1, node2] in campaignData:
    tstamp = math.floor((t - beginTime + 1) / 60)
    if tstamp < 240*60:
        infec[tstamp].add(node2)

old_number_infected = np.zeros(240*60)
for x in range(0, 240*60):
    if x == 0:
        old_number_infected[x] = len(infec[x])
    else:
        old_number_infected[x] = len(infec[x]) + old_number_infected[x-1]

x_axis = [i+1 for i in range(0, 240*60)]

plt.figure(4)
plt.plot(x_axis, old_number_infected, 'red', label="number of infected nodes")


graphComplete = nx.Graph()
for [t, node1, node2] in friendsData:
    graphComplete.add_node(node1, color='green')
    graphComplete.add_node(node2, color='green')
    graphComplete.add_edge(node1, node2)
for [camp, t, node1, node2] in campaignData:
    tstamp = int(math.floor((t-beginTime+1)/day))
    if not graphComplete.has_node(node1):
        graphComplete.add_node(node1, color='green')
    if tstamp == 0:
        graphComplete.add_node(node2, color='red')
    elif tstamp == 1:
        graphComplete.add_node(node2, color='orange')
    elif tstamp == 2:
        graphComplete.add_node(node2, color='crimson')
    elif tstamp == 3:
        graphComplete.add_node(node2, color='darkred')
    elif tstamp == 4:
        graphComplete.add_node(node2, color='deeppink')
    elif tstamp >= 5 | tstamp <= 10:
        graphComplete.add_node(node2, color='purple')
    else:
        graphComplete.add_node(node2, color='black')
    graphComplete.add_edge(node1, node2, time=tstamp)
node_color = []
for node in graphComplete.nodes(data=True):
    if 'green' in node[1]['color']:
        node_color.append('green')
    elif 'red' in node[1]['color']:
        node_color.append('red')
    elif 'orange' in node[1]['color']:
        node_color.append('orange')
    elif 'crimson' in node[1]['color']:
        node_color.append('crimson')
    elif 'darkred' in node[1]['color']:
        node_color.append('darkred')
    elif 'deeppink' in node[1]['color']:
        node_color.append('deeppink')
    elif 'purple' in node[1]['color']:
        node_color.append('purple')
    else:
        node_color.append('black')


print("finished")
# plt.figure(1)
# nx.draw(graphComplete, with_labels=False, node_size=25, node_color=node_color)


print("number of nodes: ", graphComplete.number_of_nodes())
print("number of edges: ", graphComplete.number_of_edges())


# model
best_rmse = [900, 0, 0]
for s in range(20, 21):
    for t in range(50, 51):
        alpha = s/10
        beta = t/100

        print("alpha:", alpha, " beta:", beta)
# alpha = 2.5
# beta = 0.02
        hours = 10 * 24 * 60

        model_graph = nx.Graph()
        link_set = {x: set() for x in nx.nodes(graphComplete)}
        for x in nx.nodes(graphComplete):
            link_set[x].update(nx.all_neighbors(graphComplete, x))
            link_degree = {x: len(link_set[x]) for x in link_set.keys()}
            tot_degree = sum([x**alpha for x in link_degree.values()])

        infected = set()
        infected.add(20682)
        model_graph.add_node(20682, color=-1)
        number_infected = np.empty(hours)
        for i in range(0, hours):
            new_infected = set()
            for old_node in infected:
                for new_node in link_set[old_node]:
                    if new_node not in infected and new_node not in new_infected:
                        beta_node = beta * (link_degree[old_node]/tot_degree) * graphComplete.number_of_nodes()
                        # print("beta:", beta, " t:", math.exp((-hours)/(5*24*60)), " link:", (link_degree[old_node]/tot_degree), " n:", graphComplete.number_of_nodes())
                        # print(beta_node)
                        r = random()
                        if r <= beta_node:
                            new_infected.add(new_node)
                            model_graph.add_node(new_node, color=i)
                            model_graph.add_edge(old_node, new_node)
            infected.update(new_infected)
            number_infected[i] = len(infected)


        rmse = 0
        for i in range(0, len(old_number_infected)):
            rmse += ((old_number_infected[i] - number_infected[i]) ** 2)
        rmse /= len(number_infected)
        rmse = math.sqrt(rmse)
        if rmse < best_rmse[0]:
            best_rmse[0] = rmse
            best_rmse[1] = alpha
            best_rmse[2] = beta

print("best values:", best_rmse)


node_colors = []
for node in model_graph.nodes(data=True):
    if -1 == node[1]['color']:
        node_colors.append('green')
    elif 0 <= node[1]['color'] < 12:
        node_colors.append('red')
    elif 12 <= node[1]['color'] < 24:
        node_colors.append('orange')
    elif 24 <= node[1]['color'] < 36:
        node_colors.append('crimson')
    elif 36 <= node[1]['color'] < 48:
        node_colors.append('darkred')
    elif 48 <= node[1]['color'] < 60:
        node_colors.append('deeppink')
    elif 60 <= node[1]['color'] < 72:
        node_colors.append('purple')
    else:
        node_colors.append('black')

# plt.figure(2)
# nx.draw(model_graph, with_labels=False, node_size=25, node_color=node_colors)

x_axis = [i+1 for i in range(0, hours)]

plt.figure(3)
plt.plot(x_axis, number_infected, 'red', label="number of infected nodes")

print(number_infected)

plt.show()

