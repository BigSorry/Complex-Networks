import pandas as pd
import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
import random as rnd
import sys
import math

def first10days(campaign_date):
    days10 = int(time.mktime(time.strptime("2000-01-10 12:00:00", pattern))) - int(time.mktime(time.strptime("2000-01-01 12:00:00", pattern)))
    campaign_10days = []
    for i in range(0, len(campaign_date)):
        if campaign_date[i] - campaign_date[0] <= days10:
            campaign_10days.append(campaign_date[i])
    return campaign_10days


def epoch_to_timestamp(campaign_date):
    # initialize variables to change date and time to timestamp
    modif = 1
    prev_date = campaign_date[0]
    campaign_date_ts = []

    for i in range(0, len(campaign_date)):
        if campaign_date[i] != prev_date:
            prev_date = campaign_date[i]
            modif = modif + 1
        campaign_date_ts.append(modif)
    return campaign_date_ts


def uniform_epochs(campaign_date, infected):
    uniform_infected = []
    uniform_infected.append(infected[0])
    one_minute = int(time.mktime(time.strptime("2000-01-01 12:01:00", pattern))) - int(time.mktime(time.strptime("2000-01-01 12:00:00", pattern)))

    index = 0
    crt_date = campaign_date[0]

    while index < len(infected) - 1:
        if index == len(campaign_date) - 1:
            break
        crt_date = crt_date + one_minute
        if crt_date < campaign_date[index+1]:
            uniform_infected.append(infected[index])
        else:
            index = index + 1
            uniform_infected.append(infected[index])

    return uniform_infected

def infection_evolution(campaign_date_ts):
    prev_date = 1
    cnt = 0
    infected = []
    for i in range(0, len(campaign_date_ts)):
        if prev_date != campaign_date_ts[i]:
            infected.append(cnt)
        cnt = cnt + 1
        prev_date = campaign_date_ts[i]

    return infected


C = 3


df_campaigns = pd.read_csv("Datasets/campaigns.csv")
#df_friends = pd.read_csv("Datasets/friends.csv", delimiter=';', header=None)

friends_graph = nx.Graph()
campaign_graph = nx.Graph()
infected_graph = nx.Graph()

# split the campaigns data set
campaign_elements = []
campaign_no = []

campaign_date = []
campaign_activating = []
campaign_activated = []

# split the friends data set
friend1 = []
friend2 = []


# variable used to convert date to epoch
pattern = '%Y-%m-%d %H:%M:%S'


# separate one campaign

for i in range(0, len(df_campaigns)):
    campaign_elements = df_campaigns.iloc[i][0].split(';')
    campaign_no.append(int(campaign_elements[0]))

    if campaign_no[i] == C:
        epoch = int(time.mktime(time.strptime(campaign_elements[1], pattern)))
        campaign_date.append(epoch)

        campaign_activating.append(campaign_elements[2])
        campaign_activated.append(campaign_elements[3])



# Necessary to generate faster the network
print("campaign")
campaigns = pd.read_csv("Datasets/campaigns.csv", delimiter=';', header=None)
campaignData = campaigns.loc[:, 0:3]
campaignData = campaignData.loc[campaignData[0] == C]
campaignData.loc[:, 1] = campaignData.loc[:, 1].apply(lambda x: int(time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S"))))
campaignData = campaignData.values
allFriends = set()
beginTime = sys.maxsize

for [camp, t, node1, node2] in campaignData:
    beginTime = min(beginTime, t)
    allFriends.add(node1)
    allFriends.add(node2)

    if t <= t + (10*24*60):
        campaign_graph.add_node(node1)
        campaign_graph.add_node(node2)
        campaign_graph.add_edge(node1, node2)

print("friends")
friends = pd.read_csv("Datasets/friends.csv", delimiter=';', header=None)
friendsData = friends.loc[:, 0:2]
friendsData = friendsData.loc[(friendsData[1].isin(allFriends) | friendsData[2].isin(allFriends))]
friendsData.loc[:, 0] = friendsData.loc[:, 0].apply(lambda x: int(time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S"))))
friendsData = friendsData.values
print(len(friendsData))


# Add the rest of campaign nodes to graph

for [t, node1, node2] in friendsData:
    friends_graph.add_node(node1, weight=0, infected=False)
    friends_graph.add_node(node2, weight=0, infected=False)
    friends_graph.add_edge(node1, node2)

for [id, t, node1, node2] in campaignData:
    friends_graph.add_node(node1, weight=0, infected=False)
    friends_graph.add_node(node2, weight=0, infected=False)
    friends_graph.add_edge(node1, node2)

# Lists of seed nodes in campaign
seeds = campaign_activating[0]

campaign_10days = first10days(campaign_date)
campaign_date_ts = epoch_to_timestamp(campaign_10days)



# Just for the third campaign
beta = 0.05
alpha = 3
sum_deg = 0

list_deg = list(friends_graph.degree(friends_graph.nodes))
degrees = [x[1] for x in list_deg]

for d in degrees:
    sum_deg = sum_deg + d**alpha

for node in friends_graph.nodes:
    friends_graph.nodes[int(node)]['weight'] = beta * ((friends_graph.degree(int(node))**alpha) / sum_deg) * len(friends_graph)

# Simulate spreading
sim_campaign_10days = []
sim_activating = []
sim_activated = []

# Infection of seed nodes at t=0
friends_graph.nodes[int(seeds)]['infected'] = True

for t in range(min(campaign_10days), max(campaign_10days) + 1, 60):
    for edge in friends_graph.edges:
        node1 = edge[0]
        node2 = edge[1]
        p = rnd.uniform(0, 1)
        if friends_graph.nodes[node1]['infected'] == True and friends_graph.nodes[node2]['infected'] == False:
            if friends_graph.nodes[node2]['weight'] >= p:
                friends_graph.nodes[node2]['infected'] = True
                sim_campaign_10days.append(t)
                sim_activating.append(node1)
                sim_activated.append(node2)

                infected_graph.add_node(node1)
                infected_graph.add_node(node2)
                infected_graph.add_edge(node1, node2)

        if friends_graph.nodes[node1]['infected'] == False and friends_graph.nodes[node2]['infected'] == True:
            if friends_graph.nodes[node1]['weight'] >= p:
                friends_graph.nodes[node1]['infected'] = True
                sim_campaign_10days.append(t)
                sim_activating.append(node2)
                sim_activated.append(node1)

                infected_graph.add_node(node1)
                infected_graph.add_node(node2)
                infected_graph.add_edge(node1, node2)

sim_campaign_10days_ts = epoch_to_timestamp(sim_campaign_10days)
sim_infected = infection_evolution(sim_campaign_10days_ts)

sim_campaign_10days_unique = np.sort(list(set(sim_campaign_10days))).tolist()
sim_infected_unif = uniform_epochs(sim_campaign_10days_unique, sim_infected)

print(len(campaign_graph))
print(sim_infected[-1])

plt.semilogy(range(1, len(sim_infected_unif) + 1), sim_infected_unif, 'g-')
plt.xlabel('Minutes')
plt.ylabel('Interactions')
plt.title('Campaign')
plt.show()

# Average degree
avg_deg_infected = 0
for d in infected_graph.degree():
    avg_deg_infected = avg_deg_infected + d[1]
avg_deg_infected = avg_deg_infected/len(infected_graph)

avg_deg_campaign = 0
for d in campaign_graph.degree():
    avg_deg_campaign = avg_deg_campaign + d[1]
avg_deg_campaign = avg_deg_campaign/len(campaign_graph)

# Average betweeness
avg_bet_infected = 0
for b in nx.betweenness_centrality(infected_graph).values():
    avg_bet_infected = avg_bet_infected + b
avg_bet_infected = avg_bet_infected/len(infected_graph)

avg_bet_campaign = 0
for b in nx.betweenness_centrality(campaign_graph).values():
    avg_bet_campaign = avg_bet_campaign + b
avg_bet_campaign = avg_bet_campaign/len(campaign_graph)

# Assortativity

ass_infected = nx.degree_pearson_correlation_coefficient(infected_graph)
ass_campaign = nx.degree_pearson_correlation_coefficient(campaign_graph)

print("Degree of infected:", avg_deg_infected,"\nDegree of campaign:", avg_deg_campaign)
print("Betweenness of infected:", avg_bet_infected,"\nBetweenness of campaign:", avg_bet_campaign)
print("Assortativity of infected:", ass_infected,"\nAssortativity of campaign:", ass_campaign)

