import networkx as nx
import numpy as np
import pandas as pd
import operator
from networkx.algorithms.community import k_clique_communities,girvan_newman,kernighan_lin_bisection,asyn_lpa_communities
import matplotlib.pyplot as plt

# preprocessing
# read the graph from the xml file
ga_edges = pd.read_csv('soc-sign-bitcoinotc.csv')
print (ga_edges.head())

G=nx.from_pandas_edgelist(ga_edges,'source','target',['weight','time'],nx.DiGraph())
print(nx.info(G))



# calculate Density
density = nx.density(G)
print('Density:', density)



# Centrality of the nodes
# in_degree_centrality
in_degree_centrality = nx.in_degree_centrality(G)
nx.set_node_attributes(G,in_degree_centrality,'in_degree_centrality')

sorted_in_degree_centrality = dict(sorted(in_degree_centrality.items(), key=operator.itemgetter(1), reverse=True)[:10])

print('sorted_in_degree_centrality:', sorted_in_degree_centrality)

# out_degree_centrality
out_degree_centrality = nx.out_degree_centrality(G)
nx.set_node_attributes(G, out_degree_centrality,'out_degree_centrality')

sorted_out_degree_centrality = dict(sorted(out_degree_centrality.items(), key=operator.itemgetter(1), reverse=True)[:10])

print('sorted_out_degree_centrality:', sorted_out_degree_centrality)

nx.write_gexf(G, "test.gexf")


## link prediction
G_undirected = nx.to_undirected(G)

## jaccard_coefficient

##preds_js = nx.jaccard_coefficient(G_undirected)

##preds_js_dict = {}
##for u,v,w in preds_js:
##    print(u)
##    preds_js_dict[(u,v)] = [w]


##print(sorted(preds_js_dict.items(), key=lambda x:x[1] , reverse=True)[:10])

## adamic_adar_index

##adar_js = nx.adamic_adar_index(G_undirected)

##adar_js_dict = {}

##for u,v,p in adar_js:
##    adar_js_dict[(u,v)] = p

##print(sorted(adar_js_dict.items(), key=lambda x:x[1] , reverse=True)[:10])

