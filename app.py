import os
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd

# preprocessing
# read the graph from the xml file
from algo import advance_link_preds


def remove_small_deg_nodes(G: nx.DiGraph, degree):
    degree_np = np.array(G.degree)
    small_deg_nodes = degree_np[degree_np[:, 1] < degree]
    G.remove_nodes_from(small_deg_nodes[:, 0])


def filter_based_on_time(G, times_arr, date):
    c1 = times_arr[..., 2] >= date[0]
    c2 = times_arr[..., 3] >= date[1]
    c3 = times_arr[..., 4] >= date[2]
    filtered_time = times_arr[c1 & c2 & c3]
    neg_filtered_time = times_arr[~(c1 & c2 & c3)]
    future_graph = G.copy()
    future_graph.remove_edges_from(neg_filtered_time[..., 0:2])
    G.remove_edges_from(filtered_time[..., 0:2])
    return future_graph


def timeline(G):
    g_times = nx.get_edge_attributes(G, 'time')
    times_arr = [[key[0], key[1], datetime.fromtimestamp(time_field).year, datetime.fromtimestamp(time_field).month,
                  datetime.fromtimestamp(time_field).day] for key, time_field in g_times.items()]
    times_arr = np.array(times_arr)
    unique_days = np.unique(times_arr[..., 2:], axis=0)
    return unique_days, times_arr


ga_edges = pd.read_csv('soc-sign-bitcoinotc.csv')
print(ga_edges.head())
G = nx.from_pandas_edgelist(ga_edges, 'source', 'target', ['weight', 'time'], nx.DiGraph())
max_node = np.max(G.nodes)

if not os.path.isfile('timeline.npz'):
    days, times_arr = timeline(G)
    np.savez('timeline.npz', days=days, times_arr=times_arr)
else:
    days = np.load('timeline.npz')['days']
    times_arr = np.load('timeline.npz')['times_arr']

chosen_timestamp = days[len(days) // 2]
# For testing
future_graph = filter_based_on_time(G, times_arr, chosen_timestamp)

remove_small_deg_nodes(G, degree=6)

chosen_timestamp_str = '{0}_{1}_{2}'.format(chosen_timestamp[0], chosen_timestamp[1], chosen_timestamp[2])
advance_link_preds(G, max_node=max_node,timestamp=chosen_timestamp_str)
print(1)
