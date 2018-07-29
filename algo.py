import networkx as nx
from networkx.algorithms.link_prediction import _apply_prediction
import numpy as np
from scipy.special import comb
import tqdm


def advance_link_preds(G: nx.DiGraph, max_node,timestamp):
    # Algo weights
    w1 = 0.3
    w2 = 0.3
    w3 = 0.3
    w4 = 0.3
    weights = np.array([w1, w2, w3, w4])
    scores = np.full((int(max_node), int(max_node), 4), fill_value=0.0)
    # Algo scores
    jaccard_coefficient_val = jaccard_coefficient(nx.to_undirected(G))
    number_of_couples = int(comb(len(G.nodes), 2, exact=True, repetition=False))

    all_edges_weights = np.array([(uv[0], uv[1], w) for uv, w in nx.get_edge_attributes(G, 'weight').items()])
    all_positive_edges = all_edges_weights[all_edges_weights[:, 2] >= 0]
    all_negatives_edges = all_edges_weights[all_edges_weights[:, 2] < 0]
    average_in_degree = np.average(np.array(G.in_degree)[:, 1])
    average_negative_edeges = len(all_negatives_edges) / len(G.nodes)
    average_positive_edeges = len(all_negatives_edges) / len(G.nodes)

    print(jaccard_coefficient_val.__next__())
    postive_in_rating = generate_in_rating_func(avareage_graph_rating=np.average(all_positive_edges[:, 2]),
                                                average_graph_degree=average_positive_edeges)

    negative_in_rating = generate_in_rating_func(avareage_graph_rating=np.average(all_negatives_edges[:, 2]),
                                                 average_graph_degree=average_negative_edeges)

    directed_in_coefficient_val = directed_in_signed_coefficient(G, all_edges_weights,
                                                                 neg_rating_func=negative_in_rating,
                                                                 pos_rating_func=postive_in_rating)

    for _ in tqdm.tqdm(np.arange(number_of_couples - 1)):
        try:
            e1, e2, val = jaccard_coefficient_val.__next__()
            scores[int(e1)][int(e2)][0] = val
            scores[int(e2)][int(e1)][0] = val
            e1, e2, (val_n, val_p) = directed_in_coefficient_val.__next__()
            scores[int(e1)][int(e2)][1] = val_n
            scores[int(e1)][int(e2)][2] = val_p
        except StopIteration:
            print("finished looping")
            break
    np.savez("{0}_scores.npz".format(timestamp),scores=scores)
    # time_gradient_coefficient_val = time_gradient_coefficient(G)
    #
    # scores = np.array(
    #     [jaccard_coefficient_val, directed_in_positive_coefficient_val, directed_in_negative_coefficient_val])
    #
    # score = np.sum(weights * scores)
    return 1


def jaccard_coefficient(G):
    def predict(u, v):
        union_size = len(set(G[u]) | set(G[v]))
        if union_size == 0:
            return 0
        return len(list(nx.common_neighbors(G, u, v))) / union_size

    return _apply_prediction(G, predict, None)


def directed_in_signed_coefficient(G, all_edges_weights, neg_rating_func, pos_rating_func):
    '''
    Algo -
    G - the graph
    sign - {positive,negative}
    Nv - {n | (u,n) belongs to G}
    N_u_v - { n | n belongs Nv & w(n,v) is sign}
    score  = (rating_score * |N_u_v| / |Nv|)
    '''

    def predict(u, v):
        in_edges_of_v = all_edges_weights[all_edges_weights[:, 1] == v].astype(np.int)
        out_edges_of_u = all_edges_weights[all_edges_weights[:, 0] == u].astype(np.int)
        only_nodes_that_were_rated_by_u_and_rated_v = in_edges_of_v[np.isin(in_edges_of_v[:, 0], out_edges_of_u[:, 1])]
        score_n = 0
        score_p = 0

        Nv = len(out_edges_of_u)

        if len(only_nodes_that_were_rated_by_u_and_rated_v) == 0 or Nv == 0:
            return score_n, score_p

        # Positive
        positive_edges = only_nodes_that_were_rated_by_u_and_rated_v[
            only_nodes_that_were_rated_by_u_and_rated_v[:, 2] >= 0]
        total_positive_rating = np.sum(positive_edges[:, 2])
        if len(positive_edges) > 0:
            rating_positive_score = pos_rating_func(total_rating=total_positive_rating, degree=len(in_edges_of_v)) / 10
            N_u_v_p = len(positive_edges)
            score_p = (N_u_v_p * rating_positive_score) / Nv

        # Negative
        negative_edges = only_nodes_that_were_rated_by_u_and_rated_v[
            only_nodes_that_were_rated_by_u_and_rated_v[:, 2] < 0]
        total_negative_rating = np.sum(negative_edges[:, 2])
        if len(negative_edges) > 0:
            rating_negative_score = neg_rating_func(total_rating=total_negative_rating, degree=len(in_edges_of_v)) / 10
            N_u_v_n = len(negative_edges)
            score_n = (N_u_v_n * rating_negative_score) / Nv

        return score_n, score_p

    return _apply_prediction(G, predict, None)


def time_gradient_coefficient(G: nx.DiGraph):
    def predict(u, v):
        union_size = len(set(G[u]) | set(G[v]))
        if union_size == 0:
            return 0
        return len(list(nx.common_neighbors(G, u, v))) / union_size

    return _apply_prediction(G, predict, None)


def generate_in_rating_func(avareage_graph_rating, average_graph_degree):
    def calc_in_rating(total_rating, degree):
        return (avareage_graph_rating + total_rating) / (average_graph_degree + degree)

    return calc_in_rating
