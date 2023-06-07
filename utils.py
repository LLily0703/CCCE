import numpy as np
import pandas as pd
import igraph as ig
import itertools as it
import math
import networkx as nx


def is_dag(W):
    """
    Justify the graph
    :param W: adjacent matrix of graph, numpy array type
    :return: whether the graph is a dag
    """
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def logistic(x):
    """
    Calculate the value
    :param x: a real number
    :return: the output of logistic function
    """
    return 1 / (1 + np.exp(-x))


def generate_counter_dict(adjacent_matrix):
    """
    Random generate the counterfactual weight of each edge according to the input graph
    :param adjacent_matrix: numpy array, 0 if there is no edge between two nodes
    :return: the dictionary {node:[weight list]}
    """
    G = ig.Graph.Weighted_Adjacency(adjacent_matrix.tolist())
    node_num = adjacent_matrix.shape[0]
    counter_dict = dict()
    for pos in range(node_num):
        parent_num = len(G.neighbors(pos, mode=ig.IN))
        prob = np.random.rand(1) if parent_num == 0 else np.random.rand(int(math.pow(2, parent_num)))
        counter_dict[pos] = prob
    return counter_dict


def generate_data_counter(adjacent_matrix, y, n, counter_dict):
    """
    Generate simulation data by the topo order
    :param adjacent_matrix: numpy array
    :param y: the outcome node id
    :param n: the number of samples will be generated according to the input graph
    :param counter_dict: a dictionary of probability
    :return:
    """
    if not is_dag(adjacent_matrix):
        raise ValueError('The graph must be a DAG')

    G = ig.Graph.Weighted_Adjacency(adjacent_matrix.tolist())
    ordered_vertices = G.topological_sorting()
    nodes_num = adjacent_matrix.shape[0]
    sample_num = 0
    X = np.empty((0, nodes_num))

    def _simulate_single_equation(X, w_dict):
        """
        Generate specific node value of n samples
        :param X: numpy array of [n*p], p denotes the parent number of the current node
        :param w_dict: counter_dict of the current node
        :return: numpy array of [n*1]
        """
        w = np.zeros(X.shape[0])
        for j in range(X.shape[0]):
            i = 0
            for k in range(X.shape[1]):
                i += X[j][k] * math.pow(2, k)
            w[j] = w_dict[int(i)]
        x = np.random.binomial(1, w)  # [sample_num
        return x

    while sample_num < n:
        tmp_X = np.zeros([n, nodes_num])
        for j in ordered_vertices:
            parents = G.neighbors(j, mode=ig.IN)
            parents.sort(reverse=True)
            if parents == None:
                tmp_X[:, j] = np.random.binomial(1, np.full((n), counter_dict[j][0]))
            else:
                tmp_X[:, j] = _simulate_single_equation(tmp_X[:, parents], counter_dict[j])  # n*p p*1
        y_one_pos = np.where(tmp_X[:, y] == 1)[0]
        tmp_len = 0
        if sample_num + len(y_one_pos) >= n:
            tmp_len = n - sample_num
            sample_num = n
        else:
            tmp_len = len(y_one_pos)
            sample_num += tmp_len
        X = np.concatenate((X, tmp_X), axis=0)
    return X


def get_joint_prob(var_list, val_list, X):
    """
    P(var_1=val_1, var_2=val_2)
    :param var_list: the specific node list
    :param val_list: the value of the node list
    :param X: the generated samples
    :return: the probability
    """
    return len(np.where((X[:, var_list] == list(val_list)).all(axis=1))[0]) / X.shape[0]


def get_cond_prob(prob, cond, prob_val, cond_val, X):
    """
    :return: P(prob_1=prob_val_1, prob_2=prob_val_2 | cond_1=cond_val_1, cond_2=cond_val_2)
    """
    XY = np.append(X[:, cond].reshape(X.shape[0], -1), X[:, prob].reshape(-1, len(prob)), axis=1)
    Y_num = len(list(prob)) * (-1)
    cond_index = np.where((XY[:, :Y_num] == cond_val).all(axis=1))[0]
    # 如果xpa没有当前的排列
    if len(cond_index) == 0:
        return 0
    prob_num = len(np.where((XY[cond_index, Y_num] == prob_val))[0])
    return prob_num / len(cond_index)


def get_real_cond_prob(prob, prob_val, cond_val, W, node_beta):
    """
    According to the adjacent weighted matrix and node_bata list, get the conditional probability
    :param cond_val: node sorted by topo order
    :return: the ground-truth of the conditional probability
    """
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = list(G.topological_sorting())
    parents = G.neighbors(prob, mode=ig.IN)
    val = []
    for pos in parents:
        val += [cond_val[ordered_vertices.index(pos)]]
    ans = logistic(np.dot(W[parents, prob].T, val) + node_beta[prob])
    if prob_val == 1:
        return ans
    else:
        return 1 - ans


def get_do_prob(y, xk, xpa, X, mode):
    """
    Get result after do operator
    :param y: the outcome node
    :param xk: the node used do operator
    :param xpa: the parent list of xk node
    :param X: the generated data
    :param mode: bool, do(xk = mode)
    :return: the probability
    """
    if len(xpa) == 0:
        return get_cond_prob([y], [xk], [1], [mode], X)
    pa_permutation = list(it.product(range(2), repeat=len(xpa)))
    ans = 0
    for perm in pa_permutation:
        ans += get_cond_prob([y], [xk] + list(xpa), [1], [mode] + list(perm), X) * get_joint_prob(xpa, perm, X)
    return ans


def generate_counter_dict_interaction_effect(W, node_beta, y, co_list, weight):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    node_num = W.shape[0]
    counter_dict = dict()
    for pos in range(node_num):
        parent = G.neighbors(pos, mode=ig.IN)
        parent_num = len(parent)
        if parent_num == 0:
            prob = [logistic(node_beta[pos])]
        else:
            prob = []
            pa_permutation = list(it.product(range(2), repeat=parent_num))
            if pos == y:
                co_one = [parent.index(i) for i in co_list]
                for pa in pa_permutation:
                    tmp_pa = np.array(list(pa))
                    # print()
                    if sum(tmp_pa[co_one]) == len(co_list):
                        # print(logistic(sum(np.multiply(tmp_pa,W[parent,pos]))+node_beta[pos]+weight))
                        prob += [logistic(sum(np.multiply(tmp_pa, W[parent, pos])) + node_beta[pos] + weight)]
                    else:
                        prob += [logistic(sum(np.multiply(tmp_pa, W[parent, pos])) + node_beta[pos])]
            else:
                for pa in pa_permutation:
                    tmp_pa = np.array(list(pa))
                    prob += [logistic(sum(np.multiply(tmp_pa, W[parent, pos])) + node_beta[pos])]
        counter_dict[pos] = prob

    return counter_dict


def change_x(X, std):
    for col in range(X.shape[1]):
        X[:, col] = std[col] < X[:, col]
    return X


def generate_X():
    pth = './sachs/continuous'
    Data = np.load(pth + '/data1.npy')
    Gmat = np.load(pth + '/DAG1.npy')
    G = nx.DiGraph(Gmat)
    df = pd.DataFrame(Data)
    df_mean = df.median()
    X = change_x(Data, np.array(df_mean))
    return X


def generate_counter_dict_real_data(B, X):
    G = ig.Graph.Weighted_Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    node_num = B.shape[0]
    counter_dict = dict()
    for pos in range(node_num):
        parent = G.neighbors(pos, mode=ig.IN)
        parent_num = len(parent)
        if parent_num == 0:
            prob = [get_cond_prob([pos], [], [1], [], X)]
        else:
            prob = []
            pa_permutation = list(it.product(range(2), repeat=parent_num))
            for pa in pa_permutation:
                tmp_pa = np.array(list(pa))
                prob += [get_cond_prob([pos], parent, [1], tmp_pa, X)]
        counter_dict[pos] = prob

    return counter_dict




