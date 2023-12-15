import pandas as pd
import copy
from tqdm import tqdm
from utils import *


def get_metrics(y, W, X):
    """
    Get result metrics of each baseline method:'Y_0', 'Y_1', 'Post_Prob', 'ACE', 'RR', 'PN', 'PS', 'PNS'
    :param y: the outcome node id
    :param W: the adjacent weighted matrix
    :param X: the generated samples
    :return:
    """
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    pos_one = ordered_vertices.index(y)
    parents = ordered_vertices[:pos_one]
    matrix = pd.DataFrame(np.zeros(len(parents) * 8).reshape(len(parents), 8), index=parents,
                          columns=['Y_0', 'Y_1', 'Post_Prob', 'ACE', 'RR', 'PN', 'PS', 'PNS'])

    def y0(x):
        return get_do_prob(y, x, G.neighbors(x, mode=ig.IN), X, 0)

    matrix['Y_0'] = list(pd.DataFrame(map(y0, parents)).iloc[:, 0])

    def y1(x):
        return get_do_prob(y, x, G.neighbors(x, mode=ig.IN), X, 1)

    matrix['Y_1'] = list(pd.DataFrame(map(y1, parents)).iloc[:, 0])

    def Post_prob(x):
        return get_cond_prob([x], y, 1, [1], X)

    matrix['Post_Prob'] = list(pd.DataFrame(map(Post_prob, parents)).iloc[:, 0])
    matrix['ACE'] = matrix['Y_1'] - matrix['Y_0']
    matrix['RR'] = (matrix['Y_1'] / matrix['Y_0']).replace(np.inf, 0)

    def PN(x):
        return 1 - get_cond_prob([y], x, 1, [0], X) / get_cond_prob([y], x, 1, [1],
                                                                    X)  # get_real_cond_prob(prob,prob_val,cond_val,W,node_beta):

    matrix['PN'] = list(pd.DataFrame(map(PN, parents)).iloc[:, 0])

    def PS(x):
        return 1 - get_cond_prob([y], x, 0, [1], X) / get_cond_prob([y], x, 0, [0], X)

    matrix['PS'] = list(pd.DataFrame(map(PS, parents)).iloc[:, 0])

    def PNS(x):
        return get_joint_prob([x, y], [1, 1], X) * PN(x) + get_joint_prob([x, y], [0, 0], X) * PS(
            x)  # get_joint_prob(var_list,val_list,X):

    matrix['PNS'] = list(pd.DataFrame(map(PNS, parents)).iloc[:, 0])

    return matrix




def prob_postTCE(Xk, y, x, X, topo):
    """
    the postTCE method
    :param Xk: single value, postTCE is used to justify the rationality of thinking Xk causes y
    :param y: the outcome node id
    :param x: the specific value of X
    :param X: the generated data
    :param topo: the topo order before y
    :return:
    """
    Xk_index = topo.index(Xk)
    ak = list(x[:Xk_index])
    dk = x[Xk_index + 1:]
    prob_up = 0
    less_permutation = list(it.product(range(2), repeat=int(sum(dk))))
    one_index = list(np.where(np.array(dk) == 1))[0]

    for perm in less_permutation:
        ck = np.zeros(len(dk))
        ck[one_index] = perm
        first = get_cond_prob([y], topo, [1], list(ak) + [0] + list(ck), X)
        second = 1
        for i in range(Xk_index + 1, len(topo)):
            next_x = None
            if len(ck[:i - Xk_index - 1]) == 0:
                next_x = ak + [0]
            else:
                next_x = ak + [0] + list(ck[:i - Xk_index - 1])

            sec_prob_up = get_cond_prob([topo[i]], topo[:i], [1], next_x, X)
            sec_prob_down = get_cond_prob([topo[i]], topo[:i], [1], x[:i], X)

            temp = (1 - x[i]) + x[i] * (1 - ck[i - Xk_index - 1]) + x[i] * pow(-1, 1 - ck[
                i - Xk_index - 1]) * sec_prob_up / sec_prob_down
            second *= temp

        prob_up += first * second

    prob_down = get_cond_prob([y], topo, [1], x, X)
    return x[Xk_index] * (1 - prob_up / prob_down)


def get_each_tce(x_person, unknown_node, X, buy_index, cal_dict, W):
    """
    Find the cause of result 'buy_index' for the current user with TCE
    :param unknown_node: List of unobserved nodes, possibly 0 or 1
    :return: List of postTCE values attributed to each node
    """

    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    value = []

    pain_index = list(range(len(x_person)))
    cond_index = [x for x in pain_index if x not in unknown_node]

    pos_one = ordered_vertices.index(buy_index)
    parent_pos_one = ordered_vertices[:pos_one]
    for pa in parent_pos_one:
        post_tce = 0
        tmp_dict_key = list(copy.deepcopy(x_person[cond_index]))
        tmp_dict_key.extend([pa])
        tmp_dict_key.extend([x_person[pa]])
        tmp_dict_key = tuple(tmp_dict_key)
        if tmp_dict_key in cal_dict:
            post_tce = cal_dict[tmp_dict_key]
        else:
            if len(unknown_node):
                unknown_value = list(it.product(range(2), repeat=len(unknown_node)))
                for tmp_value in unknown_value:
                    x_person[unknown_node] = tmp_value
                    tmp_before = copy.deepcopy(x_person[ordered_vertices[:pos_one]])
                    cond_prob = get_cond_prob(unknown_node, cond_index, list(tmp_value), x_person[cond_index],
                                              X)
                    post_tce += cond_prob * prob_postTCE(pa, buy_index, tmp_before, X,
                                                         ordered_vertices[:pos_one])  # postTCE(Xk,y,x,X,topo)
            else:
                tmp_before = copy.deepcopy(x_person[ordered_vertices[:pos_one]])
                post_tce = prob_postTCE(pa, buy_index, tmp_before, X,
                                        ordered_vertices[:pos_one])  # postTCE(Xk,y,x,X,topo)
            cal_dict[tmp_dict_key] = post_tce
        value += [post_tce]
    return parent_pos_one, value



def get_each_counter_sample(reason_list, cf_value, B, counter_dict, random_or_not):
    G = ig.Graph.Weighted_Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    reason_in_order_index = []
    change = set()
    for tmp_index in reason_list:
        if cf_value[tmp_index] == 0:
            continue
        reason_in_order_index += [ordered_vertices.index(tmp_index)]
        cf_value[tmp_index] = 0  # reverse
        change.add(tmp_index)

    if len(reason_in_order_index) == 0:
        return cf_value
    start_change_index = min(reason_in_order_index)
    if random_or_not == 0:
        for tmp_index in range(start_change_index + 1, len(ordered_vertices)):
            if ordered_vertices[tmp_index] not in change:
                parent = G.neighbors(ordered_vertices[tmp_index],mode=ig.IN)
                parent.sort(reverse=True)
                if len(change & set(parent)) != 0:
                    i = 0
                    for k in range(len(parent)):
                        i += cf_value[parent[k]] * math.pow(2, k)
                    cond_prob = counter_dict[ordered_vertices[tmp_index]][int(i)]
                    value = np.random.binomial(1, cond_prob)
                    if cf_value[ordered_vertices[tmp_index]] == 1 and value == 0:
                        cf_value[ordered_vertices[tmp_index]] = value  # Change the value of the counterfactual sample
                        change.add(ordered_vertices[tmp_index])
        return cf_value
    else:
        for tmp_index in range(start_change_index + 1, len(ordered_vertices)):
            if ordered_vertices[tmp_index] not in change:
                parent = G.neighbors(ordered_vertices[tmp_index],
                                     mode=ig.IN)
                parent.sort(reverse=True)
                if len(change & set(parent)) != 0:
                    i = 0
                    for k in range(len(parent)):
                        i += cf_value[parent[k]] * math.pow(2, k)
                    cond_prob = counter_dict[ordered_vertices[tmp_index]][int(i)]
                    value = np.random.binomial(1, cond_prob)
                    if cf_value[ordered_vertices[tmp_index]] != value:
                        cf_value[ordered_vertices[tmp_index]] = value  # Change the value of the counterfactual sample
                        change.add(ordered_vertices[tmp_index])
        return cf_value


def get_one_baseline(x_person, matrix, baseline, counter_dict, B, buy_index, rand_or_not):
    base1 = 0
    base2 = 0

    pn_value = dict(zip(list(matrix.index.values), list(matrix[baseline])))
    sort_dict = sorted(pn_value.items(), key=lambda item:item[1], reverse=1)
    if len(sort_dict) == 0:
        return 0, 0
    reason_list = [sort_dict[0][0]]
    person = copy.deepcopy(x_person)
    cf_value = get_each_counter_sample(reason_list, person, B, counter_dict, rand_or_not)
    base1 += 1 - cf_value[buy_index]

    if len(sort_dict) > 1:
        reason_list += [sort_dict[1][0]]
        person = copy.deepcopy(x_person)
        cf_value = get_each_counter_sample(reason_list, person, B, counter_dict, rand_or_not)
    base2 += 1 - cf_value[buy_index]
    return base1, base2


def get_baseline(matrix, X, B, buy_index, counter_dict, sample_num):
    G = ig.Graph.Weighted_Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    pos_one = ordered_vertices.index(buy_index)
    X_one = np.where(X[:, buy_index] == 1)[0]

    pn = 0
    pn2 = 0
    ps = 0
    ps2 = 0
    pns = 0
    pns2 = 0
    rr = 0
    rr2 = 0
    ace = 0
    ace2 = 0

    rand_person_num = 0
    rand_or_not = 0
    for person in tqdm(X_one[:sample_num]):
        if rand_person_num < int(sample_num * 0.6):
            rand_or_not = 1
            rand_person_num += 1
        else:
            rand_or_not = 0
        one, two = get_one_baseline(X[person, :], matrix, "PN", counter_dict, B, buy_index, rand_or_not)
        pn += one
        pn2 += two

        one, two = get_one_baseline(X[person, :], matrix, "PS", counter_dict, B, buy_index, rand_or_not)
        ps += one
        ps2 += two
        one, two = get_one_baseline(X[person, :], matrix, "PNS", counter_dict, B, buy_index, rand_or_not)
        pns += one
        pns2 += two
        one, two = get_one_baseline(X[person, :], matrix, "RR", counter_dict, B, buy_index, rand_or_not)
        rr += one
        rr2 += two
        one, two = get_one_baseline(X[person, :], matrix, "ACE", counter_dict, B, buy_index, rand_or_not)
        ace += one
        ace2 += two

    return list(np.array([pn, ps, pns, rr, ace, pn2, ps2, pns2, rr2, ace2]) / sample_num)


def get_baseline_real_data(matrix, X, B, buy_index, counter_dict, sample_num):
    G = ig.Graph.Weighted_Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    pos_one = ordered_vertices.index(buy_index)
    parent_pos_one = np.array(ordered_vertices[:pos_one])
    X_one = np.where(X[:, buy_index] == 1)[0]

    pn = 0
    pn2 = 0
    ps = 0
    ps2 = 0
    pns = 0
    pns2 = 0
    rr = 0
    rr2 = 0
    ace = 0
    ace2 = 0
    pb = 0
    pb2 = 0

    for person in tqdm(X_one[:sample_num]):
        one, two = get_one_baseline(X[person, :], matrix, "PN", counter_dict, B, buy_index, 0)
        pn += one
        pn2 += two

        one, two = get_one_baseline(X[person, :], matrix, "PS", counter_dict, B, buy_index, 0)
        ps += one
        ps2 += two
        one, two = get_one_baseline(X[person, :], matrix, "PNS", counter_dict, B, buy_index, 0)
        pns += one
        pns2 += two
        one, two = get_one_baseline(X[person, :], matrix, "RR", counter_dict, B, buy_index, 0)
        rr += one
        rr2 += two

        one, two = get_one_baseline(X[person, :], matrix, "Post_Prob", counter_dict, B, buy_index, 0)
        pb += one
        pb2 += two

        one, two = get_one_baseline(X[person, :], matrix, "ACE", counter_dict, B, buy_index, 0)
        ace += one
        ace2 += two

    # print(pn,ps,pns,rr,ace)
    return list(np.array([pn, ps, pns, rr, pb, ace, pn2, ps2, pns2, rr2, pb2, ace2]) / sample_num)