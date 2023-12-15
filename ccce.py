
import copy
from itertools import combinations
from utils import *


def prob_CCE(Xs, y, x, X, topo, w):
    """
    CCE uses to evaluate the reasonableness of recommending y to the user with this Xs as the reason
    :param Xs: The index of the selected set of reasons
    :param y: outcome node
    :param x: List of topological sequences taking values before y
    :param X:
    :param topo: Topological order node list before y
    :param w: The values of all observable variables, i.e., a record (for making personalized inferences)
    :return: a CCE value
    """
    min_index = -1
    set_xs_pos = []
    for pos in topo:
        if pos in Xs:
            if min_index == -1:
                min_index = topo.index(pos)
            set_xs_pos += [topo.index(pos)]

    Xk_index = min_index
    if Xk_index == -1:
        print("error!Xk_index=-1")
        return 0
    ak = list(x[:Xk_index])

    dk_0 = np.array(x)
    dk_0[set_xs_pos] = 0
    dk_0 = dk_0[Xk_index:].tolist()

    prob_up = 0
    lseq_permutation = list(it.product(range(2), repeat=int(sum(dk_0))))
    one_index = np.where(np.array(dk_0) == 1)
    for perm in lseq_permutation:
        ck = np.zeros(len(dk_0))
        ck[one_index] = perm
        ck = ck.tolist()

        first = get_cond_prob([y], topo, [1], ak + list(ck), X)
        second = 1
        for i in range(Xk_index, len(topo)):
            if topo[i] in Xs:
                continue
            next_x = None
            if len(ak) == 0:
                next_x = ck[:i - Xk_index]
            else:
                next_x = ak + ck[:i - Xk_index]
            if len(next_x) != 0:
                sec_prob_up = get_cond_prob([topo[i]], topo[:i], [1], next_x, X)
                sec_prob_down = get_cond_prob([topo[i]], topo[:i], [1], x[:i], X)
                temp = 1 - x[i] * ck[i - Xk_index] + x[i] * pow(-1, 1 - ck[i - Xk_index]) * sec_prob_up / sec_prob_down
                second *= temp
        prob_up += first * second

    prob_down = get_cond_prob([y], topo, [1], x, X)
    CCE_1 = 1 - prob_up / prob_down
    ob_w = w.tolist()
    del ob_w[y]

    if w[y] == 1:
        return CCE_1
    else:
        return 0  # CCE_0


def combine(tmp_list, n):
    # Get all possible combinations of the current sequence
    tmplist2 = []
    for c in combinations(tmp_list, n):
        tmplist2.append(c)
    return tmplist2


def get_each_cce(x_person, unknown_node, X, buy_index, cal_dict, topk, W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    value = []

    pain_index = list(range(len(x_person)))
    cond_index = [x for x in pain_index if x not in unknown_node]

    pos_one = ordered_vertices.index(buy_index)
    parent_pos_one = ordered_vertices[:pos_one]
    couple_list = combine(parent_pos_one, topk)
    for pa in couple_list:
        post_tce = 0
        tmp_dict_key = list(copy.deepcopy(x_person[cond_index]))
        tmp_dict_key.extend([pa])
        if topk == 1:
            tmp_dict_key.extend([x_person[pa]])
        else:
            tmp_dict_key.extend(list(x_person[list(pa)]))
        tmp_dict_key = tuple(tmp_dict_key)
        if tmp_dict_key in cal_dict:
            post_tce = cal_dict[tmp_dict_key]
        else:
            if len(unknown_node):
                unknown_value = list(it.product(range(2), repeat=len(unknown_node)))
                for tmp_value in unknown_value:
                    x_person[unknown_node] = tmp_value
                    tmp_before = copy.deepcopy(x_person[ordered_vertices[:pos_one]])
                    cond_prob = get_cond_prob(unknown_node, cond_index, tmp_value, x_person[cond_index], X)
                    # print([pa],ordered_vertices[:pos_one])  CCE(Xs,y,x,X,topo,w,back_topo):
                    # print(prob_CCE(list(pa),buy_index,tmp_before,X,ordered_vertices[:pos_one],x_person,ordered_vertices[pos_one+1:]))
                    post_tce += cond_prob * prob_CCE(list(pa), buy_index, tmp_before, X, ordered_vertices[:pos_one],
                                                     x_person, ordered_vertices[pos_one + 1:])
            else:
                tmp_before = copy.deepcopy(x_person[ordered_vertices[:pos_one]])
                post_tce += prob_CCE(list(pa), buy_index, tmp_before, X, ordered_vertices[:pos_one], x_person,
                                     ordered_vertices[pos_one + 1:])
            cal_dict[tmp_dict_key] = post_tce
        value += [post_tce]
    return couple_list, value