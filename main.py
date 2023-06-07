import warnings
warnings.filterwarnings('ignore')
import os
import csv
import argparse
from baseline import *
from ccce import *

def simulate(sample_num, random_num, save_path):
    B = np.array([[0, 1, 1, 1, 0, 0, 0],  # binary matrix for graph(i,j) means i->j;
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 1, 0, 0, 0, 1, 0],
                  [0, 0, 1, 1, 0, 1, 1],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]])

    buy_index = 3
    unknown_node = [1]
    counter_dict = generate_counter_dict(B)

    '''
    # for interaction_effect simulation
    C = B * (np.random.rand(B.shape[0], B.shape[1]) * 0.5)  # [0,0.5]
    C[0, 3] = np.random.rand() * 0.4 + 0.6  # [0.6,1]
    W = C  # weights for each edge
    co_list = [2, 4]
    weight = 2
    node_beta = np.random.rand(B.shape[0]) * 3 - 2  # [-2,2]
    node_beta[0] = np.random.rand() + 1 
    counter_dict = generate_counter_dict_interaction_effect(W, node_beta, buy_index, co_list, weight)
    '''

    X = generate_data_counter(B, buy_index, sample_num, counter_dict)
    X_one = np.where(X[:, buy_index] == 1)[0]
    G = ig.Graph.Weighted_Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()

    pos_one = ordered_vertices.index(buy_index)
    parent_pos_one = np.array(ordered_vertices[:pos_one])


    tce_cal_dict = dict()
    cce_cal_dict = dict()
    rand1 = 0
    rand2 = 0
    posttce = 0
    cce = 0
    TR_postTCE = 0
    TR_CCE = 0


    rand_person_num = 0
    rand_or_not = 0
    for person in tqdm(X_one[:sample_num]):
        if rand_person_num < int(sample_num * 0.6):
            rand_or_not = 1
            rand_person_num += 1
        else:
            rand_or_not = 0
        # rand_one
        rand_pos = list(np.random.randint(low=0, high=len(parent_pos_one), size=(1)))
        rand_reason = list(parent_pos_one[rand_pos])
        x_person = copy.deepcopy(X[person, :])
        cf_value = get_each_counter_sample(rand_reason, x_person, B, counter_dict, rand_or_not)
        rand1 += 1 - cf_value[buy_index]

        # post_TCE
        x_person = copy.deepcopy(X[person, :])
        index, value = get_each_tce(x_person, unknown_node, X, buy_index, tce_cal_dict,
                                    B)  # get_each_tce(x_person,unknown_node,X,buy_index,cal_dict)
        dict_data = dict(zip(index, value))
        sort_dict = sorted(dict_data.items(), key=lambda item: item[1], reverse=1)  # 字典按照value降序排列
        reason_list = [sort_dict[0][0]]
        x_person = copy.deepcopy(X[person, :])
        cf_value = get_each_counter_sample(reason_list, x_person, B, counter_dict, rand_or_not)
        posttce += 1 - cf_value[buy_index]

        # rand_two
        rand_pos = list(np.random.randint(low=0, high=len(parent_pos_one), size=(2)))
        rand_reason = list(parent_pos_one[rand_pos])
        x_person = copy.deepcopy(X[person, :])
        cf_value = get_each_counter_sample(rand_reason, x_person, B, counter_dict, rand_or_not)
        rand2 += 1 - cf_value[buy_index]
        # two reason postTCE
        reason_list += [sort_dict[1][0]]
        x_person = copy.deepcopy(X[person, :])
        cf_value = get_each_counter_sample(reason_list, x_person, B, counter_dict, rand_or_not)
        TR_postTCE += 1 - cf_value[buy_index]

        # two reason cce
        x_person = copy.deepcopy(X[person, :])
        index, value = get_each_cce(x_person, unknown_node, X, buy_index, cce_cal_dict, 2, B)
        dict_data = dict(zip(index, value))
        sort_dict = sorted(dict_data.items(), key=lambda item: item[1], reverse=1)  # 字典按照value降序排列
        reason_list = list(sort_dict[0][0])
        x_person = copy.deepcopy(X[person, :])
        cf_value = get_each_counter_sample(reason_list, x_person, B, counter_dict, rand_or_not)
        TR_CCE += 1 - cf_value[buy_index]

        # CCE
        x_person = copy.deepcopy(X[person, :])
        index, value = get_each_cce(x_person, unknown_node, X, buy_index, cce_cal_dict, 1,
                                    B)  # get_each_cce(x_person,unknown_node,X,buy_index,cal_dict,topk):
        dict_data = dict(zip(index, value))
        sort_dict = sorted(dict_data.items(), key=lambda item: item[1], reverse=1)  # 字典按照value降序排列sort
        reason_list = list(sort_dict[0][0])
        x_person = copy.deepcopy(X[person, :])
        cf_value = get_each_counter_sample(reason_list, x_person, B, counter_dict, rand_or_not)
        cce += 1 - cf_value[buy_index]

    metrics = get_metrics(buy_index, B, X)
    baseline_result = get_baseline(metrics, X, B, buy_index, counter_dict,
                                   sample_num)  # (matrix,X,B,buy_index,counter_dict,sample_num):
    result = pd.DataFrame(
        columns=['sample_num', 'all_sample', 'random_seed', 'Rand1', 'PN', 'PS', 'PNS', 'RR', 'ACE', 'postTCE', 'CCE', 'Rand2', 'PN', 'PS', 'PNS', 'RR', 'ACE', 'postTCE', 'CCE'])
    result.loc[0] = [sample_num, X.shape[0], random_num, rand1 / sample_num] + baseline_result[:5] + [
        posttce / sample_num, cce / sample_num, rand2 / sample_num] + baseline_result[5:] + [TR_postTCE / sample_num,
                                                                                             TR_CCE / sample_num]
    # Save the result
    path = os.path.join(save_path, "result.csv")
    result.to_csv(path, index=False)
    metrics.to_csv(path, mode='a', index=True)
    file = open(path, mode='a+', encoding='utf-8', newline='')
    sWrite = csv.writer(file)

    for item in counter_dict.items():
        sWrite.writerow(item)
    file.close()


def real(random_num, save_path):
    B = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # binary matrix for graph(i,j) means i->j;
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1],
                  [1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


    buy_index = 1
    unknown_node = [3]
    X = generate_X()
    X_one = np.where(X[:, buy_index] == 1)[0]
    counter_dict = generate_counter_dict_real_data(B, X)

    G = ig.Graph.Weighted_Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    pos_one = ordered_vertices.index(buy_index)
    parent_pos_one = np.array(ordered_vertices[:pos_one])

    tce_cal_dict = dict()
    cce_cal_dict = dict()
    rand1 = 0
    rand2 = 0
    posttce = 0
    cce = 0
    TR_postTCE = 0
    TR_CCE = 0

    sample_num = len(X_one)
    # for person in tqdm(X_one[:sample_num]):
    for person in tqdm(X_one):
        # rand_one
        rand_pos = list(np.random.randint(low=0, high=len(parent_pos_one), size=(1)))
        rand_reason = list(parent_pos_one[rand_pos])
        x_person = copy.deepcopy(X[person, :])
        cf_value = get_each_counter_sample(rand_reason, x_person, B, counter_dict, 0)
        rand1 += 1 - cf_value[buy_index]
        # post_TCE
        x_person = copy.deepcopy(X[person, :])
        index, value = get_each_tce(x_person, unknown_node, X, buy_index, tce_cal_dict,
                                    B)  # get_each_tce(x_person,unknown_node,X,buy_index,cal_dict)
        dict_data = dict(zip(index, value))
        sort_dict = sorted(dict_data.items(), key=lambda item: item[1], reverse=1)  # 字典按照value降序排列
        reason_list = [sort_dict[0][0]]
        x_person = copy.deepcopy(X[person, :])
        cf_value = get_each_counter_sample(reason_list, x_person, B, counter_dict, 0)
        posttce += 1 - cf_value[buy_index]

        # rand_two
        rand_pos = list(np.random.randint(low=0, high=len(parent_pos_one), size=(2)))
        rand_reason = list(parent_pos_one[rand_pos])
        x_person = copy.deepcopy(X[person, :])
        cf_value = get_each_counter_sample(rand_reason, x_person, B, counter_dict, 0)
        rand2 += 1 - cf_value[buy_index]
        # two reason postTCE
        reason_list += [sort_dict[1][0]]
        x_person = copy.deepcopy(X[person, :])
        cf_value = get_each_counter_sample(reason_list, x_person, B, counter_dict, 0)
        TR_postTCE += 1 - cf_value[buy_index]

        # two reason cce
        x_person = copy.deepcopy(X[person, :])
        index, value = get_each_cce(x_person, unknown_node, X, buy_index, cce_cal_dict, 2, B)
        dict_data = dict(zip(index, value))
        sort_dict = sorted(dict_data.items(), key=lambda item: item[1], reverse=1)  # 字典按照value降序排列
        reason_list = list(sort_dict[0][0])
        x_person = copy.deepcopy(X[person, :])
        cf_value = get_each_counter_sample(reason_list, x_person, B, counter_dict, 0)
        TR_CCE += 1 - cf_value[buy_index]

        # CCE
        x_person = copy.deepcopy(X[person, :])
        index, value = get_each_cce(x_person, unknown_node, X, buy_index, cce_cal_dict, 1,
                                    B)  # get_each_cce(x_person,unknown_node,X,buy_index,cal_dict,topk):
        dict_data = dict(zip(index, value))
        sort_dict = sorted(dict_data.items(), key=lambda item: item[1], reverse=1)  # 字典按照value降序排列sort
        reason_list = list(sort_dict[0][0])
        x_person = copy.deepcopy(X[person, :])
        cf_value = get_each_counter_sample(reason_list, x_person, B, counter_dict, 0)
        cce += 1 - cf_value[buy_index]

    metrics = get_metrics(buy_index, B, X)
    baseline_result = get_baseline_real_data(metrics, X, B, buy_index, counter_dict,
                                   sample_num)
    result = pd.DataFrame(
        columns=['sample_num', 'std', 'random_seed', 'Rand1', 'PN', 'PS', 'PNS', 'RR', 'Post_prob', 'ACE', 'postTCE',
                 'CCE', 'Rand2', 'PN', 'PS', 'PNS', 'RR', 'Post_prob', 'ACE', 'postTCE', 'CCE'])
    result.loc[0] = [sample_num, "mean", random_num, rand1 / sample_num] + baseline_result[:6] + [posttce / sample_num, cce / sample_num, rand2 / sample_num] + baseline_result[6:] + [TR_postTCE / sample_num, TR_CCE / sample_num]
    path = os.path.join(save_path, "result.csv")
    result.to_csv(path, index=False)
    metrics.to_csv(path, mode='a', index=True)
    file = open(path, mode='a+', encoding='utf-8', newline='')
    sWrite = csv.writer(file)

    for item in counter_dict.items():
        sWrite.writerow(item)

    file.close()

    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_num", help="random number of seed", type=int)
    parser.add_argument("--sample_num", help="the number of samples", type=int)
    parser.add_argument("--save_path", help="the path of result file", type=str)
    # args = parser.parse_args()
    args = parser.parse_args(args=['--random_num', '3000', '--sample_num', '1000', '--save_path', './'])
    np.random.seed(args.random_num)
    # for simulation setting
    simulate(args.sample_num, args.random_num, args.save_path)

    # for real data setting
    # real(args.random_num, args.save_path)