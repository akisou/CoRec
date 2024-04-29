import pandas as pd
import numpy as np

round_num = 3
def eval_precision(k_list, user_list, items_rank, test_record):
    # precision@k evaluation
    precision_list = {k: [] for k in k_list}
    for k in k_list:
        for user in user_list:
            hit_num = len(set(items_rank[user][:k]) & set(test_record[user]))
            precision_list[k].append(hit_num / k)

    precision = [np.round(np.mean(precision_list[k]), round_num) for k in k_list]

    return precision

def eval_recall(k_list, user_list, items_rank, test_record):
    # recall@k evaluation
    recall_list = {k: [] for k in k_list}
    for k in k_list:
        for user in user_list:
            hit_num = len(set(items_rank[user][:k]) & set(test_record[user]))
            recall_list[k].append(hit_num / len(set(test_record[user])))

    recall = [np.round(np.mean(recall_list[k]), round_num) for k in k_list]

    return recall

def eval_NDCG(k_list, user_list, items_rank, test_record):
    # NDCG@k evaluation
    NDCG_list = {k: [] for k in k_list}
    for k in k_list:
        for user in user_list:
            predict_items = items_rank[user][:k]
            true_items = test_record[user]

            # DCG, IDCG, NDCG
            dcg = DCG(predict_items, true_items)  # calculate DCG
            idcg = IDCG(predict_items, true_items)  # calculate IDCG
            ndcg = 0
            if dcg == 0 or idcg == 0:
                ndcg = 0
            else:
                ndcg = dcg / idcg
            NDCG_list[k].append(ndcg)

    NDCG = [np.round(np.mean(NDCG_list[k]), round_num) for k in k_list]

    return NDCG

def eval_MAP(k_list, user_list, items_rank, test_record):
    # MAP@k evaluation
    AP_list = {k: [] for k in k_list}
    for k in k_list:
        for user in user_list:
            max_k = min(k, len(test_record[user]))
            precision_list = []
            for sub_k in range(1, max_k+1):
                hit_num = len(set(items_rank[user][:sub_k]) & set(test_record[user]))
                precision_list.append(hit_num / sub_k)
            AP_list[k].append(np.mean(precision_list))

    MAP = [np.round(np.mean(AP_list[k]), round_num) for k in k_list]

    return MAP

def eval_MRR(k_list, user_list, items_rank, test_record):
    # MRR@k evaluation
    MRR_list = {k: [] for k in k_list}
    for k in k_list:
        for user in user_list:
            items = list(items_rank[user][:k])
            if_shot = [1 if iter in list(set(test_record[user])) else 0 for iter in items]
            MRR_list[k].append(np.sum([if_shot[i] * (1/(i+1)) for i in range(len(if_shot))]))

    MRR = [np.round(np.mean(MRR_list[k]), round_num) for k in k_list]

    return MRR

def DCG(A, test_set):
    # ------ 计算 DCG ------ #
    dcg = 0
    for i in range(len(A)):
        # 给r_i赋值，若r_i在测试集中则为1，否则为0
        r_i = 0
        if A[i] in test_set:
            r_i = 1
        dcg += (2 ** r_i - 1) / np.log2((i + 1) + 1) # (i+1)是因为下标从0开始
    return dcg

def IDCG(A, test_set):
    # ------ 将在测试中的a排到前面去，然后再计算DCG ------ #
    A_temp_1 = [] # 临时A，用于存储r_i为1的a
    A_temp_0 = []  # 临时A，用于存储r_i为0的a
    for a in A:
        if a in test_set:
            # 若a在测试集中则追加到A_temp_1中
            A_temp_1.append(a)
        else:
            # 若a不在测试集中则追加到A_temp_0中
            A_temp_0.append(a)
    A_temp_1.extend(A_temp_0)
    idcg = DCG(A_temp_1, test_set)
    return idcg