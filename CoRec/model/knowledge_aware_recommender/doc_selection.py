# -*- coding: utf-8 -*-

r"""
the stage 2 of CoRec algorithm
doctor selection based on the embeddings of doctors and patients, and the cowork network of doctors
"""
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import networkx as nx
import collections
import operator
import logging
import os
from itertools import combinations
from .metric_evaluation import eval_MRR, eval_NDCG, eval_MAP, eval_recall, eval_precision
from CoRec.utils import get_local_time
from .LDA.LDA import LDA
from tqdm import tqdm
from CoRec.utils import (
    set_color
)

class DoctSelection():
    def __init__(self, config, dataset, working_data, saved_file, model):
        self.config = config
        self.data_path = config.final_config_dict['data_path']
        # sigmoid
        self.sigmoid = torch.nn.Sigmoid()

        # load best model
        self.checkpoint_dir = config["checkpoint_dir"]
        saved_model_file = "{}-{}.pth".format(self.config["model"], get_local_time())
        self.saved_model_file = saved_file
        self.model = self.load_model(model, True)

        # dataset related
        dataset = dataset._dataset
        self.n_user = dataset.user_num
        self.n_item = dataset.item_num
        self.user_feat = self.get_feat(dataset.user_feat)
        self.item_feat = self.get_feat(dataset.item_feat)
        self.inter_feat = self.get_feat(dataset.inter_feat)
        self.kg_graph = self.construct_net(dataset.kg_feat)
        self.field2token_id = dataset.field2token_id
        self.depart2item = self.get_depart2item(self.item_feat)
        self.item2depart = self.get_item2depart(self.item_feat)
        self.scores_dict = self.get_DP_scores(self.model)

        # test related
        self.working_interation = self.get_feat(working_data._dataset.inter_feat)
        self.trues = self.get_truth(self.inter_feat, self.working_interation)

        # LDA related
        lda_class = LDA(self.data_path + '/LDA_related/', working_data._dataset.inter_feat, self.config.final_config_dict['USER_ID_FIELD'],
                        self.config.final_config_dict['QUERY_FIELD'], 30)
        self.user_ldaclass_df = lda_class.predict()
        self.user2ldaclass = self.user2ldaclass_process(self.user_ldaclass_df)
        self.ldaclass2user = self.ldaclass2user_process(self.user_ldaclass_df)
        self.ldaclass2department = self.ldaclass_cover_departments(self.ldaclass2user)

    def load_model(self, model, load_best_model):
        if load_best_model:
            checkpoint_file = self.saved_model_file
            checkpoint = torch.load(checkpoint_file, map_location=self.config['device'])
            model.load_state_dict(checkpoint["state_dict"])
            model.load_other_parameter(checkpoint.get("other_parameter"))

        model.eval()

        return model

    def get_truth(self, inter_train, inter_test):
        user_id = self.config.final_config_dict['USER_ID_FIELD']
        item_id = self.config.final_config_dict['ITEM_ID_FIELD']

        # truth = dict()
        # # user2doctors
        # user2doctors = pd.read_csv(self.data_path + '/user_id_closed_doctors.txt', sep='\t')
        # for i in range(len(user2doctors)):
        #     user = self.field2token_id[user_id][str(user2doctors.loc[i, 'user_id'])]
        #     docts = [self.field2token_id[item_id][str(elem)] for elem in eval(user2doctors.loc[i, 'doctor_closed'])]
        #     truth[user] = docts

        users = list(set(self.user_feat[user_id].values))
        truth = dict()
        for user in users:
            truth[user] = list(set(inter_test[inter_test[user_id] == user][item_id].values))
            truth[user].extend(list(set(inter_train[inter_train[user_id] == user][item_id].values)))
            truth[user] = list(set(truth[user]))

        return truth

    def user2ldaclass_process(self, user_class_df):
        user_id = self.config.final_config_dict['USER_ID_FIELD']
        field2userid = self.field2token_id[user_id]
        return {field2userid[str(user_class_df.loc[i, user_id])]:
                    user_class_df.loc[i, 'class'] for i in range(len(user_class_df))}

    def ldaclass2user_process(self, user_class_df):
        user_id = self.config.final_config_dict['USER_ID_FIELD']
        field2userid = self.field2token_id[user_id]
        ldaclass = sorted(list(set(user_class_df['class'])))
        group_class = user_class_df.groupby(by='class')
        return {elem: [field2userid[str(group_elem)]
                       for group_elem in list(group_class.get_group(elem)[user_id].values)] for elem in ldaclass
                }

    def ldaclass_cover_departments(self, ldaclass2user):
        # find the departments of all doctors of queries in every ldaclass
        user_id = self.config.final_config_dict['USER_ID_FIELD']
        item_id = self.config.final_config_dict['ITEM_ID_FIELD']
        dataset_inter = self.inter_feat
        ldaclass2department = dict()
        for ldaclass in ldaclass2user.keys():
            users_in_ladclass = ldaclass2user[ldaclass]
            departments_related = list(set([self.item2depart[item] for user in users_in_ladclass
                                            for item in list(dataset_inter[dataset_inter[user_id] == user][item_id].values)
                                           if item in self.item2depart.keys()]))
            ldaclass2department[ldaclass] = departments_related

        return ldaclass2department

    def cross_department_recommendation(self, logger, k_list):
        # for a user/patient, find the set of items/doctors through the algorithm
        user_id = self.config.final_config_dict['USER_ID_FIELD']
        users = list(set(self.working_interation[user_id].values))[:100]

        trues = self.trues
        # original evaluation
        logger.info(set_color(f"original evaluation: ", "red"))
        scores_dict_original_kumi = {key: sorted([(i, self.scores_dict[key][i])
                                                  for i in range(len(self.scores_dict[key]))],
                                                 key=operator.itemgetter(1), reverse=True)
                                     for key in users}
        scores_dict_original = {key: [elem[0] for elem in scores_dict_original_kumi[key]]
                                for key in users}
        self.eval_all(logger, k_list, users, scores_dict_original, trues)

        # ramdom pick evaluation
        logger.info(set_color(f"random evaluation: ", "red"))
        scores_dict_random_kumi = {
            key: np.random.permutation([(i, self.scores_dict[key][i].cpu()) for i in range(len(self.scores_dict[key]))]).tolist() for key in users}
        scores_dict_random = {key: [int(elem[0]) for elem in scores_dict_random_kumi[key]]
                              for key in users}
        self.eval_all(logger, k_list, users, scores_dict_random, trues)

        logger.info(set_color(f"cross_department evaluation: ", "red"))
        final_recommend_full_set = dict()
        max_k = max(k_list)
        loop = tqdm((users), total=len(users))
        for user in loop:
            items_recommend_all_departments = self.get_topk_items_based_on_department(user, max_k)
            items_all = [elem2 for elem1 in items_recommend_all_departments for elem2 in elem1]
            # final topk set
            final_recommend_set = []
            # candidate
            candidate_items_L = []
            # div, inc

            while len(final_recommend_set) < max_k:
                if len(final_recommend_set) == 0:
                    candidate_items_L = sorted(items_all, key=operator.itemgetter(1), reverse=True)
                    final_recommend_set.append(candidate_items_L[0][0])
                    candidate_items_L.pop(0)
                else:
                    div_inc_value = []
                    for item in candidate_items_L:
                        div = self.cal_div(final_recommend_set + [item[0]])
                        inc = self.cal_inc(final_recommend_set + [item[0]])
                        div_inc_value.append((item, div * inc))
                        # div_inc_value.append((item, div * inc * item[1]))

                    best_item = sorted(div_inc_value, key=operator.itemgetter(1), reverse=True)[0][0]
                    final_recommend_set.append(best_item[0])
                    candidate_items_L.remove(best_item)
            final_recommend_full_set[user] = final_recommend_set

        # evaluation
        self.eval_all(logger, k_list, users, final_recommend_full_set, trues)

        return True

    def eval_all(self, logger, k_list, users, final_recommend_full_set, trues):
        precision = eval_precision(k_list, users, final_recommend_full_set, trues)
        recall = eval_recall(k_list, users, final_recommend_full_set, trues)
        MAP = eval_MAP(k_list, users, final_recommend_full_set, trues)
        MRR = eval_MRR(k_list, users, final_recommend_full_set, trues)
        NDCG = eval_NDCG(k_list, users, final_recommend_full_set, trues)
        mSIDR = self.eval_mSIDR(k_list, users, final_recommend_full_set, trues)

        evaluations = {
            'precision': precision,
            'recall': recall,
            'MAP': MAP,
            'MRR': MRR,
            'NDCG': NDCG,
            'mSIDR': mSIDR
        }
        self.evaluation_logger(logger, evaluations, k_list)


    def get_feat(self, data):
        # interations ---> dataframe
        columns = data.columns
        return pd.DataFrame({
            col: data[col].tolist() for col in columns
        })

    def get_topk_items_based_on_department(self, user, k):
        # get top k items of every department
        if user in self.scores_dict.keys():
            scores = self.scores_dict[user].tolist()
            items_recommend = []
            for depart in sorted(self.depart2item.keys()):
                # topk items in this department
                items_in_depart = self.depart2item[depart]
                item_score_in_depart = sorted([(elem, scores[elem])
                                               for elem in items_in_depart], key=operator.itemgetter(1), reverse=True)
                items_topk_in_depart = item_score_in_depart[:k]
                items_recommend.append(items_topk_in_depart)

            return items_recommend
        else:
            raise ValueError(f"user {user} has no score set!")

    def get_depart2item(self, item_feat):
        # get dict of items of one department
        item_id_field = self.config.final_config_dict['ITEM_ID_FIELD']
        depart_field = self.config.final_config_dict['DEPART_FIELD']
        depart2item = collections.defaultdict(list)

        for i in range(len(item_feat)):
            if i == 0:
                continue
            depart2item[item_feat.loc[i, depart_field]].append(item_feat.loc[i, item_id_field])
        return depart2item

    def get_item2depart(self, item_feat):
        # get dict of depart of an item
        item_id_field = self.config.final_config_dict['ITEM_ID_FIELD']
        depart_field = self.config.final_config_dict['DEPART_FIELD']

        item2depart = {item_feat.loc[i, item_id_field]: item_feat.loc[i, depart_field] for i in range(len(item_feat))}
        return item2depart

    def get_DP_scores(self, model):
        # get full scores of users and items
        full_scores = torch.FloatTensor([]).to(self.config["device"])
        full_predict_batch_size = 100
        hari = 0
        # embeddings_full
        full_users = torch.IntTensor(list(map(int, self.user_feat[self.config.final_config_dict['USER_ID_FIELD']].values[1:])))
        while hari < self.n_user:
            users_part = full_users[hari:hari + full_predict_batch_size]
            hari += full_predict_batch_size
            sub_inter = {self.config.final_config_dict['USER_ID_FIELD']: users_part.to(self.config["device"])}
            with torch.no_grad():
                full_scores = torch.cat((full_scores, self.sigmoid(model.full_sort_predict(sub_inter)
                                                                   .to(self.config["device"]).
                                         reshape(len(users_part), self.n_item))), dim=0)
        return dict(zip(full_users.tolist(), full_scores))

    def cal_div(self, doctors_list):
        # diversity of doctor discipline
        item_id_field = self.config.final_config_dict['ITEM_ID_FIELD']
        depart_field = self.config.final_config_dict['DEPART_FIELD']
        departs = [self.item_feat[self.item_feat[item_id_field] == doc][depart_field].values[-1] for doc in doctors_list]
        return np.round(len(set(departs)) / len(doctors_list), 2)

    def cal_inc(self, doctors_list):
        # interpersonal closeness of doctors given
        distances_list = [self.node_distance_in_kg(elem) for elem in list(combinations(doctors_list, 2))]
        distances_list_rational = [elem for elem in distances_list if elem != 0]
        inter_closeness = np.round(np.sum([1 / elem if elem != -1 else 0 for elem in distances_list_rational])
                                   / (len(doctors_list) * (len(doctors_list) - 1)), 2)
        return inter_closeness

    def node_distance_in_kg(self, nodes_set):
        # cal distance of (node1, node2)
        node1, node2 = nodes_set
        nodes_existed = self.kg_graph.nodes
        if node1 not in nodes_existed or node2 not in nodes_existed:
            # raise ValueError(
            #    f"{node1} or {node2} not in the kg graph."
            # )
            return -1
        else:
            try:
                return nx.shortest_path_length(self.kg_graph, source=node1, target=node2)
            except:
                return -1

    def construct_net(self, kg_feat):
        head_id_field = self.config.final_config_dict['HEAD_ENTITY_ID_FIELD']
        tail_id_field = self.config.final_config_dict['TAIL_ENTITY_ID_FIELD']
        relation_id_field = self.config.final_config_dict['RELATION_ID_FIELD']
        kg_feat = pd.DataFrame({
            head_id_field: kg_feat[head_id_field].tolist(),
            relation_id_field: kg_feat[relation_id_field].tolist(),
            tail_id_field: kg_feat[tail_id_field].tolist()
        })
        edges_list = [(kg_feat.loc[i, head_id_field], kg_feat.loc[i, tail_id_field])
                      for i in range(len(kg_feat))]
        graph = nx.Graph()
        graph.add_edges_from(edges_list)
        return graph

    def eval_mSIDR(self, k_list, users, final_recommendation, trues):
        # SIDR evaluation
        mSIDR = []

        # for every k
        for k in k_list:
            # for every user
            all_user_SIDR = []
            for key in final_recommendation.keys():
                all_score_user = self.scores_dict[key].tolist()

                # ldaclass related
                ldaclass = self.user2ldaclass[key]
                departments_of_ldaclass = self.ldaclass2department[ldaclass]
                department_m = list(set([self.item2depart[item] for item in final_recommendation[key][:k]]))
                sub_SIDR = []
                for depart in department_m:
                    sub_items = [item for item in final_recommendation[key][:k] if item in self.depart2item[depart]]
                    scores_box = [all_score_user[item] for item in sub_items]
                    score_q_d = max(scores_box) if len(scores_box) > 0 else 0
                    if depart in departments_of_ldaclass:
                        score_q_m = np.round(1 / len(departments_of_ldaclass), 3)
                    else:
                        score_q_m = 0
                    sub_SIDR.append(score_q_d * score_q_m)
                all_user_SIDR.append(sum(sub_SIDR))

            # mean
            mSIDR.append(np.round(np.mean(all_user_SIDR), 3))

        return mSIDR

    def eval_Jaccard(self, result_dict):
        # jaccard evaluation
        return 0

    def evaluation_logger(self, logger, evaluation, k_list):
        # evaluation logger output
        for i in range(len(k_list)):
            metric_str = f""
            for key in evaluation.keys():
                metric_str += key + " : " + str(evaluation[key][i]) + " "

            logger.info(set_color(f"top{k_list[i]} valid result: ", "blue") + metric_str)
