# -*- coding: utf-8 -*-

r"""
KGCN
################################################

Reference:
    Hongwei Wang et al. "Knowledge graph convolution networks for recommender systems." in WWW 2019.

Reference code:
    https://github.com/hwwang55/KGCN
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from CoRec.model.abstract_recommender import KnowledgeRecommender
from CoRec.model.init import xavier_normal_initialization
from CoRec.model.loss import EmbLoss
from CoRec.utils import InputType
from CoRec.model.mutt.MLP import MLP


class KGCN(KnowledgeRecommender):
    r"""KGCN is a knowledge-based recommendation model that captures inter-item relatedness effectively by mining their
    associated attributes on the KG. To automatically discover both high-order structure information and semantic
    information of the KG, we treat KG as an undirected graph and sample from the neighbors for each entity in the KG
    as their receptive field, then combine neighborhood information with bias when calculating the representation of a
    given entity.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(KGCN, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]

        # number of iterations when computing entity representation
        self.n_iter = config["n_iter"]
        self.aggregator_class = config["aggregator"]  # which aggregator to use
        self.reg_weight = config["reg_weight"]  # weight of l2 regularization
        self.neighbor_sample_size = config["neighbor_sample_size"]
        self.config = config
        self.dataset_inter = pd.DataFrame({
            self.USER_ID: dataset.inter_feat[self.USER_ID].tolist(),
            self.ITEM_ID: dataset.inter_feat[self.ITEM_ID].tolist(),
            self.config['RATING_FIELD']: dataset.inter_feat[self.config['RATING_FIELD']].tolist()
        })

        # define embedding
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(
            self.n_relations + 1, self.embedding_size
        )

        # mix txt embedding of users, txt embedding of items
        self.txt_sum_item_weight = 0.5
        self.conv_dimension_reduction = torch.nn.Conv1d(in_channels=config['TXT_EMBEDDING_DIM'], out_channels=64, kernel_size=1)
        self.mutt = MLP(config['TXT_EMBEDDING_DIM'], 64, 0.2, 5, 'FULL')
        self.txt_embeddings = dataset.txt_embeddings
        self.user_txt_embedding = self.txt_embedding_process('user', 'sum')
        self.item_txt_embedding = self.txt_embedding_process('item', 'sum')

        # sample neighbors
        kg_graph = dataset.kg_graph(form="coo", value_field="relation_id")
        adj_entity, adj_relation = self.construct_adj(kg_graph)
        self.adj_entity, self.adj_relation = adj_entity.to(
            self.device
        ), adj_relation.to(self.device)

        # define function
        self.softmax = nn.Softmax(dim=-1)
        self.linear_layers = torch.nn.ModuleList()
        for i in range(self.n_iter):
            self.linear_layers.append(
                nn.Linear(
                    self.embedding_size
                    if not self.aggregator_class == "concat"
                    else self.embedding_size * 2,
                    self.embedding_size,
                )
            )
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l2_loss = EmbLoss()

        # MLP
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(num_features=self.embedding_size * 2)
        self.fc1 = nn.Linear(self.embedding_size * 4, self.embedding_size)

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ["adj_entity", "adj_relation"]

    def construct_adj(self, kg_graph):
        r"""Get neighbors and corresponding relations for each entity in the KG.

        Args:
            kg_graph(scipy.sparse.coo_matrix): an undirected graph

        Returns:
            tuple:
                - adj_entity(torch.LongTensor): each line stores the sampled neighbor entities for a given entity,
                  shape: [n_entities, neighbor_sample_size]
                - adj_relation(torch.LongTensor): each line stores the corresponding sampled neighbor relations,
                  shape: [n_entities, neighbor_sample_size]
        """
        # self.logger.info('constructing knowledge graph ...')
        # treat the KG as an undirected graph
        kg_dict = dict()
        for triple in zip(kg_graph.row, kg_graph.data, kg_graph.col):
            head = triple[0]
            relation = triple[1]
            tail = triple[2]
            if head not in kg_dict:
                kg_dict[head] = []
            kg_dict[head].append((tail, relation))
            if tail not in kg_dict:
                kg_dict[tail] = []
            kg_dict[tail].append((head, relation))

        # self.logger.info('constructing adjacency matrix ...')
        # each line of adj_entity stores the sampled neighbor entities for a given entity
        # each line of adj_relation stores the corresponding sampled neighbor relations
        entity_num = kg_graph.shape[0]
        adj_entity = np.zeros([entity_num, self.neighbor_sample_size], dtype=np.int64)
        adj_relation = np.zeros([entity_num, self.neighbor_sample_size], dtype=np.int64)
        for entity in range(entity_num):
            if entity not in kg_dict.keys():
                adj_entity[entity] = np.array([entity] * self.neighbor_sample_size)
                adj_relation[entity] = np.array([0] * self.neighbor_sample_size)
                continue

            neighbors = kg_dict[entity]
            n_neighbors = len(neighbors)
            if n_neighbors >= self.neighbor_sample_size:
                sampled_indices = np.random.choice(
                    list(range(n_neighbors)),
                    size=self.neighbor_sample_size,
                    replace=False,
                )
            else:
                sampled_indices = np.random.choice(
                    list(range(n_neighbors)),
                    size=self.neighbor_sample_size,
                    replace=True,
                )
            adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
            adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

        return torch.from_numpy(adj_entity), torch.from_numpy(adj_relation)

    def get_neighbors(self, items):
        r"""Get neighbors and corresponding relations for each entity in items from adj_entity and adj_relation.

        Args:
            items(torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            tuple:
                - entities(list): Entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items.
                  dimensions of entities: {[batch_size, 1],
                  [batch_size, n_neighbor],
                  [batch_size, n_neighbor^2],
                  ...,
                  [batch_size, n_neighbor^n_iter]}
                - relations(list): Relations is a list of i-iter (i = 0, 1, ..., n_iter) corresponding relations for
                  entities. Relations have the same shape as entities.
        """
        items = torch.unsqueeze(items, dim=1)
        entities = [items]
        relations = []
        for i in range(self.n_iter):
            index = torch.flatten(entities[i])
            neighbor_entities = torch.index_select(self.adj_entity, 0, index).reshape(
                self.batch_size, -1
            )
            neighbor_relations = torch.index_select(
                self.adj_relation, 0, index
            ).reshape(self.batch_size, -1)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def mix_neighbor_vectors(
        self, neighbor_vectors, neighbor_relations, user_embeddings
    ):
        r"""Mix neighbor vectors on user-specific graph.

        Args:
            neighbor_vectors(torch.FloatTensor): The embeddings of neighbor entities(items),
                                                 shape: [batch_size, -1, neighbor_sample_size, embedding_size]
            neighbor_relations(torch.FloatTensor): The embeddings of neighbor relations,
                                                   shape: [batch_size, -1, neighbor_sample_size, embedding_size]
            user_embeddings(torch.FloatTensor): The embeddings of users, shape: [batch_size, embedding_size]

        Returns:
            neighbors_aggregated(torch.FloatTensor): The neighbors aggregated embeddings,
            shape: [batch_size, -1, embedding_size]

        """
        avg = False
        if not avg:
            user_embeddings = user_embeddings.reshape(
                self.batch_size, 1, 1, self.embedding_size
            )  # [batch_size, 1, 1, dim]
            user_relation_scores = torch.mean(
                user_embeddings * neighbor_relations, dim=-1
            )  # [batch_size, -1, n_neighbor]
            user_relation_scores_normalized = self.softmax(
                user_relation_scores
            )  # [batch_size, -1, n_neighbor]

            user_relation_scores_normalized = torch.unsqueeze(
                user_relation_scores_normalized, dim=-1
            )  # [batch_size, -1, n_neighbor, 1]
            neighbors_aggregated = torch.mean(
                user_relation_scores_normalized * neighbor_vectors, dim=2
            )  # [batch_size, -1, dim]
        else:
            neighbors_aggregated = torch.mean(
                neighbor_vectors, dim=2
            )  # [batch_size, -1, dim]
        return neighbors_aggregated

    def aggregate(self, user_embeddings, entities, relations):
        r"""For each item, aggregate the entity representation and its neighborhood representation into a single vector.

        Args:
            user_embeddings(torch.FloatTensor): The embeddings of users, shape: [batch_size, embedding_size]
            entities(list): entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items.
                            dimensions of entities: {[batch_size, 1],
                            [batch_size, n_neighbor],
                            [batch_size, n_neighbor^2],
                            ...,
                            [batch_size, n_neighbor^n_iter]}
            relations(list): relations is a list of i-iter (i = 0, 1, ..., n_iter) corresponding relations for entities.
                             relations have the same shape as entities.

        Returns:
            item_embeddings(torch.FloatTensor): The embeddings of items, shape: [batch_size, embedding_size]

        """
        entity_vectors = [self.entity_embedding(i) for i in entities]
        relation_vectors = [self.relation_embedding(i) for i in relations]

        for i in range(self.n_iter):
            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = (
                    self.batch_size,
                    -1,
                    self.neighbor_sample_size,
                    self.embedding_size,
                )
                self_vectors = entity_vectors[hop]
                neighbor_vectors = entity_vectors[hop + 1].reshape(shape)
                neighbor_relations = relation_vectors[hop].reshape(shape)

                neighbors_agg = self.mix_neighbor_vectors(
                    neighbor_vectors, neighbor_relations, user_embeddings
                )  # [batch_size, -1, dim]

                if self.aggregator_class == "sum":
                    output = (self_vectors + neighbors_agg).reshape(
                        -1, self.embedding_size
                    )  # [-1, dim]
                elif self.aggregator_class == "neighbor":
                    output = neighbors_agg.reshape(-1, self.embedding_size)  # [-1, dim]
                elif self.aggregator_class == "concat":
                    # [batch_size, -1, dim * 2]
                    output = torch.cat([self_vectors, neighbors_agg], dim=-1)
                    output = output.reshape(
                        -1, self.embedding_size * 2
                    )  # [-1, dim * 2]
                else:
                    raise Exception("Unknown aggregator: " + self.aggregator_class)

                output = self.linear_layers[i](output)
                # [batch_size, -1, dim]
                output = output.reshape(self.batch_size, -1, self.embedding_size)

                if i == self.n_iter - 1:
                    vector = self.Tanh(output)
                else:
                    vector = self.ReLU(output)

                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        item_embeddings = entity_vectors[0].reshape(
            self.batch_size, self.embedding_size
        )

        return item_embeddings

    def forward(self, user, item):
        self.batch_size = item.shape[0]
        # [batch_size, dim]
        user_e = self.user_embedding(user)
        # entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items. dimensions of entities:
        # {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]}
        entities, relations = self.get_neighbors(item)
        # [batch_size, dim]
        item_e = self.aggregate(user_e, entities, relations)

        if int(self.config['TXT_EMBEDDING_FLAG']) == 1:
            user_e = self.add_txt_embeddings('user', user, user_e)
            item_e = self.add_txt_embeddings('item', item, item_e)

        return user_e, item_e

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_e, pos_item_e = self.forward(user, pos_item)
        user_e, neg_item_e = self.forward(user, neg_item)

        pos_item_score = torch.mul(user_e, pos_item_e).sum(dim=1)
        neg_item_score = torch.mul(user_e, neg_item_e).sum(dim=1)

        predict = torch.cat((pos_item_score, neg_item_score))
        target = torch.zeros(len(user) * 2, dtype=torch.float32).to(self.device)
        target[: len(user)] = 1
        rec_loss = self.bce_loss(predict, target)

        l2_loss = self.l2_loss(user_e, pos_item_e, neg_item_e)
        loss = rec_loss + self.reg_weight * l2_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e = self.forward(user, item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user_index = interaction[self.USER_ID]
        item_index = torch.tensor(range(self.n_items)).to(self.device)

        user = torch.unsqueeze(user_index, dim=1).repeat(1, item_index.shape[0])
        user = torch.flatten(user)
        item = torch.unsqueeze(item_index, dim=0).repeat(user_index.shape[0], 1)
        item = torch.flatten(item)

        user_e, item_e = self.forward(user, item)
        score = torch.mul(user_e, item_e).sum(dim=1)

        return score.view(-1)

    def txt_embedding_process(self, field, pattern='sum'):
        if int(self.config['TXT_EMBEDDING_FLAG']) == 0:
            return -1

        profile_embeddings = self.txt_embeddings['profile']
        q_embeddings = self.txt_embeddings['q']
        dialog_embeddings = self.txt_embeddings['dialogue']
        inter = self.dataset_inter

        results_embeddings = 0
        if field == 'user':
            results_embeddings = q_embeddings
        else:
            if pattern == 'sum':
                items_txt_embeddings = dict()
                for item in profile_embeddings.keys():
                    users_inter = list(inter[inter['item_id'] == item]['user_id'])
                    if len(users_inter) > 0:
                        users_inter_txt_emb = np.array([dialog_embeddings[elem] for elem in users_inter
                                                        if (elem in dialog_embeddings.keys())
                                                        and (type(dialog_embeddings[elem]) == list)])
                        if len(users_inter_txt_emb) > 0:
                            items_txt_embeddings[item] = self.txt_sum_item_weight * np.array(profile_embeddings[item]) \
                                + (1 - self.txt_sum_item_weight) * np.mean(users_inter_txt_emb, axis=0)
                        else:
                            items_txt_embeddings[item] = profile_embeddings[item]
                    else:
                        items_txt_embeddings[item] = profile_embeddings[item]
                results_embeddings = items_txt_embeddings

        # dimension reduction
        input = torch.FloatTensor(np.array([[results_embeddings[hari]] for hari in results_embeddings.keys()]))
        # print(input.size())
        input = input.permute(0, 2, 1)
        out = self.conv_dimension_reduction(input)
        txt_emb_new = [torch.reshape(hari, (len(hari),)).tolist() for hari in out]
        txt_emb_new = dict(zip(results_embeddings.keys(), txt_emb_new))

        return txt_emb_new

    def add_txt_embeddings(self, field, iters, embedding):
        if field == 'user':
            txt_embeddings = self.user_txt_embedding
        else:
            txt_embeddings = self.item_txt_embedding

        iters_list = iters.tolist()
        right_part = torch.FloatTensor(np.array(
            [txt_embeddings[iters_list[i]] if iters_list[i] in txt_embeddings.keys() else embedding[i, :].tolist() for i
             in range(len(iters))]))

        # output = self.relu(self.bn1(self.fc1(torch.cat((embedding, torch.FloatTensor(right_part).cuda()), dim=1))))
        # return output
        return torch.cat((embedding, torch.FloatTensor(right_part).cuda()), dim=1)
