# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

from CoRec.model.abstract_recommender import KnowledgeRecommender
from CoRec.model.init import xavier_uniform_initialization
from CoRec.model.layers import SparseDropout
from CoRec.model.loss import BPRLoss, EmbLoss
from CoRec.utils import InputType
from CoRec.model.mutt.MLP import MLP


class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """

    def __init__(
        self,
    ):
        super(Aggregator, self).__init__()

    def forward(
        self,
        entity_emb,
        user_emb,
        latent_emb,
        relation_emb,
        edge_index,
        edge_type,
        interact_mat,
        disen_weight_att,
    ):
        from torch_scatter import scatter_mean

        n_entities = entity_emb.shape[0]

        """KG aggregate"""
        head, tail = edge_index
        edge_relation_emb = relation_emb[edge_type]
        neigh_relation_emb = (
            entity_emb[tail] * edge_relation_emb
        )  # [-1, embedding_size]
        entity_agg = scatter_mean(
            src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0
        )

        """cul user->latent factor attention"""
        score_ = torch.mm(user_emb, latent_emb.t())
        score = nn.Softmax(dim=1)(score_)  # [n_users, n_factors]
        """user aggregate"""
        user_agg = torch.sparse.mm(
            interact_mat, entity_emb
        )  # [n_users, embedding_size]
        disen_weight = torch.mm(
            nn.Softmax(dim=-1)(disen_weight_att), relation_emb
        )  # [n_factors, embedding_size]
        user_agg = (
            torch.mm(score, disen_weight)
        ) * user_agg + user_agg  # [n_users, embedding_size]

        return entity_agg, user_agg


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(
        self,
        embedding_size,
        n_hops,
        n_users,
        n_factors,
        n_relations,
        edge_index,
        edge_type,
        interact_mat,
        ind,
        tmp,
        device,
        node_dropout_rate=0.5,
        mess_dropout_rate=0.1,
    ):
        super(GraphConv, self).__init__()

        self.embedding_size = embedding_size
        self.n_hops = n_hops
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_factors = n_factors
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.interact_mat = interact_mat
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind
        self.temperature = tmp
        self.device = device

        # define layers
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_size)
        disen_weight_att = nn.init.xavier_uniform_(torch.empty(n_factors, n_relations))
        self.disen_weight_att = nn.Parameter(disen_weight_att)
        self.convs = nn.ModuleList()
        for i in range(self.n_hops):
            self.convs.append(Aggregator())
        self.node_dropout = SparseDropout(p=self.mess_dropout_rate)  # node dropout
        self.mess_dropout = nn.Dropout(p=self.mess_dropout_rate)  # mess dropout

        # parameters initialization
        self.apply(xavier_uniform_initialization)

    def edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(
            n_edges, size=int(n_edges * rate), replace=False
        )
        return edge_index[:, random_indices], edge_type[random_indices]

    def forward(self, user_emb, entity_emb, latent_emb):
        """node dropout"""
        # node dropout
        if self.node_dropout_rate > 0.0:
            edge_index, edge_type = self.edge_sampling(
                self.edge_index, self.edge_type, self.node_dropout_rate
            )
            interact_mat = self.node_dropout(self.interact_mat)
        else:
            edge_index, edge_type = self.edge_index, self.edge_type
            interact_mat = self.interact_mat

        entity_res_emb = entity_emb  # [n_entities, embedding_size]
        user_res_emb = user_emb  # [n_users, embedding_size]
        relation_emb = self.relation_embedding.weight  # [n_relations, embedding_size]
        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](
                entity_emb,
                user_emb,
                latent_emb,
                relation_emb,
                edge_index,
                edge_type,
                interact_mat,
                self.disen_weight_att,
            )
            """message dropout"""
            if self.mess_dropout_rate > 0.0:
                entity_emb = self.mess_dropout(entity_emb)
                user_emb = self.mess_dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)
            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return (
            entity_res_emb,
            user_res_emb,
            self.calculate_cor_loss(self.disen_weight_att),
        )

    def calculate_cor_loss(self, tensors):
        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = F.normalize(tensor_1, dim=0)
            normalized_tensor_2 = F.normalize(tensor_2, dim=0)
            return (normalized_tensor_1 * normalized_tensor_2).sum(
                dim=0
            ) ** 2  # no negative

        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = (
                torch.matmul(tensor_1, tensor_1.t()) * 2,
                torch.matmul(tensor_2, tensor_2.t()) * 2,
            )  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1**2, tensor_2**2
            a, b = torch.sqrt(
                torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8
            ), torch.sqrt(
                torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8
            )  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel**2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel**2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel**2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)

        def MutualInformation(tensors):
            # tensors: [n_factors, dimension]
            # normalized_tensors: [n_factors, dimension]
            normalized_tensors = F.normalize(tensors, dim=1)
            scores = torch.mm(normalized_tensors, normalized_tensors.t())
            scores = torch.exp(scores / self.temperature)
            cor_loss = -torch.sum(torch.log(scores.diag() / scores.sum(1)))
            return cor_loss

        """cul similarity for each latent factor weight pairs"""
        if self.ind == "mi":
            return MutualInformation(tensors)
        elif self.ind == "distance":
            cor_loss = 0.0
            for i in range(self.n_factors):
                for j in range(i + 1, self.n_factors):
                    cor_loss += DistanceCorrelation(tensors[i], tensors[j])
        elif self.ind == "cosine":
            cor_loss = 0.0
            for i in range(self.n_factors):
                for j in range(i + 1, self.n_factors):
                    cor_loss += CosineSimilarity(tensors[i], tensors[j])
        else:
            raise NotImplementedError(
                f"The independence loss type [{self.ind}] has not been supported."
            )
        return cor_loss


class CoRec(KnowledgeRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(CoRec, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.n_factors = config["n_factors"]
        self.context_hops = config["context_hops"]
        self.node_dropout_rate = config["node_dropout_rate"]
        self.mess_dropout_rate = config["mess_dropout_rate"]
        self.ind = config["ind"]
        self.sim_decay = config["sim_regularity"]
        self.reg_weight = config["reg_weight"]
        self.temperature = config["temperature"]
        self.config = config
        self.dataset_inter = pd.DataFrame({
            self.USER_ID: dataset.inter_feat[self.USER_ID].tolist(),
            self.ITEM_ID: dataset.inter_feat[self.ITEM_ID].tolist(),
            self.config['RATING_FIELD']: dataset.inter_feat[self.config['RATING_FIELD']].tolist()
        })

        # load dataset info
        self.inter_matrix = dataset.inter_matrix(form="coo").astype(
            np.float32
        )  # [n_users, n_items]
        # inter_matrix: [n_users, n_entities]; inter_graph: [n_users + n_entities, n_users + n_entities]
        self.interact_mat, _ = self.get_norm_inter_matrix(mode="si")
        self.kg_graph = dataset.kg_graph(
            form="coo", value_field="relation_id"
        )  # [n_entities, n_entities]
        # edge_index: [2, -1]; edge_type: [-1,]
        self.edge_index, self.edge_type = self.get_edges(self.kg_graph)

        # define layers and loss
        self.n_nodes = self.n_users + self.n_entities
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.latent_embedding = nn.Embedding(self.n_factors, self.embedding_size)
        self.gcn = GraphConv(
            embedding_size=self.embedding_size,
            n_hops=self.context_hops,
            n_users=self.n_users,
            n_relations=self.n_relations,
            n_factors=self.n_factors,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            interact_mat=self.interact_mat,
            ind=self.ind,
            tmp=self.temperature,
            device=self.device,
            node_dropout_rate=self.node_dropout_rate,
            mess_dropout_rate=self.mess_dropout_rate,
        )
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.restore_user_e = None
        self.restore_entity_e = None

        # mix txt embedding of users, txt embedding of items
        self.txt_sum_item_weight = 0.5
        self.conv_dimension_reduction = torch.nn.Conv1d(in_channels=config['TXT_EMBEDDING_DIM'], out_channels=64,
                                                        kernel_size=1)
        self.mutt = MLP(config['TXT_EMBEDDING_DIM'], 64, 0.2, 5, 'FULL')
        self.txt_embeddings = dataset.txt_embeddings
        self.user_txt_embedding = self.txt_embedding_process('user', 'sum')
        self.item_txt_embedding = self.txt_embedding_process('item', 'sum')

        # parameters initialization
        self.apply(xavier_uniform_initialization)

    def get_norm_inter_matrix(self, mode="bi"):
        # Get the normalized interaction matrix of users and items.

        def _bi_norm_lap(A):
            # D^{-1/2}AD^{-1/2}
            rowsum = np.array(A.sum(1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            bi_lap = d_mat_inv_sqrt.dot(A).dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()

        def _si_norm_lap(A):
            # D^{-1}A
            rowsum = np.array(A.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(A)
            return norm_adj.tocoo()

        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_entities, self.n_users + self.n_entities),
            dtype=np.float32,
        )
        inter_M = self.inter_matrix
        inter_M_t = self.inter_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        if mode == "bi":
            L = _bi_norm_lap(A)
        elif mode == "si":
            L = _si_norm_lap(A)
        else:
            raise NotImplementedError(
                f"Normalize mode [{mode}] has not been implemented."
            )
        # covert norm_inter_graph to tensor
        i = torch.LongTensor(np.array([L.row, L.col]))
        data = torch.FloatTensor(L.data)
        norm_graph = torch.sparse.FloatTensor(i, data, L.shape)

        # interaction: user->item, [n_users, n_entities]
        L_ = L.tocsr()[: self.n_users, self.n_users :].tocoo()
        # covert norm_inter_matrix to tensor
        i_ = torch.LongTensor(np.array([L_.row, L_.col]))
        data_ = torch.FloatTensor(L_.data)
        norm_matrix = torch.sparse.FloatTensor(i_, data_, L_.shape)

        return norm_matrix.to(self.device), norm_graph.to(self.device)

    def get_edges(self, graph):
        index = torch.LongTensor(np.array([graph.row, graph.col]))
        type = torch.LongTensor(np.array(graph.data))
        return index.to(self.device), type.to(self.device)

    def forward(self):
        user_embeddings = self.user_embedding.weight
        entity_embeddings = self.entity_embedding.weight
        latent_embeddings = self.latent_embedding.weight
        # entity_gcn_emb: [n_entities, embedding_size]
        # user_gcn_emb: [n_users, embedding_size]
        # latent_gcn_emb: [n_factors, embedding_size]
        entity_gcn_emb, user_gcn_emb, cor_loss = self.gcn(
            user_embeddings, entity_embeddings, latent_embeddings
        )
        if int(self.config['TXT_EMBEDDING_FLAG']) == 1:
            user_gcn_emb = self.add_txt_embeddings('user', user_gcn_emb.shape[0], user_embeddings)
            entity_gcn_emb = self.add_txt_embeddings('item', entity_gcn_emb.shape[0], entity_embeddings)

        return user_gcn_emb, entity_gcn_emb, cor_loss

    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data of KG.
        Args:
            interaction (Interaction): Interaction class of the batch.
        Returns:
            torch.Tensor: Training loss, shape: []
        """

        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, entity_all_embeddings, cor_loss = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = entity_all_embeddings[pos_item]
        neg_embeddings = entity_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)
        reg_loss = self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)
        cor_loss = self.sim_decay * cor_loss
        loss = mf_loss + self.reg_weight * reg_loss + cor_loss
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, entity_all_embeddings, _ = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = entity_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_entity_e is None:
            self.restore_user_e, self.restore_entity_e, _ = self.forward()
        u_embeddings = self.restore_user_e[user.tolist()]
        i_embeddings = self.restore_entity_e[: self.n_items]

        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))

        return scores.view(-1)

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

    def add_txt_embeddings(self, field, length, embedding):
        if field == 'user':
            txt_embeddings = self.user_txt_embedding
        else:
            txt_embeddings = self.item_txt_embedding

        iters_list = list(range(length))
        right_part = torch.FloatTensor(np.array(
            [txt_embeddings[i] if i in txt_embeddings.keys() else embedding[i, :].tolist() for i
                in iters_list]))

        return torch.cat((embedding, torch.FloatTensor(right_part).cuda()), dim=1)
