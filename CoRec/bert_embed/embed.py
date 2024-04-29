import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

dataset_name = 'haodf'

def bert_sent_embed(model, tokenizer, device, id_content, output_path):
    con_emb_dict = {}
    for idx, content in tqdm(id_content.items(), desc=output_path):
        input_ids = torch.tensor([tokenizer.encode(str(content))])
        if len(input_ids[0].numpy().tolist()) > 512:
            input_ids = torch.from_numpy(np.array(input_ids[0].numpy().tolist()[0:512])).reshape(1, -1).type(
                torch.LongTensor)
        input_ids = input_ids.to(device)
        with torch.no_grad():
            features = model(input_ids)
        con_emb_dict[idx] = features[1].cpu().numpy()[0].tolist()
    with open(output_path, 'w') as f:
        json.dump(con_emb_dict, f, ensure_ascii=False)


def chunks(list, n):
    chunks_list = []
    len_list = len(list)
    step = math.ceil(len_list / n)
    for i in range(0, n):
        chunks_list.append(list[i * step:(i + 1) * step])
    return chunks_list


if __name__ == '__main__':
    profile_data = pd.read_csv(f'../../dataset/DP_chaos/doctor_info.txt', sep='\t')
    profile_data = profile_data[profile_data['profile'].notnull()]
    patient_data = pd.read_csv(f'../../dataset/DP_chaos/sampled/text_data.csv', sep='\t')
    patient_data = patient_data[['p_id', 'd_id', 'query', 'dialogue']]

    profile_data = profile_data[
        profile_data['doctor_id'].isin(list(set(list(patient_data['d_id']))))]

    id_profile = dict(zip(list(map(str, profile_data['doctor_id'].values)), profile_data['profile'].values))
    id_q = dict(zip(list(map(str, patient_data['p_id'].values)), patient_data['query'].values))
    id_dialog = dict(zip(list(map(str, patient_data['p_id'].values)), patient_data['dialogue'].values))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('../mc_bert_base/')
    model = BertModel.from_pretrained('../mc_bert_base/')
    model = model.to(device)

    embedding_path = '../../dataset/OMC-100k-origin/bert_embeddings'
    if not os.path.exists(embedding_path):
        os.makedirs(embedding_path)

    bert_sent_embed(model, tokenizer, device, id_profile, f'{embedding_path}/profile_embeddings.json')
    bert_sent_embed(model, tokenizer, device, id_q, f'{embedding_path}/q_embeddings.json')
    bert_sent_embed(model, tokenizer, device, id_dialog, f'{embedding_path}/dialogue_embeddings.json')