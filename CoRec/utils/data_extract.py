import collections
import re
import pandas as pd
import random
import networkx as nx
import os
import torch
import numpy as np
import time
import jieba
import json
import operator
from datetime import datetime
from tqdm import tqdm
#pd.set_option('display.max_colwidth', 200)

def data_extract_kg():
    # extract the entity of kg from raw data
    root_path = "E:/program/PycharmProjects/doctor_recomend/data/haodf/"
    save_path = '../../dataset/DP_chaos/'
    department = ['心血管内科', '神经内科', '消化内科', '内分泌科', '呼吸内科', '感染内科', '眼科综合',
                  '口腔科综合', '肿瘤内科', '妇科', '男科', '中医综合', '儿科综合', '康复科', '精神科']
    attrs = ['doctor_id', 'doctor_title', 'education_title', 'hospital', 'consultation_amount',
             'patient_recommendation_score', 'further_experience', 'profile', 'social_title', 'work_experience',
             'education_experience', 'total_access', 'total_article', 'total_patient', 'total_evaluation',
             'thanks_letter_num', 'thanks_present', 'new_in_time',
             'cure_experience', 'evaluation_type_set', 'cure_satisfaction', 'attitude_satisfaction']

    r_source = []
    r_target = []
    r_relation = []

    k_source = []
    k_target = []
    k_relation = []

    doctor_info = pd.DataFrame(columns=['doctor_id', 'department', 'doctor_title', 'education_title',
                                        'consultation_amount', 'patient_recommendation_score', 'profile', 'total_access',
                                        'total_article', 'total_patient', 'total_evaluation', 'thanks_letter_num',
                                        'thanks_present', 'active_years', 'cure_satisfaction', 'attitude_satisfaction'])

    for depart in department:
        doctor = pd.read_csv(root_path + depart + '/doctor.csv')
        for i in range(len(doctor)):
            did = doctor.loc[i, 'doctor_id']
            patients_of_doctor_path = root_path + depart + '/' + str(did)
            if (not os.path.exists(patients_of_doctor_path)) or (len(os.listdir(patients_of_doctor_path)) == 0):
                continue

            doctor_info_single = []
            doctor_info_single.append(did)
            doctor_info_single.append(depart)
            for attr in attrs[1:]:
                if attr in ['doctor_title', 'education_title', 'consultation_amount', 'patient_recommendation_score',
                            'profile', 'total_access', 'total_article', 'total_patient', 'total_evaluation',
                            'thanks_letter_num', 'thanks_present', 'cure_satisfaction', 'attitude_satisfaction']:
                    if pd.isnull(doctor.loc[i, attr]):
                        doctor_info_single.append("")
                    else:
                        doctor_info_single.append(doctor.loc[i, attr])
                elif attr == 'new_in_time':
                    start_year = int(doctor.loc[i, attr][:4])
                    doctor_info_single.append(2022 - start_year)
                elif attr == 'hospital':
                    for hos in eval(doctor.loc[i, attr]):
                        if hos != None:
                            r_source.append(did)
                            posi = hos.find('医院')
                            r_target.append(hos[:posi + 2])
                            r_relation.append('doctor.work_in.hospital')
                elif attr in ['further_experience', 'work_experience', 'education_experience']:
                    if not pd.isnull(doctor.loc[i, attr]):
                        for exper in eval(doctor.loc[i, attr]):
                            r_source.append(did)
                            institute = exper.split(';')[1]
                            posi = max(institute.find('医院'), institute.find('大学'))
                            r_target.append(institute[:posi + 2])
                            if attr in ['further_experience', 'work_experience']:
                                r_relation.append('doctor.work_in.hospital')
                            else:
                                r_relation.append('doctor.study_in.university')
                elif attr in ['cure_experience', 'evaluation_type_set']:
                    for label in eval(doctor.loc[i, attr])[0]:
                        k_source.append(did)
                        k_target.append(label)
                        k_relation.append('doctor.cure_disease.disease')
                elif attr == 'social_title':
                    special = ['中国医师协会', '中华医学会', '中国医生协会', '中国医学会']
                    bad_words = ['担任', '现任', '兼任']
                    if pd.isnull(doctor.loc[i, attr]):
                        continue
                    for kumi in eval(doctor.loc[i, attr]):
                        for elem in re.split(r"[.,。，]", kumi):
                            position = elem.find('会')
                            special_kumi = [elem.find(term) for term in special]
                            if (np.array(special_kumi) < 0).all():
                                if position > 0:
                                    r_source.append(did)
                                    r_target.append(elem[:position + 1])
                                    r_relation.append('doctor.member_of.institute')
                            else:
                                sub_elem = elem[position + 1:]
                                sub_position = sub_elem.find('会')
                                if sub_position > 0:
                                    r_source.append(did)
                                    institute = elem[:position + sub_position + 2]
                                    for word in bad_words:
                                        institute = institute.replace(word, '')
                                    r_target.append(institute)
                                    r_relation.append('doctor.member_of.institute')
                else:
                    print('sth wrong!')

            doctor_info.loc[len(doctor_info.index)] = doctor_info_single

        doctor_rg = pd.DataFrame({
            'doctor_id': r_source,
            'relation': r_relation,
            'target': r_target
        })

        doctor_kg = pd.DataFrame({
            'doctor_id': k_source,
            'relation': k_relation,
            'target': k_target
        })

        doctor_rg = doctor_rg.drop_duplicates(doctor_rg.keys(), keep='first')
        doctor_kg = doctor_kg.drop_duplicates(doctor_kg.keys(), keep='first')

        doctor_rg.sort_values(by='doctor_id', inplace=True, ascending=True)
        doctor_rg = doctor_rg.reset_index(drop=True)

        doctor_kg.sort_values(by='doctor_id', inplace=True, ascending=True)
        doctor_kg = doctor_kg.reset_index(drop=True)

        doctor_rg.to_csv(save_path + 'doctor_rg.txt', sep='\t', index=False)
        doctor_kg.to_csv(save_path + 'doctor_kg.txt', sep='\t', index=False)

        doctor_info.to_csv(save_path + 'doctor_info.txt', sep='\t', index=False)

def text_data_process():
    # extract text data from raw data
    # include profile, query, dialogue
    root_path = "E:/program/PycharmProjects/doctor_recomend/data/haodf/"
    save_path = '../../dataset/DP_chaos/'
    department_cn = ['心血管内科', '神经内科', '消化内科', '内分泌科', '呼吸内科', '感染内科', '眼科综合',
                     '口腔科综合', '肿瘤内科', '妇科', '男科', '中医综合', '儿科综合', '康复科', '精神科']

    result = pd.DataFrame(columns=['p_id', 'd_id', 'department', 'profile', 'query', 'dialogue',
                                   'query_time', 'first_response_time'])

    for i in range(len(department_cn)):
        sub_query = []
        sub_p_id = []
        sub_d_id = []
        sub_profile = []
        sub_dialogue = []
        sub_first_response_time = []
        sub_query_time = []

        sub_path = root_path + department_cn[i]
        doctors = os.listdir(sub_path)
        doctors.remove('doctor.csv')
        for j in tqdm(range(len(doctors))):
            if len(os.listdir(sub_path + '/' + doctors[j])) == 0:
                continue
            doctor = pd.read_csv(sub_path + '/doctor.csv')
            doctor['doctor_id'] = doctor['doctor_id'].astype(str)
            profile = doctor[doctor['doctor_id'] == str(doctors[j])]['profile'].values
            if len(profile) > 0:
                profile = profile[0]
            else:
                continue

            patients = pd.read_csv(sub_path + '/' + doctors[j] + '/' + 'patient.csv')
            patients = patients[['patient_id', 'query', 'disease_label', 'disease_description']]
            for k in range(len(patients)):
                # dialogue
                dial = pd.read_csv(sub_path + '/' + doctors[j] + '/' + str(patients.loc[k, 'patient_id']) + '.txt',
                                   sep='\t')
                dial = dial[~dial['speaker'].isin(['小牛医助'])]
                words = dial['word'].values
                dialogue = ''
                for hari in range(len(words)):
                    stop_words = ['仅主诊医生和患者本人可见', '天气逐渐转凉', '您已报到成功', '您好欢迎您使用网上诊室功能']
                    bad_word = 0
                    for stop in stop_words:
                        if str(words[hari]).find(stop) >= 0:
                            bad_word = 1
                    if bad_word == 0:
                        dialogue = dialogue + re.sub(r'\(.+留言\)', "",
                                                     str(words[hari]).replace('\n', ' ').replace('\r', ' ') + ' ')

                # query time
                query_time_effect = ''
                first_response_time_effect = ''
                disease_description = patients.loc[k, 'disease_description']
                if pd.isnull(disease_description):
                    continue
                if len(disease_description) >= 14:
                    query_time_field = disease_description[-14:]
                    query_time_find = re.search(r"(\d{4}-\d{1,2}-\d{1,2})", query_time_field)
                    if query_time_find:
                        query_time = query_time_find.group(0)
                        query_time_effect = query_time
                    else:
                        continue
                else:
                    continue

                # first response time of doctors
                dial = pd.read_csv(
                    sub_path + '/' + doctors[j] + '/' + str(patients.loc[k, 'patient_id']) + '.txt',
                    sep='\t')
                dial = dial[dial['speaker'] == '医生']
                firsttime = dial['datetime'].values
                if len(firsttime) > 0:
                    firsttime = firsttime[0]
                    first_response_time_effect = firsttime
                else:
                    continue

                # print(dialogue)
                sub_p_id.append(patients.loc[k, 'patient_id'])
                sub_query.append(patients.loc[k, 'query'])
                sub_d_id.append(doctors[j])
                sub_profile.append(profile)
                sub_dialogue.append(dialogue)
                sub_query_time.append(query_time_effect)
                sub_first_response_time.append(first_response_time_effect)

        sub_data = pd.DataFrame({
            'p_id': sub_p_id,
            'd_id': sub_d_id,
            'department': department_cn[i],
            'profile': sub_profile,
            'query': sub_query,
            'dialogue': sub_dialogue,
            'query_time': sub_query_time,
            'first_response_time': sub_first_response_time
        })

        # print(sub_data)
        result = pd.concat([result, sub_data], axis=0, ignore_index=True)

    # print(result)
    result.to_csv(save_path + 'text_data.csv', sep='\t', index=False)


def item_process():
    data_path = '../../dataset/DP_chaos/'
    save_path = '../../dataset/OMC-100k-origin/'

    text_data = pd.read_csv(data_path + 'sampled/text_data.csv', sep='\t')
    doctors = list(set(text_data['d_id']))
    print(len(doctors))

    # item part
    items_info = pd.read_csv(data_path + 'doctor_info.txt', sep='\t')
    columns = \
          {'department':                   'single_str',
           'doctor_title':                 'single_str',
           'education_title':              'single_str',
           'consultation_amount':          'float',
           'patient_recommendation_score': 'float',
           'total_access':                 'float',
           'total_article':                'float',
           'total_patient':                'float',
           'total_evaluation':             'float',
           'thanks_letter_num':            'float',
           'thanks_present':               'float',
           'active_years':                 'float',
           'cure_satisfaction':            'float',
           'attitude_satisfaction':        'float'
           }

    remove_list = []
    for i in range(len(items_info)):
        if items_info.loc[i, 'doctor_id'] not in doctors:
            remove_list.append(i)
            continue
        if pd.isnull(items_info.loc[i, 'profile']):
            # remove_list.append(i)
            continue
        for col in columns.keys():
            value = items_info.loc[i, col]
            if type(value) == int or type(value) == float or col in ['department', 'doctor_title', 'education_title']:
                continue
            elif type(value) == str:
                pattern = re.compile(r'(^[-+]?([1-9][0-9]*|0)(\.[0-9]+)?$)')
                ifnumeric = pattern.match(items_info.loc[i, col])
                if not ifnumeric:
                    items_info.loc[i, col] = items_info.loc[i-1, col]

            items_info[str(col)+':'+columns[col]] = items_info[col]

    for col in columns.keys():
        if col in ['department', 'doctor_title', 'education_title']:
            items_info[str(col)+':'+columns[col]] = items_info[col]
        items_info = items_info.drop(col, axis=1)
    items_info = items_info.drop(remove_list, axis=0)
    items_info.to_csv(save_path+'OMC-100k.item', sep='\t', index=False)

def user_process():
    data_path = '../../dataset/DP_chaos/'
    save_path = '../../dataset/OMC-100k-origin/'

    # user part
    user_columns = {
        'p_id': 'user_id:token',
        'department': 'department:single_str',
        'query_time': 'query_time:float',
        'first_response_time': 'first_response_time:float'
    }

    text_data = pd.read_csv(data_path + 'sampled/text_data.csv', sep='\t')

    # time transform
    complete_pattern = r"(\d{4}.\d{1,2}.\d{1,2})"
    part_pattern = r"(\d{1,2}.\d{1,2})"
    null_pattern = '-1'
    special_time = {'今天': '2022.11.18', '昨天': '2022.11.17'}
    for i in range(len(text_data)):
        fr_time = text_data.loc[i, 'first_response_time']
        if fr_time == null_pattern:
            continue

        if re.search(complete_pattern, fr_time):
            fr_time = fr_time
        elif re.search(part_pattern, fr_time):
            fr_time = '2022.' + fr_time
        else:
            fr_time = special_time[fr_time]

        text_data.loc[i, 'first_response_time'] = int(time.mktime(datetime.strptime(fr_time, '%Y.%m.%d').timetuple()))

        text_data.loc[i, 'query_time'] = int(
            time.mktime(datetime.strptime(text_data.loc[i, 'query_time'], '%Y-%m-%d').timetuple()))

        # print(text_data.loc[i, :])

    # output
    original_cols = text_data.columns
    for col in user_columns.keys():
        text_data[user_columns[col]] = text_data[col]
    user_data = text_data.drop(user_columns.keys() | original_cols, axis=1)
    user_data.to_csv(save_path + 'OMC-100k.user', sep='\t', index=False)

def sample_data():
    data_path = '../../dataset/DP_chaos/'
    save_path = '../../dataset/OMC-100k-origin/'

    doctor_info = pd.read_csv(save_path + 'OMC-100k.item', sep='\t')
    text_data = pd.read_csv(data_path + 'text_data.csv', sep='\t')
    patient_num = 20
    dialogue_length_min = 30
    selected_index = []
    original_cols = text_data.columns
    for doct in list(set(doctor_info['item_id:token'])):
        sub_data = text_data[text_data['d_id'] == doct]
        sub_data['query_length'] = [len(sub_data.loc[index, 'query']) if type(sub_data.loc[index, 'query']) == str
                                    else 0 for index in sub_data.index]
        sub_data['dialogue_length'] = [len(sub_data.loc[index, 'dialogue'])
                                       if type(sub_data.loc[index, 'dialogue']) == str
                                       else 0 for index in sub_data.index]
        sub_data = sub_data[sub_data['dialogue_length'] > dialogue_length_min]
        sub_data = sub_data.sort_values(by=['query_length', 'dialogue_length'], ascending=False)
        sam = sub_data.index[:patient_num]
        # print(sub_data[:10], sam)
        selected_index.extend(sam)
    final_text_data = text_data.loc[selected_index, original_cols]
    final_text_data.to_csv(data_path + 'sampled/text_data.csv', sep='\t', index=False)

def hospital_address_triple():
    # hospitals triples add
    kg_data = pd.read_csv('../../dataset/OMC-100k/OMC-100k.kg', sep='\t')
    kg_data = kg_data[kg_data['relation_id:token'] == 'doctor.work_in.hospital']
    address = pd.read_excel('../../dataset/行政区划.xlsx')
    head = 'head_id:token'
    relation = 'relation_id:token'
    tail = 'tail_id:token'

    provinces = list(set(address['省份']))
    cities = list(set(address['城市']))

    source = []
    brg = []
    target = []

    no_shot = []
    for user in list(set(kg_data[head].values)):
        hos = kg_data[kg_data[head] == user][tail].values
        for elem in hos:
            words_cut = jieba.cut(elem)
            catch = 0
            for word in words_cut:
                choice = [word, word[:2], word[:3]]
                for wrd in choice:
                    if catch == 0 and wrd in cities:
                        catch = 1
                        # city triple
                        source.append(elem)
                        brg.append('hospital.addressed_in.city')
                        target.append(wrd)
                        # province triple
                        provin = address[address['城市'] == wrd]['省份'].values[0]
                        if provin != wrd:
                            source.append(wrd)
                            brg.append('city.city_of.province')
                            target.append(provin)

                    if wrd == '复旦':
                        catch = 1
                        source.append(elem)
                        brg.append('hospital.addressed_in.city')
                        target.append('上海')
            if catch == 0:
                no_shot.append(elem)

    triples_got = pd.DataFrame({
        'head_id:token': source,
        'relation_id:token': brg,
        'tail_id:token': target
    })

    hospital_unrecognized = pd.DataFrame({
        'hospital': list(set(no_shot))
    })

    # triples_got.to_csv('../../dataset/OMC-100k/hospital_related.txt', sep='\t', index=False)
    hospital_unrecognized.to_csv('../../dataset/OMC-100k/doctor_addition_info/hospital_unrecognized.txt', sep='\t', index=False)

def hospital_address_triple_addition():
    # hospitals additional triples add
    hospitals_plus = pd.read_csv('../../dataset/OMC-100k/doctor_addition_info/hospital_unrecognized.txt', sep='\t', encoding='GBK')
    hospitals_triples = pd.read_csv('../../dataset/OMC-100k/doctor_addition_info/hospital_related.txt', sep='\t')
    address = pd.read_excel('../../dataset/行政区划.xlsx')
    provinces = list(set(address['省份']))
    cities = list(set(address['城市']))

    source = []
    brg = []
    target = []

    no_shot = []
    for i in range(len(hospitals_plus)):
        source.append(hospitals_plus.loc[i, 'hospital'])
        brg.append('hospital.addressed_in.city')
        city = hospitals_plus.loc[i, 'city']
        target.append(city)

        provin = address[address['城市'] == city]['省份'].values
        if len(provin) > 0 and provin[0] != city:
            source.append(city)
            brg.append('city.city_of.province')
            target.append(provin[0])

    triples_got = pd.DataFrame({
        'head_id:token': source,
        'relation_id:token': brg,
        'tail_id:token': target
    })
    final_result = pd.concat([hospitals_triples, triples_got], axis=0, ignore_index=True)
    final_result = final_result.drop_duplicates(keep='first')

    final_result.to_csv('../../dataset/OMC-100k/doctor_addition_info/hospital_related_final.txt', sep='\t', index=False)

def idilize_and_inter_link_file():
    # transform original id to new sequenced id
    # inter file get
    # link file get
    target_path = '../../dataset/OMC-100k/'
    embeddings_path = '../../dataset/OMC-100k-origin/bert_embeddings/'
    kg = pd.read_csv('../../dataset/DP_chaos/doctor_kg.txt', sep='\t')
    rg = pd.read_csv('../../dataset/DP_chaos/doctor_rg.txt', sep='\t')
    users = pd.read_csv('../../dataset/OMC-100k-origin/OMC-100k.user', sep='\t')
    items = pd.read_csv('../../dataset/OMC-100k-origin/OMC-100k.item', sep='\t')
    text_data = pd.read_csv('../../dataset/DP_chaos/sampled/text_data.csv', sep='\t')
    users = users.sort_values(by='department:token')
    items = items.sort_values(by='department:token')

    heads = ['profile', 'q', 'dialogue']
    embeddings = dict()

    # user_id dict
    user_id_dict = dict(zip(users['user_id:token'].values, range(1, 1 + len(users), 1)))
    # item_id_dict
    item_id_dict = dict(zip(items['item_id:token'].values, range(1, 1 + len(users), 1)))

    # users transform
    for i in range(len(users)):
        users.loc[i, 'user_id:token'] = user_id_dict[users.loc[i, 'user_id:token']]
    users = users.sort_values(by='user_id:token')
    users.to_csv(target_path + 'OMC-100k.user', sep='\t', index=False)

    # items transform
    item_columns = list(items.columns)
    item_columns.remove('profile')
    for i in range(len(items)):
        items.loc[i, 'item_id:token'] = item_id_dict[items.loc[i, 'item_id:token']]
    items = items.sort_values(by='item_id:token')
    items = items[item_columns]
    items.to_csv(target_path + 'OMC-100k.item', sep='\t', index=False)

    # embeddings
    for head in heads:
        t_path = embeddings_path + head + '_embeddings.json'
        with open(t_path, 'r', encoding='utf-8') as f:
            embeddings[head] = json.load(f)
        if head == 'profile':
            keys_transformed = [item_id_dict[int(elem)] for elem in list(embeddings[head].keys())]
        else:
            keys_transformed = [user_id_dict[int(elem)] for elem in list(embeddings[head].keys())]
        o_path = target_path + 'bert_embeddings/' + head + '_embeddings.json'
        con_emb_dict = dict(zip(keys_transformed, list(embeddings[head].values())))
        with open(o_path, 'w') as f:
            json.dump(con_emb_dict, f, ensure_ascii=False)

    # kg and link file
    kg_columns = {
        'head_id:token': 'doctor_id',
        'relation_id:token': 'relation',
        'tail_id:token': 'target'
    }
    full_kg = pd.concat([kg, rg], axis=0, ignore_index=True)
    full_kg = full_kg.sort_values(by='doctor_id')
    for col in kg_columns.keys():
        full_kg[col] = full_kg[kg_columns[col]]
    full_kg = full_kg.drop(list(kg_columns.values()), axis=1)
    full_kg.to_csv(target_path + 'OMC-100k.kg', sep='\t', index=False)

    link = pd.DataFrame({
        'item_id:token': list(item_id_dict.values()),
        'entity_id:token': list(item_id_dict.keys())
    })
    link.to_csv(target_path + 'OMC-100k.link', sep='\t', index=False)

    # inter file
    inter = pd.DataFrame({
        'user_id:token': [user_id_dict[int(elem)] for elem in text_data['p_id'].values],
        'item_id:token': [item_id_dict[int(elem)] for elem in text_data['d_id'].values],
        'rating:float': [1] * len(text_data)
    })
    inter.to_csv(target_path + 'OMC-100k.inter', sep='\t', index=False)

    # id_dict save
    user_id_dict_df = pd.DataFrame({
        'user_id:token': list(user_id_dict.values()),
        'real_id:token': list(user_id_dict.keys())
    })
    user_id_dict_df = user_id_dict_df.sort_values(by='user_id:token')
    user_id_dict_df.to_csv(target_path + 'user_id2real_id.txt', sep='\t', index=False)

    item_id_dict_df = pd.DataFrame({
        'item_id:token': list(item_id_dict.values()),
        'real_id:token': list(item_id_dict.keys())
    })
    item_id_dict_df = item_id_dict_df.sort_values(by='item_id:token')
    item_id_dict_df.to_csv(target_path + 'item_id2real_id.txt', sep='\t', index=False)

    # LDA related user_text file
    user_text = text_data[['p_id', 'query']]
    for i in range(len(user_text)):
        user_text.loc[i, 'p_id'] = user_id_dict[int(user_text.loc[i, 'p_id'])]
    user_text['user_id'] = user_text['p_id']
    user_text = user_text[['user_id', 'query']]
    user_text.to_csv(target_path + 'LDA_related/user_text.csv', sep='\t', index=False)

def doctor_name_extract():
    data_path = '../../dataset/DP_chaos/sampled/text_data.csv'
    data = pd.read_csv(data_path, sep='\t')
    data = data[['d_id', 'profile']]
    data = data.drop_duplicates(keep='first')
    data.reset_index(inplace=True)
    data['name'] = [data.loc[i, 'profile'].split('，')[0] for i in range(len(data))]
    data = data[['d_id', 'name', 'profile']]
    data.to_csv('../../dataset/OMC-100k/doctor2name.txt', sep='\t', index=False)

def docotr_id_name_hospital():
    id_name = pd.read_csv('../../dataset/OMC-100k/doctor_addition_info/doctor2name.txt', sep='\t')
    kg_data = pd.read_csv('../../dataset/OMC-100k/OMC-100k.kg', sep='\t')
    kg_data = kg_data[kg_data['relation_id:token'] == 'doctor.work_in.hospital']
    kg_data = kg_data.reset_index(drop=True)

    id2hos = collections.defaultdict(list)
    for i in range(len(kg_data)):
        id2hos[kg_data.loc[i, 'head_id:token']].append(kg_data.loc[i, 'tail_id:token'])

    hospitals = [id2hos[str(id_name.loc[i, 'd_id'])] if str(id_name.loc[i, 'd_id']) in id2hos.keys() else -1
                 for i in range(len(id_name))]

    id_name['hospital'] = hospitals
    final_results = id_name[['d_id', 'name', 'hospital']]
    final_results.to_csv('../../dataset/OMC-100k/doctor_addition_info/doctor_id_name_hospital.txt', sep='\t', index=False)


def get_att_dis(target, behaviored):
    attention_distribution = []

    for i in range(behaviored.size(0)):
        attention_score = torch.cosine_similarity(target, behaviored[i].view(1, -1))
        attention_distribution.append(attention_score)
    attention_distribution = torch.Tensor(attention_distribution)

    return attention_distribution

def get_similar_doctor():
    text_data = pd.read_csv('../../dataset/DP_chaos/sampled/text_data.csv', sep='\t')
    emb_path = '../../dataset/OMC-100k/bert_embeddings/q_embeddings.json'
    id_real_id = pd.read_csv('../../dataset/OMC-100k/user_id2real_id.txt', sep='\t')
    item_real_id = pd.read_csv('../../dataset/OMC-100k/item_id2real_id.txt', sep='\t')
    real_id2id = dict(zip(id_real_id['real_id:token'].values, id_real_id['user_id:token'].values))
    real_id2id_item = dict(zip(item_real_id['real_id:token'].values, item_real_id['item_id:token'].values))
    with open(emb_path, 'r', encoding='utf-8') as f:
        embeddings = json.load(f)

    pid_department = pd.read_csv('../../dataset/OMC-100k/multiple_query_filter_2940_clear.csv', sep='\t')
    pid2department = dict(zip(pid_department['pid'], pid_department['departs']))

    doctors_closed = []
    user_id_box = []
    max_num = 10
    for i in tqdm(range(len(text_data))):
        id = real_id2id[text_data.loc[i, 'p_id']]
        department_shot = eval(pid2department[text_data.loc[i, 'p_id']])
        users_same_department = text_data[text_data['department'].isin(department_shot)]
        # text_data[text_data['department'] == text_data.loc[i, 'department']]
        users_id = users_same_department['p_id'].values
        embs = torch.FloatTensor([embeddings[str(real_id2id[elem])] for elem in users_id])
        similarity = get_att_dis(torch.FloatTensor(embeddings[str(id)]), embs)
        box = [(real_id2id_item[text_data[text_data['p_id'] == users_id[i]]['d_id'].values[0]], similarity[i].tolist())
               for i in range(len(users_id))]
        box = sorted(box, key=operator.itemgetter(1), reverse=True)
        sub_doctors = []
        for did, score in box:
            if len(sub_doctors) >= max_num:
                break
            if did not in sub_doctors:
                sub_doctors.append(did)
        doctors_closed.append(sub_doctors)
        user_id_box.append(id)

    results = pd.DataFrame({
        'user_id': user_id_box,
        'doctor_closed': doctors_closed
    })
    results.to_csv('../../dataset/OMC-100k/user_id_closed_doctors.txt', sep='\t', index=False)

def add_interation():
    data = pd.read_csv('../../dataset/OMC-100k/OMC-100k_origin.inter', sep='\t')
    add_inter = pd.read_csv('../../dataset/OMC-100k/user_id_closed_doctors.txt', sep='\t')
    for i in tqdm(range(len(add_inter))):
        uid = add_inter.loc[i, 'user_id']
        doctors = eval(add_inter.loc[i, 'doctor_closed'])
        for doc in doctors[:20]:
            data.loc[len(data), :] = [str(uid), str(doc), str(1)]
    for col in data.columns:
        data[col] = data[col].apply(lambda x: format(int(x)))
    data.to_csv('../../dataset/OMC-100k/OMC-100k_add.inter', sep='\t', index=False)

def docotr_complete_info():
    kg_feat = pd.read_csv('../../dataset/OMC-100k/OMC-100k.kg', sep='\t')
    kg_feat = kg_feat[kg_feat['relation_id:token'] != 'doctor.member_of.institute']
    kg_feat = kg_feat.reset_index(drop=True)
    doctors = []
    diseases = []
    hospitals = []
    institutes = []
    cities = []
    provinces = []
    relations = ['cure_disease', 'member_of', 'work_in', 'city_of', 'addressed_in']
    for i in range(len(kg_feat)):
        head = kg_feat.loc[i, 'head_id:token']
        rela = kg_feat.loc[i, 'relation_id:token'].split('.')[1]
        tail = kg_feat.loc[i, 'tail_id:token']
        if rela not in ['city_of', 'addressed_in']:
            doctors.append(head)

        if rela == 'cure_disease':
            diseases.append(tail)
        elif rela == 'member_of':
            institutes.append(tail)
        elif rela == 'work_in':
            hospitals.append(tail)
        elif rela == 'city_of':
            cities.append(head)
            provinces.append(tail)
        elif rela == 'addressed_in':
            hospitals.append(head)
            cities.append(tail)

    doctors = list(set(doctors))
    diseases = list(set(diseases))
    hospitals = list(set(hospitals))
    institutes = list(set(institutes))
    cities = list(set(cities))
    provinces = list(set(provinces))

    edges_list = [(kg_feat.loc[i, 'head_id:token'], kg_feat.loc[i, 'tail_id:token'])
                  for i in range(len(kg_feat))]
    graph = nx.Graph()
    graph.add_edges_from(edges_list)

    target_df = pd.read_csv('../../dataset/OMC-100k/doctor_addition_info/doctor_id_name_hospital.txt', sep='\t')
    doctor_feat = pd.read_csv('../../dataset/OMC-100k/OMC-100k.item', sep='\t')
    columns = list(doctor_feat.columns)
    print(columns)
    [columns.remove(elem) for elem in ['department:single_str', 'doctor_title:single_str',
                                       'education_title:single_str', 'consultation_amount:float']]
    doctor_feat = doctor_feat[columns]
    columns.remove('item_id:token')
    id2feats = dict(zip(doctor_feat['item_id:token'].values, doctor_feat[columns].values))

    item_id2real_id = pd.read_csv('../../dataset/OMC-100k/item_id2real_id.txt', sep='\t')
    real_id2id = dict(zip(item_id2real_id['real_id:token'].values, item_id2real_id['item_id:token'].values))

    text_data = pd.read_csv('../../dataset/DP_chaos/text_data.csv', sep='\t')
    text_data = text_data[['d_id', 'department', 'profile']]
    text_data = text_data.drop_duplicates(keep='first')
    id2depart = dict(zip(text_data['d_id'].values, text_data['department'].values))
    id2profile = dict(zip(text_data['d_id'].values, text_data['profile'].values))

    departments = []
    profiles = []
    n_doctor = []
    n_disease = []
    n_hospital = []
    feats = []
    walk_length = 3

    for i in tqdm(range(len(target_df))):
        id = target_df.loc[i, 'd_id']
        departments.append(id2depart[id])
        profiles.append(id2profile[id])
        feats.append(id2feats[real_id2id[id]])
        entities = []
        sub = [id]
        step = 0
        while step < walk_length:
            sub = list(set([neighbor for elem in sub for neighbor in graph.neighbors(str(elem))]))
            [sub.remove(elem) for elem in entities if elem in sub]
            entities.extend(sub)
            step += 1
        entities = list(set(entities))
        n_doctor.append(sum([1 for elem in entities if elem in doctors]))
        n_disease.append(sum([1 for elem in entities if elem in diseases]))
        n_hospital.append(sum([1 for elem in entities if elem in hospitals]))

    columns = [elem.split(':')[0] for elem in columns]
    target_df['department'] = departments
    target_df[columns] = feats
    target_df['n_doctor'] = n_doctor
    target_df['n_disease'] = n_disease
    target_df['n_hospital'] = n_hospital
    target_df['profile'] = profiles
    target_df.to_csv('../../dataset/OMC-100k/doctor_addition_info/doctor_full_info.txt', sep='\t', index=False)

def sample_threshold():
    text_data = pd.read_csv('../../dataset/DP_chaos/sampled/text_data.csv', sep='\t')
    emb_path = '../../dataset/OMC-100k/bert_embeddings/q_embeddings.json'
    id_real_id = pd.read_csv('../../dataset/OMC-100k/user_id2real_id.txt', sep='\t')
    item_real_id = pd.read_csv('../../dataset/OMC-100k/item_id2real_id.txt', sep='\t')
    real_id2id = dict(zip(id_real_id['real_id:token'].values, id_real_id['user_id:token'].values))
    real_id2id_item = dict(zip(item_real_id['real_id:token'].values, item_real_id['item_id:token'].values))
    with open(emb_path, 'r', encoding='utf-8') as f:
        embeddings = json.load(f)

    doctors_closed = []
    user_id_box = []
    threshold = 0.95
    max_num = 20
    sample_num_to_cal = 1000
    for i in tqdm(range(len(text_data))):
        id = real_id2id[text_data.loc[i, 'p_id']]
        users_differ_department = text_data[text_data['department'] != text_data.loc[i, 'department']]
        users_id = random.sample(list(users_differ_department['p_id'].values), sample_num_to_cal)
        embs = torch.FloatTensor([embeddings[str(real_id2id[elem])] for elem in users_id])
        similarity = get_att_dis(torch.FloatTensor(embeddings[str(id)]), embs)
        box = [(real_id2id_item[text_data[text_data['p_id'] == users_id[i]]['d_id'].values[0]], similarity[i].tolist())
               for i in range(len(users_id))]
        box = sorted(box, key=operator.itemgetter(1), reverse=True)
        sub_doctors = []
        for did, score in box:
            if score < threshold or len(sub_doctors) >= max_num:
                break
            if did not in sub_doctors:
                sub_doctors.append(did)
                # print(did, score)
        doctors_closed.append(sub_doctors)
        user_id_box.append(id)

    results = pd.DataFrame({
        'user_id': user_id_box,
        'doctor_closed': doctors_closed
    })
    results.to_csv('../../dataset/OMC-100k/user_id_closed_doctors_differ.txt', sep='\t', index=False)

def kg_diversity():
    kg = pd.read_csv('../../dataset/OMC-100k/OMC-100k.kg', sep='\t')
    kg_without_kg = kg[kg['relation_id:token'] != 'doctor.cure_disease.disease']
    kg_without_rg = kg[kg['relation_id:token'] == 'doctor.cure_disease.disease']
    kg_without_kg.to_csv('../../dataset/OMC-100k/OMC-100k_wo_kg.kg', sep='\t', index=False)
    kg_without_rg.to_csv('../../dataset/OMC-100k/OMC-100k_wo_rg.kg', sep='\t', index=False)

def sample_based_on_symptom():
    combination = pd.read_csv('../../dataset/OMC-100k/combinations_inter.txt', sep=':')
    combination['words_split'] = combination['words'].apply(lambda x: x.split(', '))

    text_data = pd.read_csv('../../dataset/DP_chaos/text_data.csv', sep='\t')
    symptom_data = pd.read_csv('../../dataset/OMC-100k/symptom.txt')
    symptoms = [elem.strip().replace('.', '') for elem in list(symptom_data['symptoms'].values)]
    print(symptoms)

    words_all = []
    # count = []
    for elem in list(combination['words_split'].values):
        words_all.extend(elem)
        # count.append(len(elem))
    words_all = list(set(words_all))
    # print(words_all)
    # print(max(count), min(count))

    index = []
    keywords = []
    for i in tqdm(range(len(text_data))):
        wrd_count = 0
        query = text_data.loc[i, 'query']
        dialogue = text_data.loc[i, 'dialogue']
        qd = query  # + dialogue if type(dialogue) == str else query
        sub_wrd = []
        for wrd in symptoms:
            if qd.find(wrd) >= 0:
                sub_wrd.append(wrd)
                wrd_count += 1
        keywords.append(sub_wrd)
        if len(sub_wrd) > 0:
            index.append(i)
    text_data['keywords'] = keywords
    final_data = text_data.loc[index, :]

        # if wrd_count >= 5:
        #     index.append(i)

    # final_data = text_data.loc[index, :]
    final_data.to_csv('../../dataset/DP_chaos/sampled/text_data_highlight.csv', sep='\t', index=False)

def filter_based_on_department():
    combination = pd.read_csv('../../dataset/OMC-100k/combinations_inter.txt', sep=':')
    print(combination.columns)
    combination['words_split'] = combination['words'].apply(lambda x: x.split(', '))
    combination['words_split'] = [[elem.strip() for elem in combination.loc[i, 'words_split']]
                                  for i in range(len(combination))]
    combination['department_split'] = combination['departments'].apply(lambda x: x.split(' , '))
    combination['department_split'] = [[elem.strip() for elem in combination.loc[i, 'department_split']]
                                       for i in range(len(combination))]

    text_data = pd.read_csv('../../dataset/DP_chaos/sampled/text_data_query_dialogue.csv', sep='\t')
    index_save = []
    for i in tqdm(range(len(text_data))):
        query = text_data.loc[i, 'query']
        dialogue = text_data.loc[i, 'dialogue']
        sentence = query + dialogue if type(dialogue) == str else query
        department = text_data.loc[i, 'department']
        for j in range(len(combination)):
            words = combination.loc[j, 'words_split']
            departments = combination.loc[j, 'department_split']
            count = 0
            for wrd in words:
                if sentence.find(wrd) >= 0:
                    count += 1
            print(count)
            if count >= 5:
                if department in departments:
                    index_save.append(i)
    final_result = text_data.loc[index_save, :]
    final_result.to_csv('../../dataset/DP_chaos/sampled/text_data_core.csv', sep='\t', index=False)

def text_data_fliter():
    shot1 = pd.read_csv('../../dataset/OMC-100k/multiple_query_filter_2940_clear.csv', sep='\t')
    shot2 = pd.read_csv('../../dataset/OMC-100k/single_query_filter_6233_clear.csv', sep='\t')
    origin_data = pd.read_csv('../../dataset/DP_chaos/text_data.csv', sep='\t')
    pid1 = list(set(shot1['pid']))
    pid2 = list(set(shot2['pid']))
    pid = pid1

    index = []
    for i in range(len(origin_data)):
        if origin_data.loc[i, 'p_id'] in pid and not pd.isnull(origin_data.loc[i, 'profile']):
            index.append(i)

    data = origin_data.loc[index, :]
    data.to_csv('../../dataset/DP_chaos/sampled/text_data.csv', sep='\t', index=False)

if __name__ == '__main__':
    # data_extract_kg()
    # text_data_process()
    # sample_data()
    # text_data_fliter()
    # user_process()
    # item_process()
    # idilize_and_inter_link_file()

    # hospital_address_triple()
    # hospital_address_triple_addition()

    # doctor_name_extract()
    # docotr_id_name_hospital()
    # get_similar_doctor()
    add_interation()
    # docotr_complete_info()
    # sample_threshold()
    # kg_diversity()
    # sample_based_on_symptom()
    # filter_based_on_department()
