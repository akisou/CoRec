# -*- coding: utf-8 -*-
from gensim import corpora
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import warnings
import re
import os
import jieba
import operator
import logging
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
warnings.filterwarnings('ignore')

class LDA():
    # LDA class and method
    def __init__(self, path, inter, user, col, max_num):
        self.dataset_path = path
        self.users = list(set(inter[user].tolist()))
        self.USERFIELD = user
        self.TEXTFIELD = col
        self.SPLITFIELD = 'split'
        self.max_num_of_topics = max_num
        self.assist_path = "C:/Users/72444/Desktop/CoRec/CoRec-main/CoRec/model/knowledge_aware_recommender/LDA/"

    def predict(self):
        save_path = self.dataset_path + 'user_lda_classify_result.txt'
        if os.path.exists(save_path):
            return pd.read_csv(save_path, sep='\t')

        # train process
        best_model, users, dataset, corpus_full = self.get_best_ldamodel_data()
        # print(best_model.print_topics())

        # save best model
        model_save_path = self.assist_path + '/ldamodel_history/model_best.model'
        best_model.save(model_save_path)

        # get lda classification results
        predict_results = best_model.get_document_topics(corpus_full)
        # print(list(predict_results))
        final_results = [elem[0][0] for elem in predict_results]

        # save
        save_result = pd.DataFrame({
            self.USERFIELD: dataset[self.USERFIELD],
            'class': final_results
        })
        save_result.to_csv(save_path, sep='\t', index=False)
        return save_result

    def get_best_ldamodel_data(self):
        dataset_path = self.dataset_path + 'user_text_spilt.csv'
        if os.path.exists(dataset_path):
            dataset = pd.read_csv(dataset_path, sep='\t')
        else:
            dataset = self.split_words(self.USERFIELD, self.TEXTFIELD)

        # select users

        # select users and transfer to list
        users_selected = [dataset.loc[i, self.USERFIELD] for i in range(len(dataset))
                          if dataset.loc[i, self.USERFIELD] in self.users]
        split_set = [dataset.loc[i, self.SPLITFIELD].split() for i in range(len(dataset))
                     if dataset.loc[i, self.USERFIELD] in self.users]

        # dictionary construction
        dictionary = corpora.Dictionary(split_set)
        corpus = [dictionary.doc2bow(text) for text in split_set]

        # full version
        split_set_full = [dataset.loc[i, self.SPLITFIELD].split() for i in range(len(dataset))]
        corpus_full = [dictionary.doc2bow(text) for text in split_set_full]

        # find the best num_topics
        # num_topics_performance = []
        # for choice in range(5, self.max_num_of_topics, 1):
        #     ldamodel = LdaModel(corpus, num_topics=choice, id2word=dictionary, alpha="auto", eta="auto", passes=30)
        #     # topic_list = ldamodel.print_topics()
        #     save_path = self.assist_path + '/ldamodel_history/model_' + str(choice) + '.model'
        #     ldamodel.save(save_path)
        #     del ldamodel
        #     ldamodel = LdaModel.load(save_path)
        #     coherence_score = self.coherence(split_set, ldamodel, dictionary)
        #     num_topics_performance.append((choice, coherence_score))

        num_topics_performance = [(12, 5), (13, 1)]

        # self.draw_train_process(num_topics_performance)
        best_num_topics = sorted(num_topics_performance, key=operator.itemgetter(1), reverse=True)[0]
        best_ladmodel = LdaModel(corpus, num_topics=best_num_topics[0], id2word=dictionary,
                                 alpha="auto", eta="auto", passes=30)

        return [best_ladmodel, users_selected, dataset, corpus_full]

    def perplexity(self, ldamodel, corpus):
        # calculate perplexity
        return ldamodel.log_perplexity(corpus)

    def coherence(self, split_set, ldamodel, dictionary):
        # calculate coherence
        ldacm = CoherenceModel(model=ldamodel, texts=split_set, dictionary=dictionary, coherence='c_v')
        return ldacm.get_coherence()

    def draw_train_process(self, num_topics_performance):
        # draw coherence results
        num_topics = [elem[0] for elem in num_topics_performance]
        coh_list = [elem[1] for elem in num_topics_performance]
        plt.plot(num_topics, coh_list)
        plt.xlabel('num of topics')
        plt.ylabel('coherence value')
        plt.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False
        plt.title('num_topics - coherence vary figure')
        plt.show()

    def split_words(self, user, col):
        # formal splitting words process
        rpath = self.dataset_path
        old_file_name = 'user_text.csv'
        new_file_name = 'user_text_spilt.csv'
        dataset = pd.read_csv(rpath + old_file_name, sep='\t')
        if not (col in dataset.columns and col in dataset.columns):
            raise ValueError(f"{col} is wrong column name")
        users = dataset[user]
        text = dataset[col]
        split_words = []
        for (uid, txt) in zip(users, text):
            sub_split_words = self.text_pre_processing(txt)
            sub_split_words = self.seg_depart(sub_split_words)
            split_words.append(sub_split_words)

        dataset[self.SPLITFIELD] = split_words

        # save file
        if not os.path.exists(rpath + new_file_name):
            dataset.to_csv(rpath + new_file_name, sep='\t', index=0)

        return dataset[[user, self.SPLITFIELD]]

    def stopwordslist(self):
        # create stop_words list
        path = self.assist_path + 'Chinese_stop_words.txt'
        stopwords = [line.strip() for line in open(path, encoding='UTF-8').readlines()]
        return stopwords

    def text_pre_processing(self, text):
        # self customized words washing
        # text = re.sub("@.+?( |$)", "", text)  # 去除 @xxx (用户名)
        # text = re.sub("【.+?】", "", text)  # 去除 【xx】 (里面的内容通常都不是用户自己写的)
        # text = re.sub(".*?:", "", text)  # 去除微博用户的名字
        # # text = re.sub("#.*#", "", text)  # 去除话题引用
        text = re.sub("\n", "", text)
        text = re.sub("\t", "", text)
        text = re.sub("(\[^\u4e00-\u9fa5\u0030-\u0039\])", "", text)

        return text

    def seg_depart(self, sentence):
        # chinese split for sentence and stop words
        dict_path = self.assist_path + "userdict.txt"
        jieba.load_userdict(dict_path)
        sentence_depart = jieba.cut(sentence.strip())
        stopwords = self.stopwordslist()
        # output result
        outstr = ''
        # stop words
        for word in sentence_depart:
            if word not in stopwords:
                if word != '\t':
                    outstr += word + " "
        return outstr







