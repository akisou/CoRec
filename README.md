# Bridging Expertise: Doctor Recommendations for Cross-Disciplinary Collaborations in Online Medical Consultations
This is our Pytorch implementation for the paper:
> Xiaonan Wu, Zhiying Li, Hongxun Jiang and Xun Liang(2024). Bridging Expertise: Doctor Recommendations for Cross-Disciplinary Collaborations in Online Medical Consultations
# [Paper in Arxiv]. In CSCW 2024

## Introduction
Doctor Recommendations for Cross-Disciplinary Collaborations in Online Medical Consultations (CoRec) is an 
innovative framework for interdisciplinary doctor recommendations for collaborative work in OMC scenarios,
comprising two modules: one for calculating expertise and another for collaborative computing. 
Leveraging the medical knowledge graph, the former employs an expertise encoder to derive specialty embeddings 
for patient queries, doctor profiles, and historical consultations. The latter builds a collaborative circle 
for each doctor based on three dimensions: scientific citation, academic collaboration, and spatiotemporal proximity.
This circle analysis delves into the potential for cooperation between doctors, aiming to identify optimal combinations. 

## Requirement
The code has been tested running under Python 3.9.13. The required packages are as follows:
- torch>=1.10.0
- numpy>=1.17.2
- scipy>=1.6.0
- hyperopt==0.2.5
- pandas>=1.4.0
- tqdm>=4.48.2
- scikit_learn>=0.23.2
- pyyaml>=5.1.0
- colorlog==4.7.2
- colorama==0.4.4
- tensorboard>=2.5.0
- thop>=0.1.1.post2207130030
- ray>=1.13.0
- tabulate>=0.8.10 
- plotly>=4.0.0
- texttable>=0.9.0
- networkx==2.7
- geopy==2.2.0
- jieba==0.42.1
- protobuf==3.19.4
- xlwt==1.3.0

## Usage
The relevant models in this article are designed and trained based on the basic code of the recbole repository of Renmin
University of China. Relevant model and training settings are set in test_corec.yaml, including the optimal model file 
path that has been trained. Execute run_recbole.py to start training and testing.


## Dataset

We provide the dataset we have organized: OMC-100k.

|                       |               | OMC-100k |
| :-------------------: |:--------------|---------:|
| User-Item Interaction | #Users        |     2936 |
|                       | #Items        |      352 |
|                       | #Interactions |    32297 |
|    Knowledge Graph    | #Entities     |     3448 |
|                       | #Relations    |        6 |
|                       | #Triplets     |    19900 |
|                       | #Sparsity     |   96.88% |

