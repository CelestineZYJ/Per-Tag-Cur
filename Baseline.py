import ast
import math
import re
import time as t
from collections import Counter

import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm import tqdm

np.random.seed(int(t.time()))


def get_user_tags(user, dataframe):
    user_tag = []
    for tag in list(dataframe[dataframe['user_id'] == user]['hashtag']):
        user_tag.extend(ast.literal_eval(tag))
    return list(set(user_tag))


def random_rec(user, dataframe):
    tag_rec = []
    for i in range(5):
        tag_rec.append(dataframe['hashtag'].loc[np.random.randint(1000)])
    return tag_rec


def popular_rec(user, dataframe):
    tag_rec = []
    for i in range(5):
        tag_rec.append(dataframe['hashtag'].loc[i])
    return tag_rec


def latest_rec(user, dataframe):
    tag_rec = []
    dataframe = dataframe.sort_values(by=['time','tweet_id'], ascending=[False,False])  # time
    tags = get_user_tags(user, dataframe)
    tag_rec = tags[:5]
    return list(tag_rec)

def eval_result(func, name, user_set, dataframe, user_test_df):
    success = 0
    ap = 0
    ap_sum = 0
    p = 0
    relate = []
    ndcg_sum = 0
    success_user = False
    count_success_user = 0

    for user in tqdm(user_set, desc='Eval: '+name):                         # 重复出现的user要记得筛掉
        ap = 0
        success = 0
        success_user = False
        relate = []

        rec_tag_list = func(user, dataframe)
        user_tags = get_user_tags(user, user_test_df)

        for ind, tag in enumerate(rec_tag_list):
            if tag in user_tags:
                success_user = True
                success += 1
                ap += success/(ind+1)
                relate.append(1)
            else:
                relate.append(0)
        ap /= 5
        ap_sum += ap
        p += success
        if(success_user):
            dcg = sum([(pow(2, rel) - 1) / math.log(i+2, 2)
                       for i, rel in enumerate(relate)])
            idcg = sum([(pow(2, rel) - 1) / math.log(i+2, 2)
                        for i, rel in enumerate(sorted(relate, reverse=True))])

            ndcg_sum += dcg/idcg
            #ndcg_sum += metrics.ndcg_score(latest_tag_list,relate)
            count_success_user += 1

    print("{} MAP: {}".format(name,str(ap_sum/len(user_set))))
    print("{} P@5: {}".format(name,str(p/(5*len(user_set)))))
    print("There are {} users being recommend successfully".format(count_success_user))
    print("{} nDCG: {}".format(name,str(ndcg_sum/count_success_user)))


def eval_rec(dataframe1, dataframe2, dataframe3):
    user_train_df = dataframe1.drop(
        ['tweet_id', 'time', 'content', 'hashtag'], axis=1)
    user_set = set(list(user_train_df['user_id']))
    print(len(user_set))
    user_test_df = dataframe2.drop(['tweet_id', 'time'], axis=1)

    
    eval_result(latest_rec, 'Latest', user_set, dataframe1, user_test_df)
    eval_result(random_rec, 'Random', user_set, dataframe3, user_test_df)
    eval_result(popular_rec, 'Popular', user_set, dataframe3, user_test_df)


if __name__ == "__main__":
    train_path = "./data/twitter/needToCleanContent/trainSet.csv"
    test_path = "./data/twitter/needToCleanContent/testSet.csv"
    freq_tag_path = "./data/twitter/needToCleanContent/trainTag.csv"

    train_df = pd.read_csv(train_path, delimiter='\t', encoding="utf-8")
    test_df = pd.read_csv(test_path, delimiter='\t', encoding="utf-8")
    freq_tag_df = pd.read_csv(freq_tag_path, delimiter='\t', encoding="utf-8")
    eval_rec(train_df, test_df, freq_tag_df)
