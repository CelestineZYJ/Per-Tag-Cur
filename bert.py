import ast
import json
import random
import re
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

def average_user_tweet(user_list, content_user_df, con_emb_dict):
    user_arr_dict = OrderedDict()
    for user in tqdm(user_list, desc="average_user_tweet"):
        embed_list = []
        content_list = content_user_df[content_user_df['user_id'] == user].content.tolist()[0]
        for content in content_list:
            try:
                embed_list.append(con_emb_dict[content])
            except:
                embed_list.append([0]*768)
                print(1)
        embed_list = np.mean(np.array(embed_list), axis=0)
        user_arr_dict[user] = embed_list
    return user_arr_dict


def average_hashtag_tweet(tag_list, train_content_tag_df, con_emb_dict):
    tag_arr_dict = OrderedDict()
    for index, tag in enumerate(tqdm(tag_list, desc="average_hashtag_tweet")):
        embed_list = []
        try:
            content_list = train_content_tag_df[train_content_tag_df['hashtag'] == tag].content.tolist()[0]
        except:
            print(tag)
        for content in content_list:
            try:
                embed_list.append(con_emb_dict[content])
            except:
                embed_list.append([0]*768)
                print(1)
        embed_list = np.mean(np.array(embed_list), axis=0)
        tag_arr_dict[tag] = embed_list
    return tag_arr_dict


def rank_input_train(weibo, user_list, train_tag_list, user_arr_dict, tag_arr_dict, qid_train_dict):
    if(weibo):
        f = open('./weibo/weibo_part/bert/trainBert.dat', "a", encoding="utf-8")
    else:
        f = open('/mnt/zyb/twitter/twitter_half/bert/train(>10)/trainBert.dat', "a", encoding="utf-8")

    for user_num, user in enumerate(tqdm(user_list, desc="rank_input_train")):
        user_arr = user_arr_dict[user]
        f.write(f"# query {user_num + 1}")

        positive_tag_list = sorted(list(set(qid_train_dict[user])))
        temp_tag_list = sorted(list(set(train_tag_list) - set(positive_tag_list)))
        negative_tag_list = random.sample(temp_tag_list, 5*len(positive_tag_list))

        for tag in positive_tag_list:  # positive samples
            #try:
            tag_arr = tag_arr_dict[tag]
            user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
            x = 1
            Str = f"\n{x} {'qid'}:{user_num + 1}"
            for index, value in enumerate(user_tag_arr):
                Str += f" {index + 1}:{value}"
            f.write(Str)
            #except:
                #print(tag)

        for tag in negative_tag_list:  # negative samples
            #try:
            tag_arr = tag_arr_dict[tag]
            user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
            x = 0
            Str = f"\n{x} {'qid'}:{user_num + 1}"
            for index, value in enumerate(user_tag_arr):
                Str += f" {index + 1}:{value}"
            f.write(Str)
            #except:
                #print(tag)
        f.write("\n")


def rank_input_test(weibo, user_list, test_tag_list, user_arr_dict, qid_test_dict, qid_train_dict, train_cont_tag_df):
    if(weibo):
        f = open('./weibo/weibo_part/bert/testBert.dat', "a", encoding="utf-8")
        tagF = open('./weibo/weibo_part/bert/tagList.txt', "a", encoding="utf-8")
    else:
        f = open('/mnt/zyb/twitter/twitter_half/bert/train(>10)/testBert.dat', "a", encoding="utf-8")
        tagF = open('/mnt/zyb/twitter/twitter_half/bert/train(>10)/tagList.txt',"a", encoding="utf-8")

    for user_num, user in enumerate(tqdm(user_list, desc="rank_input_test")):
        user_arr = user_arr_dict[user]
        f.write(f"# query {user_num + 1}")
        tagF.write(f"# query {user_num + 1}\n")

        positive_tag_list = sorted(list(set(qid_test_dict[user]) - set(qid_train_dict[user])))
        spe_neg_list = sorted(list(set(test_tag_list) - set(positive_tag_list) - set(qid_train_dict[user])))
        
        spe_test_df = test_df.loc[(test_df['user_id'] != user)]
        spe_test_content_tag_df = spe_test_df.explode('hashtag').groupby(['hashtag'], as_index=False).agg({'content': lambda x: list(x)})
        
        positive_tag_list = sorted(list(set(positive_tag_list) & set(spe_test_content_tag_df.hashtag)))
        try:
            negative_tag_list = random.sample(spe_neg_list,100*len(positive_tag_list))
        except:
            print(user)
        for tag in positive_tag_list:  # positive samples
            #try:

            embed_list = []
            try:
                content_list = train_cont_tag_df[train_cont_tag_df.hashtag == tag].content.tolist()[0]  # tag list of train part
            except:
                content_list = []
            content_list.extend(spe_test_content_tag_df[spe_test_content_tag_df.hashtag == tag].content.tolist()[0])  # tag list of test part
            for content in content_list:
                try:
                    embed_list.append(con_emb_dict[content])
                except:
                    embed_list.append([0]*768)
                    print(1)
            tag_arr = np.mean(np.array(embed_list), axis=0)

            user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
            x = 1
            Str = f"\n{x} {'qid'}:{user_num + 1}"
            tagF.write(f"{x}_{tag}\n")
            for index, value in enumerate(user_tag_arr):
                Str += f" {index + 1}:{value}"
            f.write(Str)
            #except:
                #print(tag)

        for tag in negative_tag_list:  # negative samples
            #try:
            embed_list = []
            try:
                content_list = train_cont_tag_df[train_cont_tag_df.hashtag == tag].content.tolist()[0]  # tag list of train part
            except:
                content_list = []
            if len(spe_test_content_tag_df[spe_test_content_tag_df.hashtag == tag].content.tolist()) == 0:
                print(user)
                continue
            content_list.extend(spe_test_content_tag_df[spe_test_content_tag_df.hashtag == tag].content.tolist()[0])  # tag list of test part
            for content in content_list:
                try:
                    embed_list.append(con_emb_dict[content])
                except:
                    embed_list.append([0]*768)
                    print(1)
            tag_arr = np.mean(np.array(embed_list), axis=0)

            user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
            x = 0
            Str = f"\n{x} {'qid'}:{user_num + 1}"
            tagF.write(f"{x}_{tag}\n")
            for index, value in enumerate(user_tag_arr):
                Str += f" {index + 1}:{value}"
            f.write(Str)
            #except:
                #print(tag)
        f.write("\n")
        

def sort_user_tag(user_list, df):
    tag_list = sorted(
        list(set(df['hashtag'].explode('hashtag').tolist()))) #sorted(list(set())) to keep the fixed order
    qid_user_tag_dict = OrderedDict() 
    for user in tqdm(user_list, desc="sort_user_tag"):
        spe_user_df = df.loc[df['user_id'] == user]
        spe_user_tag_list = sorted(
            list(set(spe_user_df['hashtag'].explode('hashtag').tolist())))
        qid_user_tag_dict[user] = spe_user_tag_list
    return tag_list, qid_user_tag_dict


def read_para(train_df, weibo):
    #read and parse user list
    if(weibo):
        with open("./weibo/weibo_part/userList.txt", "r", encoding="utf-8") as f:
            user_list = ast.literal_eval(f.readline().strip())
    else:
        with open("/mnt/zyb/twitter/twitter_half/userList1.txt", "r", encoding="utf-8") as f:
            user_list = ast.literal_eval(f.readline().strip())
    
    # user1 [con1, con2, con3, ...]
    content_user_df = train_df.groupby('user_id', as_index=False).agg({'content': lambda x: list(x)})
    # tag1 [con1, con2, con3, ...] 
    train_content_tag_df = train_df.explode('hashtag').groupby('hashtag', as_index=False).agg({'content': lambda x: list(x)})
    #all tags
    train_tag_list = sorted(list(set(train_df['hashtag'].explode('hashtag').tolist())))

    if(weibo):
        with open("./weibo/weibo_part/tagList.txt",'w',encoding="utf-8") as f:
            f.writelines([tag+'\n' for tag in train_tag_list])
        content_user_df.to_csv("./weibo/weibo_part/bert/userCons.csv",index = False,sep='\t',encoding="utf-8")
        train_content_tag_df.to_csv("./weibo/weibo_part/bert/trainTagCons.csv",index = False,sep='\t',encoding="utf-8")
    else:
        with open("/mnt/zyb/twitter/twitter_half/tagList.txt",'w',encoding="utf-8") as f:
            f.writelines([tag+'\n' for tag in train_tag_list])
        content_user_df.to_csv("/mnt/zyb/twitter/twitter_half/bert/train(>10)/userCons.csv",index = False,sep='\t',encoding="utf-8")
        train_content_tag_df.to_csv("/mnt/zyb/twitter/twitter_half/bert/train(>10)/trainTagCons.csv",index = False,sep='\t',encoding="utf-8")
    emb_para_list = [user_list, content_user_df, train_tag_list, train_content_tag_df]

    return emb_para_list


if __name__ == '__main__':
    '''
    python bert.py --weibo 
    --weibo choose weibo (default: 1)
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--weibo", default=1, type=int)
    args = parser.parse_args()

    if(args.weibo == True):
        train_path = "./weibo/weibo_part/train.csv"
        test_path = "./weibo/weibo_part/test.csv"
        embed_json = "./weibo/weibo_part/embed.json"
    else:
        train_path = "/mnt/zyb/twitter/twitter_half/train.csv"
        test_path = "/mnt/zyb/twitter/twitter_half/test.csv"
        embed_json = "/mnt/zyb/twitter/twitter_half/embed.json"

    train_df = pd.read_csv(train_path, delimiter='\t', encoding="utf-8", dtype={'user_id':str,'content':str})
    test_df = pd.read_csv(test_path, delimiter='\t', encoding="utf-8", dtype={'user_id':str,'content':str})

    #read content embedding json, get from pretrainBert.py
    with open(embed_json, 'r', encoding="utf-8") as f:
        con_emb_dict = json.load(f) 

    #format: hashtag -->list  
    train_df['hashtag'] = train_df['hashtag'].apply(ast.literal_eval)
    test_df['hashtag'] = test_df['hashtag'].apply(ast.literal_eval)

    emb_para_list = read_para(train_df, args.weibo)
    emb_para_list.append(con_emb_dict)
    #emb_para_list = [0:user_list, 1:content_user_df, 2:train_tag_list, 3:train_content_tag_df, 4:con_emb_dict]
    user_arr_dict = average_user_tweet(emb_para_list[0], emb_para_list[1], emb_para_list[4])
    train_tag_arr_dict = average_hashtag_tweet(emb_para_list[2], emb_para_list[3], emb_para_list[4])

    train_tag, qid_train_dict = sort_user_tag(emb_para_list[0], train_df)
    test_tag, qid_test_dict = sort_user_tag(emb_para_list[0], test_df)

    rank_input_train(args.weibo, emb_para_list[0], train_tag, user_arr_dict, train_tag_arr_dict,qid_train_dict)
    rank_input_test(args.weibo, emb_para_list[0], test_tag, user_arr_dict, qid_test_dict, qid_train_dict, emb_para_list[3])
