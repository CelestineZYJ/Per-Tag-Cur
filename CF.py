from load_data import data
from recommender import algo
import re
import pandas as pd
from tqdm import tqdm


dataPath = 't'
classifierPath = 'CF'


def get_hashtag(content):
    hashtag = re.findall(r"['\'](.*?)['\']", str(content))
    return hashtag


def get_str(content):
    Str = str(content)
    return Str


with open('./'+dataPath+'Data/userList.txt', "r") as f:
    x = f.readlines()[0]
    user_list = get_hashtag(x)


test_df = pd.read_table('./'+dataPath+'Data/test.csv')
test_df['user_id'] = test_df['user_id'].apply(get_str)
test_df['hashtag'] = test_df['hashtag'].apply(get_hashtag)
test_tag_list = list(set(test_df['hashtag'].explode('hashtag').tolist()))

test_user_tag_dict = {}
for user in user_list:
    spe_user_df = test_df.loc[test_df['user_id'] == user]
    spe_user_tag_list = list(set(spe_user_df['hashtag'].explode('hashtag').tolist()))
    test_user_tag_dict[user] = spe_user_tag_list

trainingSet = data.build_full_trainset()

algo.fit(trainingSet)

testF2 = open('./'+dataPath+classifierPath+'/test'+classifierPath+'2.dat', "a")
preF = open('./'+dataPath+classifierPath+'/pre'+classifierPath+'.txt', 'a')
for user_num, user in enumerate(user_list):
    print(user_num)
    testF2.write(f"# query {user_num + 1}\n")
    positive_tag_list = list(set(test_user_tag_dict[user]))
    for tag in positive_tag_list:
        testF2.write(f"1 qid:{user_num+1}\n")
        precision = algo.predict(user, tag)
        preF.write(f"{precision.est}\n")
    negative_tag_list = list(set(test_tag_list) - set(positive_tag_list))
    for tag in negative_tag_list:
        testF2.write(f"0 qid:{user_num + 1}\n")
        precision = algo.predict(user, tag)
        preF.write(f"{precision.est}\n")
testF2.close()
preF.close()
