import torch
from transformers import BertModel, BertTokenizer
import pandas as pd
import json
import numpy as np


def cal_bert(embed_df):
    lines = embed_df['content'].tolist()

    bertTweet = BertModel.from_pretrained("hfl/chinese-bert-wwm")  # “vinai/bertweet-base” 
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm")  # “vinai/bertweet-base”
    # INPUT TWEET IS ALREADY NORMALIZED!

    con_emb_dict = {}
    for index, line in enumerate(lines):
        print(str(index)+'  '+line)
        input_ids = torch.tensor([tokenizer.encode(line)])
        if len(input_ids[0].numpy().tolist()) > 128:
            input_ids = torch.from_numpy(np.array(input_ids[0].numpy().tolist()[0:128])).reshape(1, -1).type(torch.LongTensor)

        with torch.no_grad():
            features = bertTweet(input_ids)  # Models outputs are now tuples
        con_emb_dict[line] = features[1].numpy()[0].tolist()

    jsObj = json.dumps(con_emb_dict)

    fileObj = open('./weiData/embeddings.json', 'w')
    fileObj.write(jsObj)
    fileObj.close()


def test_dict(embed_df, con_emb_dict):
    print(con_emb_dict)
    content_list = embed_df['content'].tolist()
    for content in content_list:
        try:
            print(con_emb_dict[content])
        except:
            print(content)
            print('\n\n\n')


if __name__ == '__main__':
    embed_df = pd.read_table('./weiData/embedSet.csv')
    #embed_df = embed_df[5434:5435]
    cal_bert(embed_df)

    #with open('./embeddings.json', 'r') as f:
        #con_emb_dict = json.load(f)
    #test_dict(embed_df, con_emb_dict)


