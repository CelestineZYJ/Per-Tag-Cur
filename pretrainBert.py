import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import json

embed_df = pd.read_table('./data/embedSet.csv')
embed_df = embed_df.loc[:10]
lines = embed_df['content'].tolist()

bertTweet = AutoModel.from_pretrained("vinai/between-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/between-base")
# INPUT TWEET IS ALREADY NORMALIZED!

con_emb_dict = {}
for line in lines:
    input_ids = torch.tensor([tokenizer.encode(line)])
    with torch.no_grad():
       features = bertTweet(input_ids)  # Models outputs are now tuples
    con_emb_dict[line] = features
    
jsObj = json.dumps(con_emb_dict)

fileObj = open('embeddings.json', 'w')
fileObj.write(jsObj)
fileObj.close()

## with Tensorflow 2.0+:
# from transformers import TFAutoModel
# bertweet = TFAutoModel.from_pretrained("vinai/between-base")
