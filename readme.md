# Personalized Hashtag Recommendation
## Data
### Weibo

- Basic Statistics

+----------+-----------+-------+-------+------+
| Quantity | Embedding | Train | Valid | Test |
+----------+-----------+-------+-------+------+
|  Weibo   |   90572   | 87769 |  1402 | 1401 |
|   User   |    215    |  215  |  215  | 215  |
| Hashtag  |   58596   | 56865 |  1539 | 1530 |
+----------+-----------+-------+-------+------+

- Distribution

### Twitter

- Basic Statistics

| Num | Embed | Train | Test |
| --- | --- | --- | --- |
| Tweet | TODO | TODO | TODO |
| User | TODO | TODO | TODO |
| Hashtag | TODO | TODO | TODO |

- Distribution

---

## Model
- bert+lstm+mlp
- bert+mlp
- bert+gbdt
- Lda+gbdt
- tfidf+gbdt

### ranklib:

> train: java -jar RankLib-2.14.jar -train trainBert.dat -ranker 0 -save modelBertGbdt.dat

> test: java -jar RankLib-2.14.jar -load modelBertGbdt.dat -test testBert.dat -metric2T MAP@5 -idv BertGbdt.MAP@5.txt 

** Description ** :

1. train.dat & test.dat example
#query user_index
  label(0 or 1, 1 positive) qid:user_index features...
  ...
2. BertGbdt.MAP@5.txt : MAP of each user (qid)
