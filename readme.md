# Personalized Hashtag Recommendation
## Data
### Weibo

- Basic Statistics

| Num | Embed | Train | Test |
| --- | --- | --- | --- |
| Weibo | TODO | TODO | TODO |
| User | TODO | TODO | TODO |
| Hashtag | TODO | TODO | TODO |

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

** Description **:

1. train.dat & test.dat example
#query user_index
  label(0 or 1, 1 positive) qid:user_index features...
  ...
2. BertGbdt.MAP@5.txt : MAP of each user (qid)
