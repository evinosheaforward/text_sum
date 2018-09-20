# Work towards a text summarization model
## The main flaw with this is the lack of pre-processing. The sentances are not separated ideally
## The model works using various doc2vec models. It then finds the sentances with the highest inner product 
## (i.e. most relevant to the rest of the article) and outputs those sentances in order
