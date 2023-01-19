The data are prepared using the script ml.cs.tsinghua.edu.cn/~shuyu/sml/20news.py. We have done the basic text preprocessing, including removing stop words and characters other than letters.

20news.vocab is the vocabulary, each line is a term, of the format
[id]  [word]    [freq]
where [id] is the word id, and [freq] is the term frequency.

20news.libsvm is the corpus, each line is the bag-of-words representation of a document, of the format
[docid]    [id]:[count] [id]:[count] ...
