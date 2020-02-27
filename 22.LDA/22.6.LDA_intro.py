# !/usr/bin/python
# -*- coding:utf-8 -*-

from gensim import corpora, models, similarities
from pprint import pprint
import warnings

# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


if __name__ == '__main__':
    f = open('/Users/liulebin/Documents/codeing/codeingForSelfStudy/ML-Basic-Theory-Study/ML_Learning_code/22.LDA/LDA_test.txt')
    stop_list = set('for a of the and to in'.split())
    # texts = [line.strip().split() for line in f]
    # print 'Before'
    # pprint(texts)
    print('After')   # 空格或者tab键分开
    texts = [[word for word in line.strip().lower().split() if word not in stop_list] for line in f]
    print('Text = ')
    pprint(texts)

    dictionary = corpora.Dictionary(texts)
    print("-------------dictionary------------")
    print(dictionary) # gensim 提供好的数据机构
    V = len(dictionary)
    #
    corpus = [dictionary.doc2bow(text) for text in texts]
    """Convert `document` into the bag-of-words (BoW) format = list of `(token_id, token_count)` tuples."""
    print("-------------corpus------------")
    print(corpus)
    corpus_tfidf = models.TfidfModel(corpus)[corpus]
    # corpus_tfidf = corpus

    print('TF-IDF:')#获得每一篇文档的tfidf
    for c in corpus_tfidf:
        print(c)

    print('\nLSI Model:')
    lsi = models.LsiModel(corpus_tfidf, num_topics=2, id2word=dictionary)# id2word=dictionary 参数在显示的时候可以显示词，而非编号

    topic_result = [topic for topic in lsi[corpus_tfidf]]
    print("-------------topic_result------------")
    pprint(topic_result)
    """
      -------------topic_result------------
      [[(0, 0.34057117986842017), (1, -0.20602251622679588)],
       [(0, 0.6933040002171554), (1, 0.007232758390390146)],
       [(0, 0.5902607670389728), (1, -0.3526046949085571)],
       [(0, 0.521490182182514), (1, -0.33887976154055344)],
       [(0, 0.39533193176354425), (1, -0.05919285336659805)],
       [(0, 0.03635317352849333), (1, 0.1814655020881899)],
       [(0, 0.14709012328778945), (1, 0.4943294812782231)],
       [(0, 0.2140711731756532), (1, 0.640645666445394)],
       [(0, 0.4006656831817071), (1, 0.6413108299094004)]]
       
       表示第0篇文档的   谈第0号主题的相似度是   0.34057117986842017       谈第1号主题的相似度是  -0.20602251622679588  这里为负数是lsi不太让人接受的地方，他只做矩阵的分解。 可能做非负矩阵分解      
       表示第1篇文档的   谈第1号主题的概率是   0.34057117986842017

    """
    print('LSI Topics:')
    pprint(lsi.print_topics(num_topics=2, num_words=5)) # 打印两个主题的前5个词，及主题下每个词各自的相关的概率值
    similarity = similarities.MatrixSimilarity(lsi[corpus_tfidf])   # similarities.Similarity()
    #                   喂给训练好的LSI模型语料corpus_tfidf,获得每个文档的主题分布，以主题分布作为文档表示向量，可以求相似度；任何两个文档之间的相似性
    print('-------- LSI Topics  Similarity:--------')
    pprint(list(similarity))

    print('\nLDA Model:') #corpus_tfidf -> corpus是lda理论应输入的数据
    num_topics = 2
    lda = models.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary,
                          alpha='auto', eta='auto', minimum_probability=0.001, passes=10)
    #eta就是词分布的超参数。  minimum_probability=0.001 表示当概率小于多少时候就显示为0； passes=10表示要对语料运行多少回；
    doc_topic = [doc_t for doc_t in lda[corpus_tfidf]]
    print('--------LDA  Document-Topic:\n')
    pprint(doc_topic)
    """
    --------LDA  Document-Topic:
       [[(0, 0.6457475), (1, 0.35425252)],
        [(0, 0.2905261), (1, 0.70947385)],
        [(0, 0.6261517), (1, 0.3738484)],
        [(0, 0.6423126), (1, 0.35768735)],
        [(0, 0.24508792), (1, 0.75491214)],
        [(0, 0.2468656), (1, 0.75313437)],
        这个值是按照概率理论做的所以一定是非负的。它不是矩阵分解做的(LSI)
        数据上并没有完全收敛
    """

    #打印每一个文档的主题分布
    for doc_topic in lda.get_document_topics(corpus_tfidf):
        print("-------- LDA doc_topic --------")
        print(doc_topic)
     #
    for topic_id in range(num_topics):
        print('Topic', topic_id)
        # pprint(lda.get_topic_terms(topicid=topic_id))
        #打印每一个topic下对应的词的分布
        pprint(lda.show_topic(topic_id))


    #打印相似度    数据上看，我们的参数或者做法是有问题的，因为相似度都接近1.0
    similarity = similarities.MatrixSimilarity(lda[corpus_tfidf])
    print('Similarity:')
    pprint(list(similarity))


    # LDA做结构化处理，
    hda = models.HdpModel(corpus_tfidf, id2word=dictionary)
    topic_result = [a for a in hda[corpus_tfidf]]
    print('\n\nUSE WITH CARE--\nHDA Model:')
    pprint(topic_result)
    print('HDA Topics:')
    print(hda.print_topics(num_topics=2, num_words=5))
