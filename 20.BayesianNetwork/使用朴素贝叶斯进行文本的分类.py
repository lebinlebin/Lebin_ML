"""
【机器学习实验】使用朴素贝叶斯进行文本的分类
2. 数据集
该实验的数据可以通过scikit-learn获取一组新闻信息。 
数据集由19,000个新闻信息组成，其中包含了20个不同的主题，包含政治、体育、科学等内容。 
该数据集可以分成训练和测试两部分，训练和测试数据的划分基于某个特定日期。
数据的加载有两种方式：
1.sklearn.datasets.fetch_20newsgroups，该函数返回一个原数据列表，可以将它作为文本特征提取的接口(sklearn.feature_extraction.text.CountVectorizer)的输入
2.sklearn.datasets.fetch_20newsgroups_vectorized，该接口直接返回直接可以使用的特征，可以不再使用特征提取了
"""
from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='all')
print (news.keys())
print (type(news.data), type(news.target), type(news.target_names))
print (news.target_names)
print (len(news.data))
print (len(news.target))
print (news.data[0])
print (news.target[0], news.target_names[news.target[0]])
"""
打印的新闻内容,类别为10，类别名为rec.sport.hockey。
"""
"""
划分训练与测试数据
在进行转换工作之前，我们需要将数据划分为训练和测试数据集。
由于载入的数据是随机顺序出现的，我们可以将数据划分为两部分，75%作为训练数据，25%作为测试数据：
"""
SPLIT_PERC = 0.75
split_size = int(len(news.data)*SPLIT_PERC)
X_train = news.data[:split_size]
X_test = news.data[split_size:]
Y_train = news.target[:split_size]
Y_test = news.target[split_size:]
"""
因为sklearn.datasets.fetch_20newsgroups本身可以根据subset参数来选择训练数据和测试数据，
这里训练数据有11,314条，占总数据集的60%，测试数据集占40%。可以通过如下方式得到：
"""
news_train = fetch_20newsgroups(subset='train')
news_test = fetch_20newsgroups(subset='test')
X_train = news_train.data
X_test = news_test.data
Y_train = news_train.target
Y_test = news_test.target

"""
scikit-learn提供了一些实用工具可以用最常见的方式从文本内容中抽取数值特征，比如说：
标记（tokenizing）文本以及为每一个可能的标记(token)分配的一个整型ID，例如用空格和标点符号作为标记的分割符（中文的话涉及到分词的问题）
计数（counting）标记(token)在每个文本中的出现频率
在大多数样本/文档中都出现的标记的重要性递减过程中，进行标准化(normalizing)和加权(weighting) 

将每个独立的标记(token)的出现频率（不管是否标准化）看做是特征 
给定一个文档的所有标记的频率构成向量看做是一个多变量的样本 
这样一个文本的语料库就可以表征为一个矩阵，其中每一行代表了一个文档，而每一列代表了在该语料库中出现的标记词。
文本可以用词语的出现频率表征，这样可以完全忽略词在文本中的相对位置信息，这一点应该就保证了贝叶斯的条件独立性。
"""
"""
稀疏性
大多数文档通常只会使用语料库中所有词的一个子集，因而产生的矩阵将有许多特征值是0（通常99%以上都是0）。 
例如，一组10,000个短文本（比如email）会使用100,000的词汇总量，而每个文档会使用100到1,000个唯一的词。 
为了能够在内存中存储这个矩阵，同时也提供矩阵/向量代数运算的速度，通常会使用稀疏表征例如在scipy.sparse包中提供的表征。
"""
"""
文本特征提取的接口
sklearn.feature_extraction.text提供了以下构建特征向量的工具：
feature_extraction.text.CountVectorizer([…]) Convert a collection of text documents to a matrix of token counts
feature_extraction.text.HashingVectorizer([…]) Convert a collection of text documents to a matrix of token occurrences
feature_extraction.text.TfidfTransformer([…]) Transform a count matrix to a normalized tf or tf-idf representation
feature_extraction.text.TfidfVectorizer([…]) Convert a collection of raw documents to a matrix of TF-IDF features.
解释：
CountVectorizer方法构建单词的字典，每个单词实例被转换为特征向量的一个数值特征，每个元素是特定单词在文本中出现的次数
HashingVectorizer方法实现了一个哈希函数，将标记映射为特征的索引，其特征的计算同CountVectorizer方法
TfidfVectorizer使用了一个高级的计算方法，称为Term Frequency Inverse Document 
Frequency (TF-IDF)。这是一个衡量一个词在文本或语料中重要性的统计方法。直觉上讲，该方法通过比较在整个语料库的词的频率，
寻求在当前文档中频率较高的词。这是一种将结果进行标准化的方法，
可以避免因为有些词出现太过频繁而对一个实例的特征化作用不大的情况(我猜测比如a和and在英语中出现的频率比较高，
但是它们对于表征一个文本的作用没有什么作用)
"""
"""
构建朴素贝叶斯分类器
由于我们使用词的出现次数作为特征，可以用多项分布来描述这一特征。
在sklearn中使用sklearn.naive_bayes模块的MultinomialNB类来构建分类器。 
我们使用Pipeline这个类来构建包含量化器(vectorizers)和分类器的复合分类器(compound classifer)。
"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.naive_bayes import GaussianNB
#nbc means naive bayes classifier
nbc_1 = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB()),#Naive Bayes classifier for multinomial models
])
nbc_2 = Pipeline([
    ('vect', HashingVectorizer()),#贝叶斯估计不允许输入为负值non_negative=True
    ('clf', GaussianNB()),#Gaussian Naive Bayes (GaussianNB)
])
nbc_3 = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', MultinomialNB()),
])

nbcs = [nbc_1, nbc_2, nbc_3]

"""
交叉验证
我们下面设计一个对分类器的性能进行测试的交叉验证的函数：
"""
# from sklearn.cross_validation import cross_val_score, KFold
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem
import numpy as np

def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold croos validation iterator of k=5 folds
    # cv = KFold(len(y), K, shuffle=True, random_state=0)
    cv = KFold(K, shuffle=True, random_state=0)

    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(scores)
    print(np.mean(scores), sem(scores))
"""
将训练数据分成5份，输出验证的分数：
"""
for nbc in nbcs:
    evaluate_cross_validation(nbc, X_train, Y_train, 5)
"""
[ 0.82589483 0.83473266 0.8272205 0.84136103 0.83377542] 
Mean score: 0.833 (+/-0.003) 
[ 0.76358816 0.72337605 0.72293416 0.74370305 0.74977896] 
Mean score: 0.741 (+/-0.008) 
[ 0.84975696 0.83517455 0.82545294 0.83870968 0.84615385] 
Mean score: 0.839 (+/-0.004)
从上面的结果看出，CountVectorizer和TfidfVectorizer进行特征提取的方法要比HashingVectorizer的效果好。
"""

"""
优化提取单词规则参数 token_pattern
TfidfVectorizer的一个参数token_pattern用于指定提取单词的规则。 
默认的正则表达式是r"\b\w\w+\b"，这个正则表达式只匹配单词边界并考虑到了下划线，也可能考虑到了横杠和点。 
新的正则表达式是r"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b"。
"""

nbc_4 = Pipeline([
    ('vect', TfidfVectorizer(
                token_pattern=r"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b",
    )),
    ('clf', MultinomialNB()),
])
evaluate_cross_validation(nbc_4, X_train, Y_train, 5)
"""
这个分数已经比之前的0.839提高了一些了。
"""

"""
优化省略词(stopwords)参数
TfidfVectorizer的一个参数stop_words这个参数指定的词将被省略不计入到标记词的列表中，
比如一些出现频率很高的词，但是这些词对于特定的主题不能提供任何的先验支持。
"""
def get_stop_words():
    result = set()
    for line in open('/Users/liulebin/Documents/codeing/codeingForSelfStudy/ML-Basic-Theory-Study/ML_Learning_code/20.BayesianNetwork/stopword.txt', 'rb+').readlines():
        print(line)
        result.add(line.strip())
    return result


stop_words = get_stop_words()
nbc_5 = Pipeline([
    ('vect', TfidfVectorizer(
                stop_words=stop_words,
                token_pattern=r"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b",
    )),
    ('clf', MultinomialNB()),
])
evaluate_cross_validation(nbc_5, X_train, Y_train, 5)

"""
[ 0.88731772 0.88731772 0.878038 0.88466637 0.88107869] 
Mean score: 0.884 (+/-0.002)
分数又提升到了0.884。
"""
"""
优化贝叶斯分类器的alpha参数
MultinomialNB有一个alpha参数，该参数是一个平滑参数，默认是1.0，我们将其设为0.01。
"""
nbc_6 = Pipeline([
    ('vect', TfidfVectorizer(
                stop_words=stop_words,
                token_pattern=r"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b",
    )),
    ('clf', MultinomialNB(alpha=0.01)),
])
evaluate_cross_validation(nbc_6, X_train, Y_train, 5)

"""
[ 0.91073796 0.92532037 0.91604065 0.91294741 0.91202476] 
Mean score: 0.915 (+/-0.003)
这下分数已经优化的很好了。
"""
"""
评估分类器性能
我们通过交叉验证得到了效果比较好的分类器参数，下面我们可以用该分类器来测试我们的测试数据了。
"""
from sklearn import metrics
nbc_6.fit(X_train, Y_train)
print ("Accuracy on training set:")
print (nbc_6.score(X_train, Y_train))
print ("Accuracy on testing set:")
print (nbc_6.score(X_test,Y_test))
y_predict = nbc_6.predict(X_test)
print ("Classification Report:")
print (metrics.classification_report(Y_test,y_predict))
print ("Confusion Matrix:")
print (metrics.confusion_matrix(Y_test,y_predict))

"""
这里只输出准确率:
Accuracy on training set: 
0.997701962171 
Accuracy on testing set: 
0.846919808816
"""