from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn import svm
import nltk
import itertools
from numpy import array
import numpy as np
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from sklearn.model_selection import learning_curve


class Baseline(object):
    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2

        # self.model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=10000) #E:0.82 S:0.77 353.74461698532104
        # self.model = ExtraTreesClassifier(n_estimators=30) #E:0.82 S:0.76 24.421394109725952
        # self.model = RandomForestClassifier(n_estimators=30) # E:0.82 S:0.77 24.714120864868164
        # self.model = DecisionTreeClassifier(max_depth=None) #E:0.82 S:0.76 26.83554482460022
        # self.model = svm.SVC(gamma=100, kernel='rbf')# rbf E:0.82 S:0.76 50.560959815979004
        # self.model = svm.LinearSVC()  #E:0.65 S:0.70 28.32796597480774
        # self.model = svm.SVC(decision_function_shape='ovo')# E:0.77 S:0.71 62.07681107521057
        # self.model = svm.NuSVC()  #E:0.77 S:0.72 84.16532301902771
        # self.model = svm.SVC(gamma=10, kernel='rbf', decision_function_shape='ovo')  # E:0.82 S:0.76 47.88548707962036
        self.model = svm.SVC(gamma=100, kernel='rbf',decision_function_shape='ovr')# E:0.82 S:0.76 45.822433948516846

    def extract_features(self, word):
        len_tokens = len(word.split(' '))
        len_chars = len(word) / self.avg_word_length
        strl_ist = word.split(',')
        pos = nltk.pos_tag(strl_ist)
        for i in range(len(pos)):
            a = pos[i][1] # the pos of the word
        v = array(list(a)) # one hot encoding
        label_encoder = LabelEncoder()
        data_encoded = label_encoder.fit_transform(v)# transfrom the string to number
        pos_f = float(list(data_encoded)[0]) # the feature should use the type float
        return [len_tokens,len_chars,pos_f]

    def tf_feature(self,trainset,testset):
        word = []
        for sent in trainset+testset:
            sen = sent['sentence'].split()
            for i in sen:
                word.append(i)

        tf = nltk.FreqDist(word)
        return tf

    def train(self,trainset,testset):
        X = []
        y = []
        term_f = self.tf_feature(trainset,testset)
        for sent in trainset+testset:
            x=self.extract_features(sent['target_word'])
            x.append(term_f[sent['target_word']])
            X.append(x)
            y.append(sent['gold_label'])

        self.model.fit(X, y)
        # print(X)
############################################plot learning Curves###################################################
        # plt.figure()
        # title = "Learning Curves (SVM.NuSVC)"
        # plt.title(title)
        # plt.xlabel("Training examples")
        # plt.ylabel("Score")
        # estimator = self.model
        # train_sizes = np.linspace(.1, 1.0,5)
        # train_sizes, train_scores,test_scores= learning_curve(estimator, X, y, n_jobs=1, train_sizes=train_sizes)
        # train_scores_mean = np.mean(train_scores, axis=1)
        # train_scores_std = np.std(train_scores, axis=1)
        # plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="r")
        # plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
        #
        # plt.show()
###########################################################################################################################
    def test(self, trainset,testset):
        X = []
        term_f = self.tf_feature(trainset,testset)
        for sent in testset:
            x= self.extract_features(sent['target_word'])
            x.append(term_f[sent['target_word']])
            X.append(x)

        # for sent in testset:
        #     X.append(self.extract_features(sent['target_word']))
        return self.model.predict(X)
