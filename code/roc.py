#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 19:30:05 2018

@author: carsonluuu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc

df = pd.read_csv("sROC.csv")

df['grams'] = df.grams.apply(lambda x: x[1:-1].split(' '))
#df['grams'] = df['grams'].str[1:-1].split()


X = df.iloc[:, :-2]
y1 = df.iloc[:, 1]
y2 = df.iloc[:, 2]

from sklearn.cross_validation import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y1, test_size = 0.5, random_state = 0)
X2_train, X2_test, y2_train, y2_test = train_test_split(X, y2, test_size = 0.5, random_state = 0)



logistic_clf = Pipeline([('vect', CountVectorizer(min_df=5, max_df=0.99)),
                         ('tfidf', TfidfTransformer())])


logistic_clf = logistic_clf.fit(X1_train, y1_train)

predicted_prob = hard_svc_clf.predict_proba(test_data)[:, 1]
predicted = hard_svc_clf.predict(test_data)


clf = LogisticRegression()

clf = clf.fit(X_svd, train_target)
predicted_prob = clf.predict_proba(y_svd)[:, 1]
predicted = clf.predict(y_svd)

fpr, tpr, thresholds = roc_curve(test_target, predicted_prob)
auroc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (auroc = %0.3f)' % auroc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve--Logistic Regression using LSI')
plt.legend(loc="lower right")
plt.show()