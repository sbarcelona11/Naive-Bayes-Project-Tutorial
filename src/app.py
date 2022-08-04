import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews_dataset.csv')
df_raw

df_raw.info()
df_raw.describe(include='all')
df_raw.shape
df_raw.describe()
df_raw.sample(20)

df_raw['polarity'].value_counts()

df_copy = df_raw.copy()
df_transf = df_copy.drop('package_name', axis=1)

df_transf['review'] = df_transf['review'].str.strip()
df_transf['review'] = df_transf['review'].str.lower()
df_transf

df = df_transf.copy()
X = df['review']
y = df['polarity']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

clf_CV = Pipeline([('cont_vect', CountVectorizer()), ('clf', MultinomialNB())])
clf_CV.fit(X_train, y_train)
pred = clf_CV.predict(X_test)

clf_TF = Pipeline([('tfidf_vect', TfidfVectorizer()), ('clf', MultinomialNB())])
clf_TF.fit(X_train, y_train)
pred_1 = clf_TF.predict(X_test)

clf_CVTF = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
clf_CVTF.fit(X_train, y_train)
pred_2 = clf_CVTF.predict(X_test)

print('CountVectorizer')
print(classification_report(y_test, pred))
print('')

print('TfidfVectorizer')
print(classification_report(y_test, pred_1))
print('')

print('CountVectorizer and TfidfTransformer')
print(classification_report(y_test, pred_2))
print('')

print(f'clf_CV Accuracy = {metrics.accuracy_score(y_test,pred)}')
print(f'clf_TF Accuracy = {metrics.accuracy_score(y_test,pred_1)}')
print(f'clf_CVTF Accuracy = {metrics.accuracy_score(y_test,pred_2)}')

# hiperparameters

parameters = {'cont_vect__ngram_range': [(1, 1), (1, 2)], 'clf__alpha': (1e-2, 1e-3)}
clf_1 = RandomizedSearchCV(clf_CV, parameters, n_iter=5)
clf_1.fit(X_train, y_train)
pred_1 = clf_1.predict(X_test)
print(f'Prediction is: {pred_1}')
print(f'Best params: {clf_1.best_params_}')

parameters = {'clf__alpha': (1e-2, 1e-3)}
clf_2 = RandomizedSearchCV(clf_TF, parameters, n_iter=5)
clf_2.fit(X_train, y_train)
pred_2 = clf_2.predict(X_test)
print(f'Prediction is: {pred_2}')
print(f'Best params: {clf_2.best_params_}')

parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}
clf_3 = RandomizedSearchCV(clf_CVTF, parameters, n_iter=5)
clf_3.fit(X_train, y_train)
pred_3 = clf_3.predict(X_test)
print(f'Prediction is: {pred}')
print(f'Best params: {clf_3.best_params_}')

print('Report 1')
print(classification_report(y_test, pred_1))
print('Report 2')
print(classification_report(y_test, pred_2))
print('Report 3')
print(classification_report(y_test, pred_3))

best_model = clf_3.best_estimator_
pickle.dump(best_model, open('../models/best_model.pickle', 'wb'))



