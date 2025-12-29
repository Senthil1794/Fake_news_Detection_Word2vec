# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 10:04:56 2025

@author: sea
"""

import sqlite3
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from xgboost import XGBClassifier

stopwords_list = stopwords.words("english")
exclude_punctuations = string.punctuation

# fetch data from data base
def fetch_data_from_db(database, table):
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    '''
    cursor.execute("Select name from sqlite_master where type='table';")
    tables = cursor.fetchall()
    print(tables)
    '''
    query = "select * from " + table
    data = pd.read_sql_query(query, conn)
    return data


def remove_punctuation(text):
    return text.translate(str.maketrans("","",exclude_punctuations))

def stop_words_removal(text):
    data = [word for word in text.split() if word not in stopwords_list]
    return " ".join(data)

stemmer = PorterStemmer()
def apply_stemming(text):
    data = [stemmer.stem(word) for word in text.split()]
    return " ".join(data)
lemmatizer = WordNetLemmatizer()
def apply_lemmatizer(text):
    data = [lemmatizer.lemmatize(word) for word in text.split()]
    return " ".join(data)
def text_data_processing(data):
    data["text_original"] = data["text"].copy()
    data["text"] = data["text"].str.lower()
    data["text"] = data["text"].apply(remove_punctuation)
    data["text"] = data["text"].apply(stop_words_removal)
    #data["text"] = data["text"].apply(apply_stemming)
    data["text"] = data["text"].apply(apply_lemmatizer)
    return data
    
def tf_idf_features_fit(df):
    tfidf = TfidfVectorizer(min_df=0.01, max_df=0.1)
    tfidf_matrix = tfidf.fit_transform(df["text"])
    return tfidf, tfidf_matrix.toarray()

def tf_idf_features_transform(tfidf, df):
    tfidf_matrix = tfidf.transform(df["text"])
    return tfidf_matrix.toarray()

def train_test_split(data_train,data_test,data_train_matrix,data_test_matrix):
    y_train= data_train['label'] 
    y_test = data_test['label']
    x_train = data_train_matrix.copy()
    x_test = data_test_matrix.copy()
    return x_train,x_test,y_train,y_test

def fit_and_evaluate_model(x_train, x_test, y_train, y_test):
    xgb =  XGBClassifier(random_state=0) 
    xgb.fit(x_train, y_train)
    xgb_predict = xgb.predict(x_test)
    xgb_conf_matrix = confusion_matrix(y_test, xgb_predict)
    xgb_acc_score = accuracy_score(y_test, xgb_predict)
    print("confussion matrix")
    print(xgb_conf_matrix)
    print("\n")
    print("Accuracy of XGBoost:",xgb_acc_score*100,'\n')
    print(classification_report(y_test,xgb_predict))
    return xgb

def get_important_features(model, features):
    importances = pd.DataFrame(model.feature_importances_)
    print(importances)
    
    importances['features'] = features
    importances.columns = ['importance','feature']
    importances.sort_values(by = 'importance', ascending= False,inplace=True)

    return importances


    