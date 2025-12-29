# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 16:01:49 2025

@author: sea
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from gensim.models import Word2Vec,FastText

from data_processing_and_features import(
    fetch_data_from_db,
    text_data_processing,
    fit_and_evaluate_model,
    get_important_features
    )
data = fetch_data_from_db("Classification.db", "fake_news_classification")

# Drop unnecessary columns
data.drop(["Unnamed: 0.1", "Unnamed: 0"],axis=1,inplace=True)
# Drop entire row which has missinig feilds
data = data.dropna(subset=["text", "label"])

print(data.info())

data = text_data_processing(data)
print(data["label"].value_counts())

data["tokens"] = data["text"].apply(lambda x: x.split())
model = Word2Vec(sentences=data['tokens'], vector_size=300, window=5, min_count=5,sg = 1)
print(model.wv["information"])
print(model.wv.most_similar('information', topn=5))
data_train = data[0:55031]
data_test = data[55031:70754]

print("+++")

features = model.get_feature_names_out()

def average_word_vector(tokens):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    else:
        return sum(vectors)/len(vectors)
data_train["vectors"] = data_train['tokens'].apply(average_word_vector)
data_test["vectors"] = data_test["tokens"].apply(average_word_vector)


x_train = data_train["vectors"].tolist()
x_test = data_test["vectors"].tolist()
y_train = data_train["label"]
y_test = data_test["label"]

model = fit_and_evaluate_model(x_train, x_test, y_train, y_test)

feature_importance = get_important_features(model, features)
print(feature_importance.head(10))


