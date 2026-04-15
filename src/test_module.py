# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 15:37:16 2023

@author: Elena Alafuzova + George Kordonis
"""

# import os
# import re
import nltk
import joblib
import numpy as np

# import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.tag import pos_tag
# from sklearn.feature_extraction.text import TfidfVectorizer
# from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import accuracy_score
# from sklearn.feature_selection import SelectKBest, chi2
# from sklearn.linear_model import LogisticRegression
# from train import extract_reviews_and_annotations

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('maxent_ne_chunker')
nltk.download('words')


stopwords_list = stopwords.words('english')

# the dictionaries to encode our output values
def map_sentiment_to_numeric(sentiment):
    if sentiment == 'positive':
        return 1
    elif sentiment == 'negative':
        return -1
    else:  # neutral
        return 0

def map_numeric_to_sentiment(numeric_label):
    if numeric_label == 1:
        return 'positive'
    elif numeric_label == -1:
        return 'negative'
    else:  # 0
        return 'neutral'

# function which run the model on the part specified for testing
def test_models(split_features_labels, split_sentiment_labels,file_to_be_used):   
  # this is like in training 
  # Create lists to store the arrays
  f_array_list = []
  s_array_list = []

  # Extract arrays from the dictionary using the variable names
  f_array_value = split_features_labels[str(file_to_be_used)]
  s_array_value = split_sentiment_labels[str(file_to_be_used)]
  # Append arrays to the list
  f_array_list.append(f_array_value)
  s_array_list.append(s_array_value)
  
  # Concatenate arrays into a single array
  combined_features = np.concatenate(f_array_list)
  combined_labels = np.concatenate(s_array_list)

  # Convert labels to numeric values
  labels = combined_labels.tolist()
  labels_encoded = [map_sentiment_to_numeric(label) for label in labels]
     
  # Load the models from pickle files
  logistic_regression_model = joblib.load('logistic_regression_model.pkl')
  
  # Test the models
  logistic_regression_predictions = logistic_regression_model.predict(combined_features)
  #svm_predictions = svm_model.predict(features_dense)
    
  # Calculate accuracy
  logistic_regression_accuracy = accuracy_score(labels_encoded, logistic_regression_predictions)

  return logistic_regression_predictions,logistic_regression_accuracy




