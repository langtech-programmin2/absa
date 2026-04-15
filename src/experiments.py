# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 19:20:26 2023

@author: Elena Alafuzova + George Kordonis
"""
# import numpy as np
from nltk.corpus import stopwords
# from sklearn.model_selection import KFold
# from sklearn.metrics import accuracy_score
from test_module import test_models
from train import train_models, extract_reviews_and_annotations, read_files_and_extract_reviews,data_preprocessing,feature_extraction_forML,count_sentences_per_file


stopwords_list = stopwords.words('english')


def load_data(test_file_used,f_n_components,pca):
    #load data
    file_list = [1,2,3,4,5,6,7,8,9,10]
    # Load the XML files and extract features and labels for training and testing
    final_reviews = read_files_and_extract_reviews(file_list)

    # Extract texts and labels from the loaded data
    combined_features,entity_labels, attribute_labels, sentiment_labels,targets_per_file = data_preprocessing(final_reviews, stopwords_list)
    # Extract features and  labels to run on ML model
    combined_features,df_combined_features,split_features_labels,split_sentiment_labels = feature_extraction_forML(combined_features,entity_labels, attribute_labels,sentiment_labels,targets_per_file,f_n_components,pca)
    # dictionary to split the total amount of sentences into 10 parts correcponding to xml files
    count_d=count_sentences_per_file(targets_per_file)

    return split_features_labels,split_sentiment_labels,count_d



def run_cross_validation(f_n_components=0,pca = False):

     fold_accuracies = []
    
     for iteration in range(1, 11):

         train_file_list = [1,2,3,4,5,6,7,8,9,10]
         train_file_list.remove(iteration) # we keep one file for test

         split_features_labels,split_sentiment_labels,count_d = load_data(iteration,f_n_components,pca)
         train_models(split_features_labels,split_sentiment_labels,train_file_list)
                    
         y_pred,accuracy = test_models(split_features_labels,split_sentiment_labels,iteration)
                     
         fold_accuracies.append(accuracy)
     # calculate average accuracy of 10 folds    
     average_accuracy = sum(fold_accuracies) / len(fold_accuracies)
             
     return fold_accuracies,average_accuracy
 
# this function is to reduce dimentions with PCA method    
def test_feature_selection_technique():
    reduced_dimentionality = [2,500,2500] # we try to reduce to these numbers of features
    red_dim_accuracies = []
    red_dim_avg_accuracies = []
    for k in reduced_dimentionality:
        fold_accuracies,average_accuracy = run_cross_validation(k, True)
        red_dim_accuracies.append(fold_accuracies)
        red_dim_avg_accuracies.append(average_accuracy)

    return  red_dim_accuracies, red_dim_avg_accuracies 

def main():
      
    # Perform cross-validation on the dataset with all features selected in function feature_extraction_forML in test.py
    # We can manually select the features while defining combined_features variable
    accuracies,average_accuracy = run_cross_validation()
    print("The accuracies are: ",accuracies)
    print("The average accuracy is: ",average_accuracy)
    
    # perform cross-validation on the data set after applying PCA to it
    reduced_features_accuracies,reduced_features_average_accuracy =test_feature_selection_technique()
    print("The accuracies with reduced features are: ",reduced_features_accuracies)
    print("The average accuracy with reduced features is: ",reduced_features_average_accuracy)


if __name__ == "__main__":
    main()
