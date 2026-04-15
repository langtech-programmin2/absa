# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 15:36:14 2023

@author: Elena Alafuzova + George Kordonis
"""
# load the libraries we need for the task
import os
import re
import nltk
import joblib
import numpy as np
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# list with stop words from NLTK library to clean the texts
stopwords_list = stopwords.words('english')


# function to extract info from reviews and store it into a list (text, target words, labels)
def extract_reviews_and_annotations(dataset_file, file_number):
    tree = ET.parse(dataset_file)
    root = tree.getroot()
    counts_review = 0
    counts_sentence = 0
    reviews = []
    for review in root.findall('Review'):
        review_sentences = []
        counts_review = counts_review + 1

        for sentence in review.findall('sentences/sentence'):
            sentence_id = sentence.get('id')
            sentence_text = sentence.find('text').text
            counts_sentence = counts_sentence + 1
            sentence_annotations = []
            opinions = sentence.find('Opinions')
            if opinions is not None:
                for opinion in opinions.findall('Opinion'):
                    target = opinion.get('target')
                    category = opinion.get('category')
                    category_split = category.split('#')
                    entity_label = category_split[0]
                    attribute_label = category_split[1] if len(category_split) > 1 else ""
                    sentiment = opinion.get('polarity')
                    sentence_annotations.append((target, entity_label, attribute_label, sentiment))

            review_sentences.append((file_number,sentence_id, sentence_text, sentence_annotations))

        reviews.append(review_sentences)
    

    return reviews

# function with which we can specify from which files we will extract info
def read_files_and_extract_reviews(file_count):
    folder_path = "./data"  # Replace with the actual folder path
    
    # Validate the file_count parameter
    if len(file_count) < 2 or len(file_count) > 10 :
        print("Invalid file count. Please provide a valid range of numbers between 1 and 10.")
        return
    
    for file_number in file_count:
        if file_number <1 or file_number >10:
            print("Invalid file number: {file_number}. Please provide a valid range of numbers between 1 and 10.")
            return
    
    file_count.sort()
    
    start_number = file_count[0]
    end_number = file_count[-1]
    
    final_reviews = []
    
    # Iterate over the files within the given range
    for i in range(start_number, end_number + 1):
        file_name = f"part_{i}.xml"
        file_path = os.path.join(folder_path, file_name)
        
        # Check if the file exists
        if os.path.isfile(file_path):
            reviews = extract_reviews_and_annotations(file_path,i)
            final_reviews.extend(reviews)
        else:
            print(f"File '{file_name}' does not exist.")
    return final_reviews


# function to process the extracted reviews and annotations and to prepare them for feature extraction stage
def data_preprocessing(final_reviews, stopwords_list):
    processed_sentences = []
    entity_labels = []
    attribute_labels = []
    sentiment_labels = []
    copied_list = []
    targets_per_file = []
    
    for review in final_reviews:    
        for file_number,sentence_id, sentence_text, sentence_annotations in review:
            if isinstance(sentence_text, str):
                # here we format the text 
                sentence_text = re.sub(r'[^\w\s]', ' ', sentence_text)
                sentence_text = re.sub(r"\s+", " ", sentence_text)
                sentence_text = sentence_text.lower()
                sentence_tokens = word_tokenize(sentence_text)
                filtered_tokens = [token for token in sentence_tokens if token not in stopwords_list] #[file_number] + 
    
                for target, entity_label, attribute_label, sentiment in sentence_annotations:
                    
                    if target:
                        nested_list=[[target]]
                    else:
                        nested_list = [[]]
                    # in case there more then one target in the sentence, we duplicate the sentence to be able to assign all corresponding target label to the sentences
                    for nested_item in nested_list:
                        copied_item = [sentence_id,filtered_tokens, nested_item,entity_label, attribute_label, sentiment]
                        copied_list.append(copied_item)
                    # we collect the formatted sentences and all the target words and labels corresponding to them    
                    targets_per_file.append(f"{file_number}_{sentence_id}_{entity_label}_{sentiment}")
                    entity_labels.append(entity_label)
                    attribute_labels.append(attribute_label)
                    sentiment_labels.append(sentiment)
                    processed_sentences.append(filtered_tokens)

    return processed_sentences, entity_labels, attribute_labels, sentiment_labels, targets_per_file

# this function is to count sentences per file, we needed it to correctly associate the sentences of the data with the splitted xml documents
def count_sentences_per_file(targets_per_file):
    # Initialize the count variable
    count_dict = {}
    
    # Loop through each item in the list
    for item in targets_per_file:
        
        match = re.match(r'^(\d+)_', item)
        if match:
            first_char = match.group(1)
            count_dict[first_char] = count_dict.get(first_char, 0) + 1
    
    return count_dict

# here we extract the features with which we will train our model
def feature_extraction_forML(processed_sentences, entity_labels, attribute_labels, sentiment_labels, targets_per_file,f_n_components,pca = False):
    # now we are going to create features with which we will train and test our model
    # we create lists for the values we extract
    pos_tags = []
    sentiment_sentence_score =[]
    
    # for each feature we write a method how to extract it and appent to the respective list
    for sentence in processed_sentences:    
        
        # POS tagging of each sentence with NLTK library
        sent_pos_tag = pos_tag(word_tokenize(' '.join(sentence)), tagset='universal')
        pos_tags.append(sent_pos_tag)

    
    #TF-IDF this will create a sparce matrix with tf-idf vectors for each sentence based on unigrams
    vectorizer1 = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 1),max_features=5000 ) # dimensionality (2,2) (3,3), (2,3) the latter for both bi- and trigrams
    tfidf1 = vectorizer1.fit_transform([' '.join(sentence) for sentence in processed_sentences])
    # bigrams
    vectorizer2 = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(2, 2),max_features=5000)
    tfidf2 = vectorizer2.fit_transform([' '.join(sentence) for sentence in processed_sentences])
    # trigrams
    vectorizer3 = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(3, 3),max_features=5000)
    tfidf3 = vectorizer3.fit_transform([' '.join(sentence) for sentence in processed_sentences])
    
    # Load AFINN-111 sentiment lexicon
    afinn_file = "AFINN-111.txt"
    sentiment_lexicon = {}
    with open(afinn_file, "r") as file:
        for line in file:
            word, score = line.strip().split("\t")
            sentiment_lexicon[word] = int(score)
    
    # Calculate sentiment features for each sentence, we will use overall sentiment score for each sentence
    for sentence in processed_sentences:
        positive_count = 0
        negative_count = 0
        overall_score = 0
        for word in sentence:
            if word in sentiment_lexicon:
                score = sentiment_lexicon[word]
                overall_score += score
                if score > 0:
                    positive_count += 1
                elif score < 0:
                    negative_count += 1
        sentiment_sentence_score.append(overall_score)

    # Making our POS tags numeric
    # take all POS tags and create a set of unique tags
    pos_tags_flat = [tag for sentence in pos_tags for _, tag in sentence]
    unique_tags = set(pos_tags_flat)
    
    # Assign numeric labels to the unique POS tags
    tag_to_label = {tag: label for label, tag in enumerate(sorted(unique_tags))}
    
    # Replace POS tags in sent_pos_tag with numeric labels
    numeric_pos_tags = []
    
    for sentence_pos_tags in pos_tags:
        numeric_sentence_tags = [tag_to_label[tag] for _, tag in sentence_pos_tags]
        numeric_pos_tags.append(numeric_sentence_tags)
    
    # create bigrams of POS tags
    num_pos_bigrams = []
    
    for tag_list in numeric_pos_tags:
        num_pos_bigram = list(zip(tag_list, tag_list[1:]))
        num_pos_bigrams.append(num_pos_bigram)
    
    
    # Verify the number of samples in each feature
    num_samples1 = tfidf1.shape[0]
    num_samples2 = tfidf2.shape[0]
    num_samples3 = tfidf3.shape[0]
    assert num_samples1 == num_samples2 == num_samples3 == len(num_pos_bigrams) == len(numeric_pos_tags) == len(sentiment_sentence_score), "Number of samples in features doesn't match."
    
    # Convert sentiment_scores to a sparse matrix
    sentiment_scores_sparse = csr_matrix(np.array(sentiment_sentence_score).reshape(-1, 1))
    
    # Convert num_pos_bigrams and numeric_pos_tags to sparse matrices
    max_bigram_count = max(len(bigrams) for bigrams in num_pos_bigrams)
    max_tag_count = max(len(tags) for tags in numeric_pos_tags)
    
    num_pos_bigrams_sparse = csr_matrix((num_samples1, max_bigram_count), dtype=int)
    numeric_pos_tags_sparse = csr_matrix((num_samples1, max_tag_count), dtype=int)
    
    for i, bigrams in enumerate(num_pos_bigrams):
        num_pos_bigrams_sparse[i, :len(bigrams)] = 1  # Assigning a value of 1 for each bigram present
    
    for i, tags in enumerate(numeric_pos_tags):
        numeric_pos_tags_sparse[i, :len(tags)] = tags
    

    # encode the values of entity and attribute labels into numeric form
    entity_labels_mapping = {
    'RESTAURANT': 0,
    'FOOD': 1,
    'DRINKS': 2,
    'AMBIENCE': 3,
    'SERVICE': 4,
    'LOCATION': 5
    }

    attribute_labels_mapping = {
    'GENERAL': 0,
    'PRICES': 1,
    'QUALITY': 2,
    'STYLE_OPTIONS': 3,
    'MISCELLANEOUS': 4
    }

    # associate categorical values of entity labels with the numbers form above dictionaries
    num_entity_labels = [entity_labels_mapping[label] for label in entity_labels]
    # adjust them for further combining with other features
    entity_labels_flat = csr_matrix(num_entity_labels).reshape((-1, 1))
    
    #the same for attribute labels
    num_attribute_labels = [attribute_labels_mapping[label] for label in attribute_labels]
    attribute_labels_flat = csr_matrix(num_attribute_labels).reshape((-1, 1))
    
    # Combine all our features using hstack
    combined_features = hstack((tfidf1, tfidf2, tfidf3, num_pos_bigrams_sparse, numeric_pos_tags_sparse, sentiment_scores_sparse, entity_labels_flat, attribute_labels_flat))

    # when we will reduce dimentionality with PCA method we will need to convert sparse matrix to dense    
    # when pca method is used we reduce dimentionality (we do it here, before we splitt the matrix associating each piece with a specific part of xml files containig 35 reviews)
    if pca:
        ar_combined_features = combined_features.toarray()
        pca = PCA(n_components=f_n_components)
        reduced_features = pca.fit_transform(ar_combined_features)
        ar_combined_features = reduced_features 
    else:
        ar_combined_features = combined_features.toarray()
    
    # here is an array with out terget variables
    ar_sentiment_labels = np.array(sentiment_labels)
    
    # Dictionary with counts of target values in each file
    count_dict = count_sentences_per_file(targets_per_file)
    
    # Split the arrays based on counts. Each entry here correlates with the 10 parts of our data
    split_combined_features = {}
    split_sentiment_labels = {}
    
    for key, count in count_dict.items():
        file_number = key.split('_')[0]
        split_combined_features[file_number] = ar_combined_features[:count]
        split_sentiment_labels[file_number] = ar_sentiment_labels[:count]
        ar_combined_features = ar_combined_features[count:]
        ar_sentiment_labels = ar_sentiment_labels[count:]
    

    return combined_features,ar_combined_features,split_combined_features, split_sentiment_labels


# dictionaries to associate our target values with their numeric representations
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

# function to train our model
def train_models(split_features_labels,split_sentiment_labels,files_to_be_used): 
    
    # Create lists to store feature and sentiment arrays.
    f_array_list = []
    s_array_list = []
    
    # Iterate over the file numbers, each number signifies a specific part of our data
    for key in files_to_be_used:
        # Extract arrays from the dictionary using the variable names
        f_array_value = split_features_labels[str(key)]
        s_array_value = split_sentiment_labels[str(key)]
        # Append arrays to the list
        f_array_list.append(f_array_value)
        s_array_list.append(s_array_value)
    
    
    # Concatenate arrays into a single array
    combined_features = np.concatenate(f_array_list)
    combined_labels = np.concatenate(s_array_list)

        
    # Convert labels to numeric values
    labels = combined_labels.tolist()
    labels_encoded = [map_sentiment_to_numeric(label) for label in labels] # associate labels with values from the dictionary
    
    # Initialize the models
    logistic_regression_model = LogisticRegression(max_iter=1000)
    
    # Concatenate arrays into a single array
    combined_features = np.concatenate(f_array_list)
   
    # Train the models
    logistic_regression_model.fit(combined_features, labels_encoded)

    # Save the trained models
    joblib.dump(logistic_regression_model, 'logistic_regression_model.pkl')
   
    return logistic_regression_model









