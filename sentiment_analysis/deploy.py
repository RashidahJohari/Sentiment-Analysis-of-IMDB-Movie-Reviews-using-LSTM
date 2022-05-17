# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:21:23 2022

@author: Acer
"""

from tensorflow.keras.models import load_model
import os
import json
from sentiment_analysis_modules import ExploratoryDataAnalysis
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np

#%% Constant / Static here
MODEL_PATH = os.path.join(os.getcwd(),'model.h5')
JSON_PATH = os.path.join(os.getcwd(),'tokenizer_data.json')

#%% Model Loading
sentiment_classifier = load_model(MODEL_PATH)
sentiment_classifier.summary()

#%% Tokenizer Loading
with open(JSON_PATH,'r') as json_file:
    token = json.load(json_file)
    
#%% EDA
# Step 1) Data Loading
# new_review = ['<br \> I thinnk the first one hour is interesting but \
#               the second half of the movie is boring. This movie just wasted my precious \
#                   time and hard earned money. This movie should be banned to avoid \
#                       time being wasted.<br \>']
                      
new_review = [input()]                     

# Step 2) Data Cleaning
# to clean the data
eda = ExploratoryDataAnalysis()
removed_tags = eda.remove_tags(new_review)
cleaned_input = eda.lower_split(removed_tags)

# Step 3) Features selection
# Step 4) Data preprocessing
# to vectorize the new review
# to feed the tokens into keras
loaded_tokenizer = tokenizer_from_json(token)

# # to vectorize the review into integers
new_review = loaded_tokenizer.texts_to_sequences(cleaned_input)
new_review = eda.sentiment_pad_sequences(new_review)

# model prediction
# shape = 200,1 axis =0
outcome = sentiment_classifier.predict(np.expand_dims(new_review,axis=-1)) # ->1,20,1
# positive is [0,1]
# negative is [1,0]
sentiment_dict = {1:'positive', 0:'negative'}
print('this review is ' + sentiment_dict[np.argmax(outcome)])













