# -*- coding: utf-8 -*-
"""
Created on Wed May 11 15:06:37 2022

@author: Acer
"""

from tensorflow.keras.models import load_model
import os
import re
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

#%% Constant/Static here
MODEL_PATH = os.path.join(os.getcwd(),'model.h5')
JSON_PATH = os.path.join(os.getcwd(),'tokenizer_data.json')

#%% Model Loading
sentiment_classifier = load_model(MODEL_PATH)
sentiment_classifier.summary()

#%% Tokenizer Loading
with open(JSON_PATH,'r') as json_file:
    loaded_tokenizer = json.load(json_file)

#%% Deployment

new_review = ['<br \> I thinnk the first one hour is interesting but \
              the second half of the movie is boring. This movie just wasted my precious \
                  time and hard earned money. This movie should be banned to avoid \
                      time being wasted.<br \>']
    
# Data cleaning
for index, text in enumerate(new_review):
    #review_dummy[iii] = kata.replace('<br /', '')
    new_review[index] = re.sub('<.*?>', '', text)
    #review_dummy[iii] = re.sub('[^a-zA-Z]', '', kata)
    
# 2. to convert to lowercase and split it & to remove numerical text
for index,text in enumerate(new_review):
    new_review[index] = re.sub('[^a-zA-Z]',' ', text).lower().split()
    
# to vectorize the new review
loaded_tokenizer = tokenizer_from_json(loaded_tokenizer)
new_review = loaded_tokenizer.texts_to_sequences(new_review)
new_review = pad_sequences(new_review,
                             maxlen=200,
                             truncating='post',
                             padding='post')

#%% Model Prediction
# shape = 200,1 axis =0
outcome = sentiment_classifier.predict(np.expand_dims(new_review,axis=-1)) # ->1,20,1
print(np.argmax(outcome))

# positive is [0,1]
# negative is [1,0]
sentiment_dict = {1:'positive', 0:'negative'}
print('this review is ' + sentiment_dict[np.argmax(outcome)])






    
    