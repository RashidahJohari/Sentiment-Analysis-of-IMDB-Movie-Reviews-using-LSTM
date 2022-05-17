# -*- coding: utf-8 -*-
"""
Created on Thu May 12 03:23:39 2022

@author: Acer

"""
import pandas as pd
import numpy as np
import json
import re
import os
import datetime
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.layers import Embedding, Bidirectional
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
#from sys import getsizeof

#%% CONSTANT / STATIC 
URL = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'
TOKENIZER_JSON_PATH = os.path.join(os.getcwd(),'tokenizer_data.json')
PATH_LOGS = os.path.join(os.getcwd(),'log')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')

#%% Classification Problem (positive/negative)
# EDA
# Step 1) Data Loading
df = pd.read_csv(URL)

review = df['review']
review_dummy = review.copy() # X_train

sentiment = df['sentiment']
sentiment_dummy = sentiment.copy() # Y_train

# Step 2) Data Inspection
review_dummy[3] # review 3rd data
sentiment_dummy[3]

review_dummy[11]
sentiment_dummy[11]

# Step 3) Data cleaning
# replace(): not robust
#<> html text
# filter <br />
# everything inside the <> will be removed
# write back to 
# everytime make changes need reload again
# ? : dont be greedy. limit remove only inide<>
# inside diamond <>, filter
#re.sub('<.*?>', '', kata) # everything inside the <> will be filtered
# .* symbols means include everything inside the '<>' will be filtered 
# .? symbol means dont be greedy

# to remove html tags(<>)
# enumerate(): to split position & data
# first row replace with filtered row
# re.sub : re.sub(<data yg nak remove?>, new data yg nak replace)
# replace()

# 1.to remove html tags(<>)
for index, text in enumerate(review_dummy):
    #review_dummy[iii] = kata.replace('<br /', '')
    review_dummy[index] = re.sub('<.*?>', '', text)
    #review_dummy[iii] = re.sub('[^a-zA-Z]', '', kata)
    
# 2. to convert to lowercase and split it & to remove numerical text
for index,text in enumerate(review_dummy):
    review_dummy[index] = re.sub('[^a-zA-Z]',' ', text).lower().split()

# to check the review text in row 223
review_dummy[223]

# Step 4) Features selection
#%% Step 5) Data preprocessing
# a)Data vectorization
# b)One hot encoding for label

# a)Data vectorization
# pick something that :5000
num_words = 5000  # 5000 #1000 to be save # jangan less than 100
oov_token = '<OOV>' # vocab used already, 
# new word not seen before will be replced with oop. assigned digit=1
# any new word coming in will be token return to 1 oov=1
# vectorize string/text

# toeknizer to vectorize the words
tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
tokenizer.fit_on_texts(review_dummy)

# to save the tokenizer for deployment purpose
token_json = tokenizer.to_json()

with open(TOKENIZER_JSON_PATH,'w') as json_file:
    json.dump(token_json, json_file)

# to observe the number of words
word_index = tokenizer.word_index # with integer
print(word_index)
# list(word_index.items())[0:10]
# print(dict())
print(dict(list(word_index.items())[0:10]))

# to vectorize the sequence of text
review_dummy = tokenizer.texts_to_sequences(review_dummy)

# every review has different lenght
# so need to make they have same length
# so need to padding
# pad_sequences(review_dummy, maxlen=200) 

temp = [np.shape(i) for i in review_dummy] # check no.of words inside the list
np.mean(temp)# check no.of words inside the list (234)

review_dummy = pad_sequences(review_dummy,
                             maxlen=200,
                             padding='post',
                             truncating='post')

# b)One hot encoding for label
one_hot_encoder = OneHotEncoder(sparse=False)
#one_hot_encoder.fit_transform(sentiment_dummy) # error array.reshape(1, -1)
sentiment_encoded = one_hot_encoder.fit_transform(np.expand_dims(sentiment_dummy,
                                             axis=-1)) #

# split train test
X_train,X_test,y_train,y_test= train_test_split(review_dummy, 
                                                sentiment_encoded, 
                                                test_size=0.3,
                                                random_state=123)

# X_train = np.array(X_train)
# X_test = np.array(X_test)

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

one_hot_encoder.inverse_transform(np.expand_dims(y_train[0], axis=0))
# positive is [0,1]
# negative is [1,0]

#%% Model Creation

# LSTM (RNN/LSTM/GRU)
model = Sequential()
model.add(Embedding(num_words,64)) # added the embedding layer, 128, 512, 156
model.add(Bidirectional(LSTM(32,return_sequences=True))) # added bidirectional
#model.add(LSTM(32, input_shape=(np.shape(X_train)[1:]),return_sequences=True)) #[1],1
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(32))) # added bidirectional
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax')) #target: positive, negative
model.summary()

# model = Sequential()
# model.add(LSTM(128, input_shape=(np.shape(X_train)[1:]),return_sequences=True)) #[1],1
# model.add(Dropout(0.2))
# model.add(LSTM(128)) # added bidirectional
# model.add(Dropout(0.2))
# model.add(Dense(2, activation='softmax')) #target: positive, negative
# model.summary()

#%% Callbacks
log_dir = os.path.join(PATH_LOGS, 
                         datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

# Tensorboard Callbacks
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

#%% Compile and Model Fitting
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics='acc')

hist = model.fit(X_train,y_train,epochs=3,
          validation_data=(X_test,y_test),
          callbacks=[tensorboard_callback])

#%% Model Evaluation
# model.predict() only accept 3D (batch,length,features)

# append approach
#1 by 1 prediction at 1 time
# predicted = []
# for test in X_test:
#     predicted.append(model.predict(np.expand_dims(test, axis=0)))

# preallocation of memory approach
predicted_advanced = np.empty([len(X_test), 2]) # create empty array
for index,test in enumerate(X_test):
    predicted_advanced[index,:] = model.predict(np.expand_dims(test,axis=0))

#getsizeof(predicted_advanced)

#%% Model Anaysis
y_pred = np.argmax(predicted_advanced, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true,y_pred))
print(accuracy_score(y_true, y_pred))

#%% Model Deployment
model.save(MODEL_SAVE_PATH)

#%%
# tensorboard --logdir C:\Users\Acer\Desktop\RNN\sentiment_analysis\log
# localhost:6006


