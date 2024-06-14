import os
import re
import string

import pandas as pd
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
from sklearn .metrics import classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load the dataset
data = pd.read_csv('hateSpeech_dataset.csv')

def clean_text(text):
    delete_dict = {sp_character: '' for sp_character in string.punctuation}
    delete_dict[' '] = ' '
    table = str.maketrans(delete_dict)
    text1 = text.translate(table)
    textArr = text1.split()
    text2 = ' '.join([w for w in textArr if (not w.isdigit() and len(w) > 3)])

    return text2.lower()

data["tweet"] = data["tweet"].apply(clean_text)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['tweet'], data['class'], test_size=0.2, random_state=42)

# Tokenize text data
tokenizer = Tokenizer(num_words=5000)  # Maximum vocabulary size
tokenizer.fit_on_texts(X_train)
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# Pad sequences to a fixed length
maxlen = 100  # Maximum sequence length
X_train_padded = pad_sequences(X_train_sequences, maxlen=maxlen, padding='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=maxlen, padding='post')

# Calculate class weights
class_weights = {
    0: len(y_train) / (3 * (len(y_train[y_train == 0]))),
    1: len(y_train) / (3 * (len(y_train[y_train == 1]))),
    2: len(y_train) / (3 * (len(y_train[y_train == 2])))
}

# Check if the trained model file exists
if not os.path.exists('hate_speech_rnn_model.h5'):
    # Define RNN model
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=maxlen))
    model.add(LSTM(units=128))
    model.add(Dense(units=1, activation='sigmoid'))

    # Compile the model with class weights
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model with class weights
    model.fit(X_train_padded, y_train, epochs=10, batch_size=32, validation_data=(X_test_padded, y_test), class_weight=class_weights)

    # Save the trained model
    model.save('hate_speech_rnn_model.h5')

# Load RNN model for hate speech detection
rnn_model = load_model('hate_speech_rnn_model.h5')
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_test)

# Function to predict hate speech using RNN with adjusted threshold
def predict_hate_speech_rnn(text):
    text_sequence = tokenizer.texts_to_sequences([text])
    text_padded = pad_sequences(text_sequence, maxlen=100, padding='post')
    prediction = rnn_model.predict(text_padded)
    threshold = 0.6  # Adjust the threshold based on your evaluation
    if prediction[0][0] >= threshold:
        return "Hate Speech"
    else:
        return "Not Hate Speech"

# Test the classifier with user input
user_input = input("Enter a string: ")
prediction_rnn = predict_hate_speech_rnn(user_input)
print("RNN Prediction:", prediction_rnn)