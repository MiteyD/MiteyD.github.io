---
title: "Neural Network SMS Text Classifier"
date: 2020-09-09
tags: [data wrangling, data science, messy data]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Machine Learning, Data Wrangling, Data Science, Messy Data"
mathjax: "true"
---

# Neural Network SMS Text Classifier

This project involved creating a machine learning model that will classify SMS messages as either "ham" or "spam".

A "ham" message is a normal message sent by a friend. A "spam" message is an advertisement or a message sent by a company.

## Project Instructions

*Note: You are currently reading this using Google Colaboratory which is a cloud-hosted version of Jupyter Notebook. This is a document containing both text cells for documentation and runnable code cells. If you are unfamiliar with Jupyter Notebook, watch this 3-minute introduction before starting this challenge: https://www.youtube.com/watch?v=inN8seMm7UI*

---

In this challenge, you need to create a machine learning model that will classify SMS messages as either "ham" or "spam". A "ham" message is a normal message sent by a friend. A "spam" message is an advertisement or a message sent by a company.

You should create a function called `predict_message` that takes a message string as an argument and returns a list. The first element in the list should be a number between zero and one that indicates the likeliness of "ham" (0) or "spam" (1). The second element in the list should be the word "ham" or "spam", depending on which is most likely.

For this challenge, you will use the [SMS Spam Collection dataset](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/). The dataset has already been grouped into train data and test data.

The first two cells import the libraries and data. The final cell tests your model and function. Add your code in between these cells.



```python
# import libraries
try:
  # %tensorflow_version only exists in Colab.
  !pip install tf-nightly
except Exception:
  pass
import tensorflow as tf
import pandas as pd
from tensorflow import keras
!pip install tensorflow-datasets
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
```


```python
# get data files
TRAIN_DATA_URL = "https://raw.githubusercontent.com/beaucarnes/fcc_python_curriculum/master/sms/train-data.tsv"
TEST_DATA_URL = "https://raw.githubusercontent.com/beaucarnes/fcc_python_curriculum/master/sms/valid-data.tsv"

train_file_path = tf.keras.utils.get_file("train-data.tsv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("valid-data.tsv", TEST_DATA_URL)
```


```python
# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)
```


```python
!head {train_file_path}
```


```python
header_list = ["sms_class", "message"]
df_train = pd.read_csv(train_file_path, delimiter='\t', quoting=3, names=header_list)
df_test = pd.read_csv(test_file_path, delimiter='\t', quoting=3, names=header_list)
```


```python
df_train = df_train[['message', 'sms_class']]
df_test = df_test[['message', 'sms_class']]
```


```python
# Replacing string values to numbers
df_train['sms_class'] = df_train['sms_class'].apply({'ham':0, 'spam':1}.get) 
df_test['sms_class'] = df_test['sms_class'].apply({'ham':0, 'spam':1}.get) 
```


```python
# Cleaning the texts in the training set
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus_train = []
for i in range(0, 4179):
    review = re.sub('[^a-zA-Z0-9]', ' ', df_train['message'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus_train.append(review)
```


```python
# Cleaning the texts in the test set
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus_test = []
for i in range(0, 1392):
    review = re.sub('[^a-zA-Z0-9]', ' ', df_test['message'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus_test.append(review)
```


```python
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(corpus_train)

X_train = tokenizer.texts_to_sequences(corpus_train)
X_test = tokenizer.texts_to_sequences(corpus_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

print(corpus_train[2])
print(X_train[2])
```


```python
from keras.preprocessing.sequence import pad_sequences

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

print(X_train[0, :])
```


```python
from keras.models import Sequential
from keras import layers

embedding_dim = 50

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=maxlen))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
```


```python
history = model.fit(X_train, y_train,
                    epochs=10,
                    verbose=True,
                    validation_data=(X_test, y_test),
                    batch_size=10)
```


```python
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
```


```python
# function to predict messages based on model
# (should return list containing prediction and label, ex. [0.008318834938108921, 'ham'])
def predict_message(pred_text):
  pred_text = [pred_text]
  df_pred = pd.DataFrame(pred_text)
  df_pred = df_pred.rename(columns={0:'message'})
  # Cleaning the texts in the test set
  import re
  import nltk
  nltk.download('stopwords')
  from nltk.corpus import stopwords
  from nltk.stem.porter import PorterStemmer
  corpus_pred = []
  for i in range(0, len(pred_text)):
    review = re.sub('[^a-zA-Z0-9]', ' ', df_pred['message'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus_pred.append(review)

    sequence = tokenizer.texts_to_sequences(corpus_pred)
    # pad the sequence
    sequence = pad_sequences(sequence, maxlen=maxlen)
    # get the prediction
    prediction = model.predict(sequence)

  if prediction >= 0.5:
    prediction = ([prediction, 'spam'])
  else:
    prediction = ([prediction, 'ham'])

  
  
  return (prediction)

pred_text =  "how are you doing today"

prediction = predict_message(pred_text)
print(prediction)
```


```python
# Run this cell to test your function and model. Do not modify contents.
def test_predictions():
  test_messages = ["how are you doing today",
                   "sale today! to stop texts call 98912460324",
                   "i dont want to go. can we try it a different day? available sat",
                   "our new mobile video service is live. just install on your phone to start watching.",
                   "you have won £1000 cash! call to claim your prize.",
                   "i'll bring it tomorrow. don't forget the milk.",
                   "wow, is your arm alright. that happened to me one time too"
                  ]

  test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
  passed = True

  for msg, ans in zip(test_messages, test_answers):
    prediction = predict_message(msg)
    if prediction[1] != ans:
      passed = False

  if passed:
    print("You passed the challenge. Great job!")
  else:
    print("You haven't passed yet. Keep trying.")

test_predictions()

```
