import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
import pandas as pd

from time import time
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, Dropout, LSTM, Conv1D
from tensorflow.python.keras import layers
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.callbacks import TensorBoard

pd.set_option('max_colwidth', 300)
data = pd.read_csv('mixed.csv', sep=';')
data.sample(20)

X = data["text"]
y = data["sentiment"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)
num_words = 10000
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(x_train)
x_train_tokens = tokenizer.texts_to_sequences(x_train)
x_test_tokens = tokenizer.texts_to_sequences(x_test)

num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)

max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)

pad = 'pre'
x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens,
                            padding=pad, truncating=pad)
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens,
                           padding=pad, truncating=pad)

model = Sequential()

embedding_size = 8
model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    input_length=max_tokens,
                    name='layer_embedding'))

model.add(LSTM(units=256, return_sequences=True))
model.add(Dropout(0.6))
model.add(LSTM(units=128))
model.add(Dropout(0.2))

model.add(Dense(3, activation='softmax'))

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
optimizer = Adam(lr=0.005)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
print(model.summary())

model.fit(x_train_pad, y_train, validation_split=0.05, epochs=5, batch_size=2048, callbacks=[tensorboard])

result = model.evaluate(x_test_pad, y_test, batch_size=4096)
print(result)

model.save('modelwithres_test.h5')
