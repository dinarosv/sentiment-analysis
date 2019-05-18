import tensorflow as tf
import numpy as np
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

# Load data
data = pd.read_csv('ns_tweets.csv', sep=';', error_bad_lines=False)
X = data["text"]
y = data["sentiment"]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)

# VALUES
EMBEDDING_SIZE = 8
NUM_WORDS = 10000
UNITS = 64
DROPOUT = 0.6
R_DROPOUT = 0.3
UNITS2 = 32
DROPOUT2 = 0.6
R_DROPOUT2 = 0.3
LR = 0.005
EPOCHS = 6
BATCHSIZE = 256
MODELNAME = "model.h5"

# Tokenizer
num_words = NUM_WORDS
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(x_train)
x_train_tokens = tokenizer.texts_to_sequences(x_train)
x_test_tokens = tokenizer.texts_to_sequences(x_test)

# Padding
num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
pad = 'pre'
x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens, padding=pad, truncating=pad)
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens, padding=pad, truncating=pad)

# Model
model = Sequential()

# Embedding
model.add(Embedding(input_dim=num_words,
                    output_dim=EMBEDDING_SIZE, # Embedding size
                    input_length=max_tokens,
                    name='layer_embedding'))

# Layers
model.add(LSTM(units=UNITS, dropout=DROPOUT, recurrent_dropout=R_DROPOUT, return_sequences=True))
model.add(LSTM(units=UNITS2, dropout=DROPOUT2, recurrent_dropout=R_DROPOUT2))
model.add(Dense(3, activation='softmax'))

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# Sette lossfunction og optimaliseringsfunksjon for modellen
optimizer = Adam(lr=LR)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

print(model.summary())

# Trene modellen på treningssettet
model.fit(x_train_pad, y_train, validation_split=0.05, epochs=EPOCHS, batch_size=BATCHSIZE, callbacks=[tensorboard])

result = model.evaluate(x_test_pad, y_test, batch_size=BATCHSIZE)
print(result)

model.save(MODELNAME)
