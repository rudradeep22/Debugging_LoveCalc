import pandas as pd
import numpy as np
import string
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout, Lambda
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

df = pd.read_csv('love_calculator_data.csv')
np.random.seed(42)
tf.random.set_seed(42)

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df['name1'].tolist() + df['name2'].tolist())
name1_seq = tokenizer.texts_to_sequences(df['name1'])
name2_seq = tokenizer.texts_to_sequences(df['name2'])

max_len = max(max(len(seq) for seq in name1_seq), max(len(seq) for seq in name2_seq))
name1_seq = tf.keras.preprocessing.sequence.pad_sequences(name1_seq, maxlen=max_len, padding='post')
name2_seq = tf.keras.preprocessing.sequence.pad_sequences(name2_seq, maxlen=max_len, padding='post')

X1 = np.concatenate([name1_seq, name2_seq], axis=0)
X2 = np.concatenate([name2_seq, name1_seq], axis=0)
y = np.concatenate([df['percentage'].values, df['percentage'].values], axis=0)

X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, test_size=0.2, random_state=42)
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 16

input1 = Input(shape=(max_len,))
input2 = Input(shape=(max_len,))

embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)

emb1 = embedding(input1)
emb2 = embedding(input2)

flat1 = Flatten()(emb1)
flat2 = Flatten()(emb2)

concat = Concatenate()([flat1, flat2])
dense1 = Dense(128, activation='relu')(concat)
dropout1 = Dropout(0.5)(dense1)
dense2 = Dense(64, activation='relu')(dropout1)
dropout2 = Dropout(0.5)(dense2)
dense3 = Dense(1, activation='sigmoid')(dropout2)

output = Lambda(lambda x: x * 100)(dense3)

model = Model(inputs=[input1, input2], outputs=output)
optimizer = tf.keras.optimizers.Adam(0.0000001, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mean_absolute_error'])

model.summary()

checkpoint_filepath = 'model.weights.h5'
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,
                                      save_weights_only=True,
                                      save_best_only=True,
                                      monitor='val_loss',
                                      mode='min',
                                      verbose=1)

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
history = model.fit([X1_train, X2_train], y_train,
                    epochs=1000,
                    batch_size=32,
                    validation_data=([X1_test, X2_test], y_test),
                    callbacks=[checkpoint_callback, early_stopping_callback])
loss, mae = model.evaluate([X1_test, X2_test], y_test)
print(f"Test MAE: {mae:.2f}")
