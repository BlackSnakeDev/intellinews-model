# Step 1 Load and Inspect Data
import pandas as pd

# Load the dataset
data = pd.read_csv('pathtodata.csv')

# Inspect the dataset
print(data.head())
print(data['Label'].value_counts())

# Step 2 Data Preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenization and Padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['Text'])
sequences = tokenizer.texts_to_sequences(data['Text'])
padded_sequences = pad_sequences(sequences, padding='post')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, data['Label'], test_size=0.2, random_state=42)

# Convert labels to integers and make them zero-indexed
label_mapping = {'positive' 2, 'neutral' 1, 'negative' 0}

y_train = y_train.apply(lambda x label_mapping[x])
y_test = y_test.apply(lambda x label_mapping[x])

# Step 3 Use Pre-trained Embeddings
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Download and load GloVe embeddings
!wget httpnlp.uoregon.edudownloadembeddingsglove.6B.100d.txt -P content

embedding_index = {}
with open('contentglove.6B.100d.txt', 'r', encoding='utf-8') as file
    for line in file
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1], dtype='float32')
        embedding_index[word] = coefs

# Create an embedding matrix
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items()
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None
        embedding_matrix[i] = embedding_vector

# Create an embedding layer
embedding_layer = layers.Embedding(input_dim=vocab_size,
                                   output_dim=embedding_dim,
                                   weights=[embedding_matrix],
                                   input_length=X_train.shape[1],
                                   trainable=False)

# Step 4 Model Building
model = tf.keras.Sequential([
    embedding_layer,
    layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')  # Adjust the output layer for three classes
])

# Step 5 Compile and Train the Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Using EarlyStopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, callbacks=[early_stopping])

# Step 6 Evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print(fTest Accuracy {accuracy})

# Step 7 Tuning and Optimization
# Experiment with different model architectures, hyperparameters, and embeddings to improve performance.
