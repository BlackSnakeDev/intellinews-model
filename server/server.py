from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model, tokenizer, and label encoder
model = tf.keras.models.load_model('sentiment_model.h5')

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pkl', 'rb') as handle:
    label_encoder = pickle.load(handle)

# Initialize Flask app
app = Flask(__name__)

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.get_json()
    headline = data['headline']

    # Preprocess the input
    sequence = tokenizer.texts_to_sequences([headline])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')

    # Make prediction
    prediction = model.predict(padded_sequence)
    label = label_encoder.inverse_transform([prediction.argmax(axis=1)[0]])

    # Return the result
    return jsonify({'headline': headline, 'sentiment': label[0]})

if __name__ == '__main__':
    app.run(debug=True)
