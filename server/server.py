
# $env:TF_ENABLE_ONEDNN_OPTS=0


from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from waitress import serve
import time

# Load model and tokenizers globally to ensure they are loaded only once
model = tf.keras.models.load_model('sentiment_model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_encoder.pkl', 'rb') as handle:
    label_encoder = pickle.load(handle)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        headline = data.get('headline', '')

        if not headline:
            return jsonify({'error': 'No headline provided'}), 400

        sequence = tokenizer.texts_to_sequences([headline])
        padded_sequence = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')

        prediction = model.predict(padded_sequence)
        label = label_encoder.inverse_transform([prediction.argmax(axis=1)[0]])

        return jsonify({'headline': headline, 'sentiment': label[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    latency = measure_latency()
    if latency is None:
        return jsonify({'status': 'unhealthy', 'message': 'Latency measurement failed'}), 500
    
    return jsonify({'status': 'healthy', 'latency': latency})

def measure_latency():
    # Simulate a latency measurement (you can replace this with actual logic)
    start_time = time.time()
    # Perform a simple operation to simulate latency
    time.sleep(0.1)
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    return round(latency_ms, 2)

