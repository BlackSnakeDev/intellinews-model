from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from waitress import serve
import time
import threading
from transformers import MarianMTModel, MarianTokenizer

model = tf.keras.models.load_model('sentiment_model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_encoder.pkl', 'rb') as handle:
    label_encoder = pickle.load(handle)

model_name = 'Helsinki-NLP/opus-mt-fr-en'
translation_tokenizer = MarianTokenizer.from_pretrained(model_name)
translation_model = MarianMTModel.from_pretrained(model_name)

app = Flask(__name__)

start_time = time.time()
request_count = 0
translation_count = 0
request_lock = threading.Lock()

def translate(text, tokenizer, model):
    tokens = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**tokens)
    translation = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    return translation

@app.route('/predict', methods=['POST'])
def predict():
    global request_count
    with request_lock:
        request_count += 1
    
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

@app.route('/translate', methods=['POST'])
def translate_text():
    global translation_count
    try:
        data = request.get_json(force=True)
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        with request_lock:
            translation_count += 1

        translation = translate(text, translation_tokenizer, translation_model)
        return jsonify({ 'translation': translation})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    latency = measure_latency()
    uptime = time.time() - start_time
    with request_lock:
        total_requests = request_count
        total_translations = translation_count
    
    if latency is None:
        return jsonify({'status': 'unhealthy', 'message': 'Latency measurement failed'}), 500
    
    return jsonify({
        'status': 'healthy',
        'latency': latency,
        'uptime': round(uptime, 2),
        'total_requests': total_requests,
        'total_translations': total_translations
    })

def measure_latency():
    start_time = time.time()
    time.sleep(0.1)
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    return round(latency_ms, 2)

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5000)
