from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Define local paths
model_path = 'final_model_vectorizer/model/model/model.pkl'
vectorizer_path = 'final_model_vectorizer/vectorizer/vectorizer.pkl'

# Load the model and vectorizer
try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    model, vectorizer = None, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vectorizer:
        return render_template('index.html', prediction='Model or vectorizer not loaded')
    
    sms_text = request.form.get('sms_text')
    
    if not sms_text:
        return render_template('index.html', prediction='No text provided')

    text_vector = vectorizer.transform([sms_text])
    prediction = model.predict(text_vector)
    label = 'Spam' if prediction[0] else 'Not Spam'
    
    return render_template('index.html', prediction=label)

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5001)
