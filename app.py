from flask import Flask, request, jsonify
from model import load_model  # Assume a function to load your trained model

app = Flask(__name__)
model = load_model('path_to_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    prediction = model.predict(image)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)