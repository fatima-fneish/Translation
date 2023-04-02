import os
import logging

from flask import Flask, request, jsonify

from model.model import EngFrenTranslator

app = Flask(__name__)  

# define model path
model_path = './model/my_model.h5'
eng_tokenizer_path='./model/eng_tokenizer.json'
frn_tokenizer_path='./model/frn_tokenizer.json'
# create instance
model = EngFrenTranslator(model_path,eng_tokenizer_path,frn_tokenizer_path)
logging.basicConfig(level=logging.INFO)

@app.route("/")
def index():
    """Provide simple health check route."""
    return "Hello world!"


@app.route("/v1/predict", methods=["GET", "POST"])
def predict():
    """Provide main prediction API route. Responds to both GET and POST requests."""
    logging.info("Predict request received!")
    input = request.args.get("input")
    prediction = model.predict(input)
    
    logging.info("prediction from model= {}".format(prediction))
    return jsonify({"predicted_class": str(prediction)})

def main():
    """Run the Flask app."""
    app.run(host="0.0.0.0", port=8000, debug=False) 


if __name__ == "__main__":
    main()