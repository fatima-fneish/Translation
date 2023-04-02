import os
import numpy as np
# hide TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json

class EngFrenTranslator :

    def __init__(self, model_path,eng_tokenizer_path, frn_tokenizer_path):
        
        logging.info("EngFrenTranslator class initialized")
        self.model = load_model(model_path)
        logging.info("Model is loaded!")

        with open(eng_tokenizer_path, 'r') as f:
            eng_tokenizer_json = json.load(f)

        # Convert the dictionary to a JSON string
            eng_tokenizer_json_str = json.dumps(eng_tokenizer_json)

        # Convert the JSON string to a tokenizer object
            self.eng_tokenizer = tokenizer_from_json(eng_tokenizer_json_str)

        with open(frn_tokenizer_path, 'r') as f:
            frn_tokenizer_json = json.load(f)
            # Convert the dictionary to a JSON string
            frn_tokenizer_json_str = json.dumps(frn_tokenizer_json)

        # Convert the JSON string to a tokenizer object
            self.frn_tokenizer = tokenizer_from_json(frn_tokenizer_json_str)

    def prepare(self,input_sentence):

    # Convert the input sentence to a sequence of integers
        input_seq = self.eng_tokenizer.texts_to_sequences([input_sentence])
    
    # Pad the input sequence to match the model's input shape
        input_seq = pad_sequences(input_seq, maxlen=15, padding='post')
     
        return input_seq
    

    def predict(self, input):
    # prepare the text
        txt = self.prepare(input)

    # predict the output
        result = self.model.predict(txt)

    # return output
        output = self.output_format(result)
        return output

    
    def output_format(self,output):

         # Convert the output sequence to a sequence of integers
         output = np.argmax(output, axis=-1)

         # Convert the sequence of integers to a list and remove any padding
         output= output[0].tolist()
         output = [i for i in output if i != 0]

         # Convert the list of integers to a French sentence
         output_sentence = self.frn_tokenizer.sequences_to_texts([output])[0]

         # Remove any special characters 
         output_sentence=output_sentence.replace('<start> ', '').replace(' <end>', '')
         # return the output sentence
         return output_sentence

def main():
	model = EngFrenTranslator('my_model.h5','eng_tokenizer.json','frn_tokenizer.json')
	translation = model.predict("she is driving the truck")
	logging.info("Output sentence: {}".format(translation))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
