import os
# hide TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class EngFrenTranslator :

    def __init__(self, model_path):
        
        logging.info("EngFrenTranslator class initialized")
        self.model = load_model(model_path)
        logging.info("Model is loaded!")
        

     def translator(self,input_sentence):
    # Convert the input sentence to a sequence of integers
        input_seq = eng_tokenizer.texts_to_sequences([input_sentence])
    
    # Pad the input sequence to match the model's input shape
        input_seq = pad_sequences(input_seq, maxlen=max_eng_length, padding='post')
    # Get the model's predicted output sequence
        output_seq = model.predict(input_seq)
    # Convert the output sequence to a sequence of integers
        output_seq = np.argmax(output_seq, axis=-1)
    # Convert the sequence of integers to a list and remove any padding
        output_seq = output_seq[0].tolist()
        output_seq = [i for i in output_seq if i != 0]
    # Convert the list of integers to a French sentence
        output_sentence = frn_tokenizer.sequences_to_texts([output_seq])[0]
    # Remove any special characters and return the output sentence
    return output_sentence.replace('<start> ', '').replace(' <end>', '')


def main():
	model = EngFrenTranslator('my_model.h5')
	predicted_class = model.predict("https://cdn.britannica.com/60/8160-050-08CCEABC/German-shepherd.jpg")
	logging.info("This is an image of a {}".format(predicted_class)) 


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()