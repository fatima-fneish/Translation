from flask import Flask, render_template, request
from model.model import EngFrenTranslator

app = Flask(__name__, template_folder='templates')

# define model path
model_path = './model/my_model.h5'
eng_tokenizer_path='./model/eng_tokenizer.json'
frn_tokenizer_path='./model/frn_tokenizer.json'
# create instance
model = EngFrenTranslator(model_path, eng_tokenizer_path, frn_tokenizer_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['input_text']
        output_text = model.predict(input_text)
        return render_template('index.html', output_text=output_text)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run()
