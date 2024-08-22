#pip install pandas, flask, flask-restful, flask-cors

from flask import Flask, render_template, request, jsonify
from transformers import TFAutoModel, AutoTokenizer
import tensorflow as tf
import os,sys,re
import pandas as pd
from datasets import load_dataset, DatasetDict

BATCH_SIZE = 64   

def remove_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
#=================================================================

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
  return tokenizer(batch["text"], padding=True, truncation=True)

def order(inp):
    '''
    This function will group all the inputs of BERT
    into a single dictionary and then output it with
    labels.
    '''
    data = list(inp.values())
    return {
        'input_ids': data[1],
        'attention_mask': data[2],
        'token_type_ids': data[3]
    }, data[0]

class BERTForClassification(tf.keras.Model):

    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert = bert_model
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.bert(inputs)[1]
        return self.fc(x)

app = Flask(__name__)

@app.route('/')
def index():
    #return "Hello World"
    return render_template("index.html")

@app.route('/', methods=['POST'])
def make_prediction():

    try:
        remove_files_in_folder("static")
        remove_files_in_folder("static/file1")
        remove_files_in_folder("static/file2")

        #render_template('index2.html', result="The model is running. Please wait.")
        
        file = request.files['input_data1']
        if file.filename == '':
            return 'No selected file'
        if file:
            data1_name = file.filename
            file.save('static/file1/' + file.filename)

        dataset = load_dataset("csv", data_files=os.path.join('static/file1/' + file.filename))
        
        ds_train_devtest = dataset['train'].train_test_split(test_size=0.2, seed=42)
        ds_devtest = ds_train_devtest['test'].train_test_split(test_size=0.5, seed=42)


        ds_splits = DatasetDict({
            'train': ds_train_devtest['train'],
            'valid': ds_devtest['train'],
            'test': ds_devtest['test']
        })
        
        ds_encoded = ds_splits.map(tokenize, batched=True, batch_size=None)
        ds_encoded.set_format('tf', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])

        input_text = request.form['input_textClass']
        # converting train split of `ds_encoded` to tensorflow format
        train_dataset = tf.data.Dataset.from_tensor_slices(ds_encoded['train'][:])
        # set batch_size and shuffle
        train_dataset = train_dataset.batch(BATCH_SIZE).shuffle(1000)
        # map the `order` function
        train_dataset = train_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)

        #... doing the same for test set ...
        test_dataset = tf.data.Dataset.from_tensor_slices(ds_encoded['test'][:])
        test_dataset = test_dataset.batch(BATCH_SIZE)
        test_dataset = test_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)

        classifier = BERTForClassification(model, num_classes=6)

        classifier.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        history = classifier.fit(
            train_dataset,
            epochs=3
        )

        evaluation = classifier.evaluate(test_dataset)
        render_template('index2.html', result=evaluation)
    except:
        render_template('index2.html', result="Prediction Failed. Please check your input and retry.")

#model = load_model('my_model.h5')

if __name__ == '__main__':
    app.run(port=5000, debug=True)

