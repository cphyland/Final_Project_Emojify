from flask import Flask, render_template, redirect
from sqlalchemy import create_engine
import pandas as pd
from flask import request
from flask import jsonify
import os
import sys
import numpy as np
import tensorflow as tf
import pickle
import pdb
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


shark_api_url = "http://api.fish.wa.gov.au/webapi/v1/RawData"
shark_table_name = "sharks1"
shark_csv_path = "../data/sharks/sharks_cleaned.csv"

# Flask Setup
app = Flask(__name__)

def get_model():
	global model
	model = load_model('emoji_model.h5')
	print('Model Loaded!!')

graph = tf.get_default_graph()

tokenizer = pickle.load(open('tokenizer.pickle','rb'))


# Flask Routes
@app.route("/")
def index():
  return render_template("view.html")

  @app.route('/predict',methods = ['POST'])
def predict():
	global graph
	global tokenizer
	with graph.as_default():
		maxlen = 50
		text = request.form['name']
		test_sent = tokenizer.texts_to_sequences([text])
		test_sent = pad_sequences(test_sent, maxlen = maxlen)
		pred = model.predict(test_sent)
		response = {
		'prediction': int(np.argmax(pred))
		}
	return jsonify(response)

    @app.route('/update',methods = ['POST'])
def update():
	global graph
	global tokenizer
	with graph.as_default():
		maxlen = 50
		text = request.form['sentence']
		test_sent = tokenizer.texts_to_sequences([text])
		test_sent = pad_sequences(test_sent, maxlen = maxlen)
		test_sent = np.vstack([test_sent] * 5)
		actual_output = request.form['dropdown_value']
		output_hash = {
			'Angry': np.array([1.,0.,0.,0.,0.,0.,0.]),
			'Discusted': np.array([0.,1.,0.,0.,0.,0.,0.]),
			'Fearful': np.array([0.,0.,1.,0.,0.,0.,0.]),
			'Happy': np.array([0.,0.,0.,1.,0.,0.,0.]),
			'Neutral': np.array([0.,0.,0.,0.,1.,0.,0.]),
			'Sad': np.array([0.,0.,0.,0.,0.,1.,0.]),
			'Surprised': np.array([0.,0.,0.,0.,0.,0.,1.]),
					}
		actual_output = output_hash[actual_output].reshape((1,7))
		actual_output = np.vstack([actual_output] * 5)
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.fit(test_sent, actual_output, epochs = 10, batch_size = 32, shuffle=True)
		model.save('emoji_model.h5')
		get_model()
		response = {
		'update_text': 'Updated the values!! Should work in next few attempts..'
		}
	return redirect("/")

if __name__ == "__main__":
	get_model()
	app.run(host="0.0.0.0", port=5000,debug=True)