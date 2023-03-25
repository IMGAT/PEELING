import os
from flask import Flask, request
from flask_cors import CORS
import json
from model.training_model import SpERTTrainer
import re
from gevent import pywsgi
from online_predict import EventExtraction
import logging
app = Flask(__name__)
CORS(app, supports_credentials=True)


@app.route('/api/EntityRelationEx/', methods=['GET', 'POST'])
def EntityRelationEx():
	data = request.data
	print(data)
	data = data.decode(encoding="utf-8")
	content = json.loads(data)
	
	text = content['text']
	text = ''.join(text.split())
	text = content['text']
	text_list = []
	text_list.append(text)
	summary = model.predict(text_list)
	return json.dumps(summary)



@app.route('/')
def index():
	return app.send_static_file("index.html")
if __name__ == '__main__':
	model = SpERTTrainer()
	print("cuda")
	server = pywsgi.WSGIServer(('0.0.0.0', 8000), app)
	server.serve_forever()
