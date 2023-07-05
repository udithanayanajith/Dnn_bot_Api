from tensorflow.python.framework import ops
import tflearn
import numpy as np
import pickle
import nltk
import json
from flask import Flask, request, jsonify
import random
from flask_cors import CORS

stemmer = nltk.stem.lancaster.LancasterStemmer()


with open("intents.json") as file:
    data = json.load(file)

with open("model.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

app = Flask(__name__)
CORS(app)

@app.route('/get', methods=['GET'])
def get_bot_response():
    # global seat_count
    message =request.args.to_dict()
    if 'msg' in message:
        message = message['msg'].lower()
        results = model.predict([bag_of_words(message, words)])[0]
        result_index = np.argmax(results)
        tag = labels[result_index]
        if results[result_index] > 0.7:
            for tg in data['intents']:
                if tg['tag'] == tag:
                    responses = tg['responses']
            response = random.choice(responses)
        else:
            response = "I didn't quite get that, please try again."
        return jsonify(response=response), 200, {'Content-Type': 'application/json'}
    return jsonify({"Missing Data!"})



if __name__ == "__main__":
    app.run()

