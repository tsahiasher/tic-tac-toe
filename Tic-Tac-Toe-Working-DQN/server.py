from flask import Flask, jsonify, render_template, request

from deep_reinforcement_learning import *
usingTensorFlow = False

import numpy as np

if usingTensorFlow:
    sess = tf.InteractiveSession()
    x , prediction, _ = createNetwork()
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("model")
    if checkpoint and checkpoint.model_checkpoint_path:
        s = saver.restore(sess,checkpoint.model_checkpoint_path)
        print("Successfully loaded the model:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
    graph = tf.get_default_graph()
else:
    graph = TorchNetwork()
    try:
        checkpoint = torch.load("./model/model.pt", map_location=torch.device('cpu'))
        graph.load_state_dict(checkpoint['model'])
        graph.eval()
        print("Successfully loaded the model:", "./model/model.pt")
    except:
        print("Could not find old saved model")
        exit()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def bestmove(input):
    global graph
    if usingTensorFlow:
        with graph.as_default():
            data = (sess.run(tf.argmax(prediction.eval(session = sess,feed_dict={x:[input]}),1)))
        move = data[0].item()
    else:
        move = int(np.argmax(graph(torch.as_tensor(input, dtype=torch.float32).reshape(-1)).detach().numpy()))
    return move

@app.route('/api/ticky', methods=['POST'])
def ticky_api():
    data = request.get_json()
    data = np.array(data['data'])
    data = data.tolist()
    b  = bestmove(data)
    #print(b)
    return jsonify(b)

# @app.after_request
# def add_header(r):
#     """
#     Add headers to both force latest IE rendering engine or Chrome Frame,
#     and also to cache the rendered page for 10 minutes.
#     """
#     r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
#     r.headers["Pragma"] = "no-cache"
#     r.headers["Expires"] = "0"
#     r.headers['Cache-Control'] = 'public, max-age=0'
#     return r

if __name__ == '__main__':
    app.run(host='127.0.0.1',port=80,debug=True)

