import flask
import numpy as np
from flask import Flask
from flask import request
import onnxruntime as ort

app = Flask(__name__)

Model = ort.InferenceSession('./models/mnist_mlp_sklearn.onnx')

def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()

def prediction(images:np.array):
    if len(images[0]) != 784:
        print(f"Length of image is {len(images[0])}")
        raise ValueError()
    onnx_input = {"input": images}
    output = Model.run(None,onnx_input)
    print(output)
    return [int(output[0][0]),list(output[1][0].values())]

@app.route("/predict")
def predict():
    input_image = request.args.get("image")
    
    response = {}
    response["response"] = prediction(input_image)
    return flask.jsonify(response)

@app.route("/health")
def health():
    response = {}
    response['response'] = prediction([np.load("./.data/test_array.npy")])
    return flask.jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)    
