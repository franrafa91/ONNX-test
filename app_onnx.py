import flask
import numpy as np
from flask import Flask
from flask import request
import onnxruntime as ort
import json


app = Flask(__name__)

Model = ort.InferenceSession('./models/mnist_mlp_sklearn.onnx')

def prediction(images: np.array):
    """Run inference on MNIST digit images using the ONNX model.
    
    This function takes a batch of flattened 28x28 grayscale images and returns
    predictions for each image. The model outputs both a predicted digit and
    probability scores for all 10 digit classes (0-9).
    
    Parameters
    ----------
    images : np.ndarray
        Array of shape (n_samples, 784) where each row is a flattened 28x28 image.
        Pixel values should be normalized (typically in range [0, 1]).
        Example: For a single image, shape should be (1, 784).
    
    Returns
    -------
    list
        A list containing:
        - int: Predicted digit (0-9) for the first image in the batch
        - list: Probability scores for all 10 digit classes (0-9) for the first image
    
    Raises
    ------
    ValueError
        If any image does not have exactly 784 features (28x28 pixels).
    
    Examples
    --------
    >>> # Single image prediction
    >>> image = np.random.rand(1, 784)  # Random 28x28 image
    >>> predicted_digit, probabilities = prediction(image)
    >>> print(f"Predicted: {predicted_digit}")
    Predicted: 7
    >>> print(f"Probabilities: {probabilities}")
    Probabilities: [0.01, 0.02, ..., 0.85, ...]  # 10 values
    
    >>> # Multiple images (only first is returned)
    >>> batch = np.random.rand(5, 784)  # 5 images
    >>> predicted_digit, probabilities = prediction(batch)
    """
    if len(images[0]) != 784:
        print(f"Length of image is {len(images[0])}")
        raise ValueError()
    onnx_input = {"input": images}
    output = Model.run(None, onnx_input)
    print(output)
    return [int(output[0][0]), list(output[1][0].values())]

@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Handle prediction requests for MNIST digit classification.
    
    This endpoint accepts a JSON-encoded array of 784 pixel values representing
    a flattened 28x28 grayscale image. Supports both GET (query parameter) and
    POST (JSON body) methods. Returns the predicted digit and probability scores
    for all 10 classes.
    
    Parameters (GET)
    ----------------
    image : str (query parameter)
        JSON-formatted string containing an array of 784 float values in range [0, 1].
        Example: ?image=[0.0,0.1,0.2,...,0.9]
    
    Parameters (POST)
    -----------------
    image : list (JSON body)
        Array of 784 float values in range [0, 1].
        Example: {"image": [0.0, 0.1, 0.2, ..., 0.9]}
    
    Returns
    -------
    flask.Response
        JSON response with the following structure:
        {
            "response": [
                <int>: predicted_digit,  # Predicted digit (0-9)
                <list>: probabilities    # List of 10 probability scores
            ]
        }
    
    Error Responses
    ---------------
    400: No data provided or invalid JSON format
    500: Internal prediction error
    
    Examples
    --------
    GET /predict?image=[0.0,0.1,0.2,...,0.9]
    
    POST /predict with JSON body:
    {"image": [0.0, 0.1, 0.2, ..., 0.9]}
    
    Response:
    {
        "response": [7, [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.65, 0.04, 0.03]]
    }
    
    Notes
    -----
    For large data (784 floats), POST is recommended over GET to avoid URL length limits.
    """
    # Try to get data from POST body first, then fall back to GET query parameter
    if request.method == "POST":
        data = request.get_json()
        if data is None or "image" not in data:
            return flask.jsonify({"error": "No data provided. Send JSON with 'image' key"}), 400
        data_list = data["image"]
    else:  # GET
        data_str = request.args.get("image")
        if data_str is None:
            return flask.jsonify({"error": "No data provided. Use ?image=[...]"}), 400
        try:
            data_list = json.loads(data_str)
        except json.JSONDecodeError:
            return flask.jsonify({"error": "Invalid JSON format. Use format: [0.1, 0.2, ...]"}), 400
    
    try:
        # Convert to numpy array with proper shape
        data_array = np.array(data_list, dtype=np.float32)
        
        # Reshape to (1, 784) if it's a single flat image
        if data_array.shape == (784,):
            data_array = data_array.reshape(1, 784)
        elif len(data_array.shape) == 1:
            data_array = data_array.reshape(1, -1)
        
        # Make prediction
        result = prediction(data_array)
        
        return flask.jsonify({"response": result})
    except Exception as e:
        return flask.jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    """Health check endpoint that tests the model with a sample image.
    
    This endpoint loads the first available test image from the test data directory
    and runs a prediction to verify the model is functioning correctly.
    
    Returns
    -------
    flask.Response
        JSON response with prediction result for the test image.
        Same structure as /predict endpoint.
    
    Notes
    -----
    This endpoint is useful for:
    - Verifying the model is loaded correctly
    - Testing API connectivity
    - Health checks in deployment environments
    """
    import os
    test_dir = './.data/test_data/'
    with open(f"{test_dir}/{os.listdir(test_dir)[0]}", 'r') as file:
        data = json.load(file)
    
    response = {}
    response['response'] = prediction([np.array(data)])
    return flask.jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)    
