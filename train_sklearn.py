"""Train an sklearn MLPClassifier on MNIST (equivalent architecture to the PyTorch MLP).

This script:
- loads MNIST via torchvision (ToTensor -> [0,1])
- flattens images to 784-d vectors
- trains a Pipeline of StandardScaler + MLPClassifier
- evaluates on the test set and saves the trained pipeline with joblib
- exports the trained pipeline to ONNX using skl2onnx if available

Notes:
- The original PyTorch script normalized images to roughly [-1,1]. Here we use
  StandardScaler inside a Pipeline so the same preprocessing is included in the
  exported ONNX graph.
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib
import torchvision

# Optional imports for ONNX export
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    SKL2ONNX_AVAILABLE = True
except Exception:
    SKL2ONNX_AVAILABLE = False


def dataset_to_numpy(dataset):
    """Convert a torchvision dataset with ToTensor() transform to numpy arrays.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 784)
    y : np.ndarray, shape (n_samples,)
    """
    X_list = []
    y_list = []
    for img, label in dataset:
        # img is a tensor with shape (1, 28, 28), convert to numpy and flatten
        arr = img.numpy().reshape(-1)
        X_list.append(arr)
        y_list.append(int(label))
    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def main():
    # Load datasets (ToTensor gives values in [0,1])
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_set = torchvision.datasets.MNIST(root='.data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='.data', train=False, download=True, transform=transform)

    print(f'Number of train samples: {len(train_set)}')
    print(f'Number of test samples: {len(test_set)}')

    X_train, y_train = dataset_to_numpy(train_set)
    X_test, y_test = dataset_to_numpy(test_set)

    # Build a pipeline that includes scaling + classifier. This ensures the same
    # preprocessing is applied at inference and is included when exporting to ONNX.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                              batch_size=32, max_iter=5, verbose=True, random_state=0))
    ])

    print('Starting training sklearn Pipeline (StandardScaler + MLPClassifier)...')
    # Fit pipeline directly on raw X (scaler will be fit internally)
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy on test set: {acc:.4f}')
    print('\nClassification report:\n')
    print(classification_report(y_test, y_pred))

    # Save pipeline (includes scaler and classifier)
    joblib.dump(pipeline, 'mnist_mlp_sklearn_pipeline.joblib')
    print('Saved trained pipeline to mnist_mlp_sklearn_pipeline.joblib')

    # Export to ONNX if skl2onnx is available
    onnx_path = 'mnist_mlp_sklearn.onnx'
    if SKL2ONNX_AVAILABLE:
        try:
            # The pipeline expects input of shape (N, 784)
            initial_type = [('input', FloatTensorType([None, 784]))]
            onx = convert_sklearn(pipeline, initial_types=initial_type)
            with open(onnx_path, 'wb') as f:
                f.write(onx.SerializeToString())
            print(f'Successfully exported sklearn pipeline to {onnx_path}')
        except Exception as e:
            print(f'Failed to export sklearn pipeline to ONNX: {e}')
    else:
        print('skl2onnx is not available; skipping ONNX export. Install "skl2onnx" to enable this.')


if __name__ == '__main__':
    main()
