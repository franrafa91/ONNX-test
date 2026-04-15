import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import torchvision

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


# Load datasets (ToTensor gives values in [0,1])
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
test_set = torchvision.datasets.MNIST(root='.data', train=False, download=True, transform=transform)

print(f'Number of test samples: {len(test_set)}')
X_test, y_test = dataset_to_numpy(test_set)

np.save("./.data/test_array",X_test[0])