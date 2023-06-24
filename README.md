# K-Nearest Neighbors (KNN) Classifier

This repository contains an implementation of the K-Nearest Neighbors (KNN) classifier in Python. The KNN algorithm is a simple yet powerful classification algorithm that can be used for various machine learning tasks.

## Installation

To use the KNN classifier, you need to have Python installed on your system. You also need to install the following dependencies:

- numpy
- scikit-learn
- matplotlib

You can install the dependencies using pip:

```
pip install numpy scikit-learn matplotlib
```

## Usage

To use the KNN classifier, follow these steps:

1. Import the necessary modules and functions:

   ```python
   from knn import KNN
   import numpy as np
   from sklearn import datasets
   from sklearn.model_selection import train_test_split
   import matplotlib.pyplot as plt
   from matplotlib.colors import ListedColormap
   ```

2. Load the dataset:

   ```python
   iris = datasets.load_iris()
   X, y = iris.data, iris.target
   ```

3. Split the dataset into training and testing sets:

   ```python
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=1234)
   ```

4. Initialize the KNN classifier with the desired value of k:

   ```python
   k = 3
   clf = KNN(k=k)
   ```

5. Train the classifier using the training data:

   ```python
   clf.fit(X_train, y_train)
   ```

6. Make predictions on the test data:

   ```python
   predictions = clf.predict(X_test)
   ```

7. Evaluate the accuracy of the classifier:

   ```python
   def accuracy(y_true, y_pred):
       accuracy = np.sum(y_true == y_pred) / len(y_true)
       return accuracy

   print("Custom KNN classification accuracy:", accuracy(y_test, predictions))
   ```

## Example

Here's an example usage of the KNN classifier using the Iris dataset:

```python
# Import modules and functions
from knn import KNN
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load the Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234)

# Initialize the KNN classifier
k = 3
clf = KNN(k=k)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)

# Evaluate the accuracy
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

print("Custom KNN classification accuracy:", accuracy(y_test, predictions))
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.