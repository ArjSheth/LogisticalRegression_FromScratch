Below is a **clean architecture for implementing Logistic Regression from scratch in Python** and training it on **Fashion-MNIST**.
The goal is modular code so you can **train, evaluate, and save the model easily**.

Fashion-MNIST contains **28×28 grayscale images (784 features)** and **10 classes**, so the typical approach is **multinomial logistic regression (softmax)**.

---

# 1. Overall Pipeline

Your implementation will follow this structure:

```
load_data()
↓
preprocess_data()
↓
initialize_parameters()
↓
train_model()
    ├── create_mini_batches()
    ├── forward_pass()
    ├── compute_loss()
    ├── backward_pass()
    └── update_parameters()
↓
predict()
↓
evaluate_accuracy()
↓
save_model() / load_model()
```

---

# 2. Functions to Implement

## 1. Load Dataset

```python
def load_fashion_mnist(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
```

**Purpose**

Load train and test sets.

**Input**

* `path: str`
  Directory containing Fashion-MNIST files.

**Output**

```
X_train : np.ndarray (N_train, 784)
y_train : np.ndarray (N_train,)
X_test  : np.ndarray (N_test, 784)
y_test  : np.ndarray (N_test,)
```

**Notes**

* Flatten images from `28×28 → 784`

---

## 2. Normalize Features

```python
def normalize_features(X: np.ndarray) -> np.ndarray:
```

**Purpose**

Scale pixel values.

**Input**

```
X : (N, D)
```

**Output**

```
X_norm : (N, D)
```

**Operation**

```
X_norm = X / 255.0
```

---

## 3. One-Hot Encode Labels

```python
def one_hot_encode(y: np.ndarray, num_classes: int) -> np.ndarray:
```

**Input**

```
y : (N,)
num_classes : int
```

**Output**

```
Y : (N, num_classes)
```

---

## 4. Initialize Parameters

```python
def initialize_parameters(n_features: int, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
```

**Output**

```
W : (n_features, n_classes)
b : (n_classes,)
```

**Example**

```
W = np.random.randn(n_features, n_classes) * 0.01
b = np.zeros(n_classes)
```

---

## 5. Softmax Function

```python
def softmax(z: np.ndarray) -> np.ndarray:
```

**Input**

```
z : (N, num_classes)
```

**Output**

```
probabilities : (N, num_classes)
```

**Formula**

```
softmax(z_i) = exp(z_i) / sum(exp(z))
```

---

## 6. Forward Pass

```python
def forward_pass(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
```

**Input**

```
X : (batch_size, D)
W : (D, C)
b : (C,)
```

**Output**

```
P : (batch_size, C)
```

**Computation**

```
logits = X @ W + b
P = softmax(logits)
```

---

## 7. Compute Cross-Entropy Loss

```python
def compute_loss(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
```

**Input**

```
Y_true : (batch_size, C)
Y_pred : (batch_size, C)
```

**Output**

```
loss : float
```

**Formula**

```
L = - mean(sum(Y_true * log(Y_pred)))
```

---

## 8. Backward Pass (Gradient Calculation)

```python
def backward_pass(X: np.ndarray, Y_true: np.ndarray, Y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
```

**Input**

```
X : (batch_size, D)
Y_true : (batch_size, C)
Y_pred : (batch_size, C)
```

**Output**

```
dW : (D, C)
db : (C,)
```

**Formula**

```
dZ = Y_pred - Y_true
dW = X.T @ dZ / batch_size
db = mean(dZ)
```

---

## 9. Update Parameters

```python
def update_parameters(
    W: np.ndarray,
    b: np.ndarray,
    dW: np.ndarray,
    db: np.ndarray,
    learning_rate: float
) -> tuple[np.ndarray, np.ndarray]:
```

**Update Rule**

```
W = W - lr * dW
b = b - lr * db
```

---

## 10. Create Mini-Batches (Batch SGD)

```python
def create_mini_batches(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int
) -> list[tuple[np.ndarray, np.ndarray]]:
```

**Output**

```
[
 (X_batch, Y_batch),
 ...
]
```

---

# 3. Training Function (Multiple Epochs)

```python
def train_model(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float
) -> tuple[np.ndarray, np.ndarray]:
```

**Steps**

```
1. initialize_parameters()
2. for epoch in epochs:
       batches = create_mini_batches()

       for batch:
           P = forward_pass()
           loss = compute_loss()
           dW, db = backward_pass()
           W, b = update_parameters()
3. return W, b
```

**Output**

```
W : trained weights
b : trained bias
```

---

# 4. Prediction Function

```python
def predict(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
```

**Output**

```
predicted_labels : (N,)
```

**Operation**

```
P = forward_pass()
pred = argmax(P)
```

---

# 5. Accuracy Evaluation

```python
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
```

**Output**

```
accuracy : float
```

**Formula**

```
acc = mean(y_true == y_pred)
```

---

# 6. Save Model

```python
def save_model(W: np.ndarray, b: np.ndarray, path: str) -> None:
```

Example:

```
np.savez(path, W=W, b=b)
```

---

# 7. Load Model

```python
def load_model(path: str) -> tuple[np.ndarray, np.ndarray]:
```

Example:

```
data = np.load(path)
return data["W"], data["b"]
```

---

# 8. Main Training Script

Example usage:

```python
X_train, y_train, X_test, y_test = load_fashion_mnist("data")

X_train = normalize_features(X_train)
X_test = normalize_features(X_test)

Y_train = one_hot_encode(y_train, 10)

W, b = train_model(
    X_train,
    Y_train,
    epochs=20,
    batch_size=64,
    learning_rate=0.01
)

y_pred = predict(X_test, W, b)

print("Accuracy:", accuracy(y_test, y_pred))

save_model(W, b, "logreg_fashion_mnist.npz")
```

---

# 9. Expected Performance

Typical results:

```
Training epochs: 20
Batch size: 64
Accuracy: ~83–86%
```

(for pure logistic regression on Fashion-MNIST)

---

# 10. Optional Improvements

You can add:

* **L2 regularization**
* **learning rate scheduling**
* **momentum SGD**
* **gradient clipping**
* **early stopping**

---

✅ If you'd like, I can also show a **complete 150-line working implementation of logistic regression from scratch (NumPy only)** that trains on Fashion-MNIST.
