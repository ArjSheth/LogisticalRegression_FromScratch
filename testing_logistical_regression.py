import multinomial_regression_classifier as mlr
from torchvision import datasets
import matplotlib.pyplot as plt



train = datasets.FashionMNIST(root="data", train=True, download=True)
test  = datasets.FashionMNIST(root="data", train=False, download=True)

X_train = train.data.numpy()
y_train = train.targets.numpy()

X_test = test.data.numpy()
y_test = test.targets.numpy()

X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

X_train = X_train / 255.0
X_test = X_test / 255.0




theta = mlr.softmax_sgd(X_train, y_train, iterations=2000, batch_size=128, learning_rate=0.1)
y_prediction = mlr.predict_softmax(X_test, theta)



i = 0
plt.imshow(X_test[i,:].reshape(28,28), cmap="gray")
print("Prediction:", y_prediction[i])
print("True label:", y_test[i])