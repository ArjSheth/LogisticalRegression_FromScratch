from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import logistic_regression_classifier as lg

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Scaling input data to avoid floating point calculation errors.
X = (X - X.mean(axis=0)) / X.std(axis=0)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Logistic regression model
new_theta = lg.logit_sgd(X_train, y_train, 50, 50, 0.01)
for i in range (10) :
    print(lg.predicted_probability(new_theta, X_test[i]))
    print(lg.predicted_label(new_theta, X_test[i]), "      ", y_test[i])
    print()

# model = LogisticRegression(max_iter=5000)
# model.fit(X_train, y_train)

# Accuracy
# print("Accuracy:", model.score(X_test, y_test))
