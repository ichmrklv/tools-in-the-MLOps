import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import logging
import argparse
from mlflow.models import infer_signature


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

df = pd.read_csv('C:/Users/HUAWEI/PycharmProjects/clearml/src/data/housing.csv')
print(f'df shape:{df.shape}')

y = df.Y
X = df.drop(['Y'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

class LinearRegressionSGD():
    def __init__(self, lr=0.01, max_iter=1000, batch_size=32, tol=1e-3, intercept = True):
      self.lr = lr
      self.max_iter = max_iter
      self.batch_size = batch_size
      self.tol  = tol
      self.intercept = intercept
      self.theta = None
      self.n = None   # num_samples
      self.d = None   # num_features

    def fit(self, X, y):
      self.X = X.copy()
      self.y = y.copy()
      if self.intercept:
        self.X = np.hstack((np.ones((self.X.shape[0], 1)), self.X))

      self.n, self.d = self.X.shape
      self.theta = np.random.randn(self.d) # weights

      steps, errors = [], []
      step = 0

      for _ in range(self.max_iter):
        # перемешивание данных и разбивка на мини-батчи
        indices = np.random.permutation(self.n)
        #indices = np.random.choice(self.n, size=self.batch_size)

        for i in range(0, self.n, self.batch_size):
          batch_indices = indices[i:i + self.batch_size]
          X_batch = self.X[batch_indices]
          y_batch = self.y[batch_indices]

          # Вычисление градиента и обновление параметров
          grad = self.gradient(X_batch, y_batch)
          self.theta -= self.lr * grad

        new_error = ((self.y - self.X @ self.theta).T @ (self.y - self.X @ self.theta)) / self.n
        step += 1
        steps.append(step)
        errors.append(new_error)

        # проверка сходимости
        if np.linalg.norm(grad) < self.tol:
          break
      return steps, errors


    def gradient(self, X, y):
      return X.T @ (X @ self.theta - y) / len(y)

    def predict(self, X):
      if self.intercept:
        X_ = np.hstack((np.ones((X.shape[0],1)), X))
      else:
        X_ = X
      return X_ @ self.theta

    def MSE(self, X, y):
      return ((y - self.predict(X)).T @ (y - self.predict(X))) / len(y)

    def MAE(self, X, y):
      return abs(y - self.predict(X)).mean()

    def MAPE(self, X, y):
      return abs((y - self.predict(X))/y).mean()

    def r2_score(self, X, y):
      y_pred = self.predict(X)
      ss_total = ((y - y.mean())**2).sum()
      ss_residual = ((y - y_pred)**2).sum()
      r2 = 1 - (ss_residual / ss_total)
      return r2


# Define Parameters
params = {
    "lr": 0.1,
    "max_iter": 150,
    "batch_size": 16,
    "tol": 1e-6,
    "intercept": True
}

if 'intercept' in params and params['intercept']:
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

task.connect(params)

# Train Model
start_time = time.time()
modelSGD = LinearRegressionSGD(**params)
steps, errors = modelSGD.fit(X_train, y_train)
t_training = time.time() - start_time
print("Время обучения:", t_training)

for step, error in zip(steps, errors):
    print(f"Iteration {step}: MSE_test: {error}")

#modelSGD.save_model("LinRegrSGD")

# Predict and metrics
y_pred = modelSGD.predict(X_test)
print(f'MSE_test: {modelSGD.MSE(X_test, y_test)}')
print(f'MAPE_test: {modelSGD.MAPE(X_test, y_test)}')
print(f'MAE_test: {modelSGD.MAE(X_test, y_test)}')
print(f'R2: {modelSGD.r2_score(X_test, y_test)}')
#print("Коэффициенты: ", modelSGD.theta)

# Plot
fig, ax = plt.subplots(figsize = (7,3))
plt.plot(steps, errors)
plt.title('SGD for housing')
ax.set_xlabel("step")
ax.set_ylabel("error")
plt.show();