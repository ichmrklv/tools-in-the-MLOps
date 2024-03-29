import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score

from clearml import Task


# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name='ClearML_Merkulova', task_name='XGBoost SimpleSamp', output_uri=True)

# Read the data
fashion_mnist_test = pd.read_csv("data/fashion-mnist_test.csv", sep=',')
fashion_mnist_train = pd.read_csv("data/fashion-mnist_train.csv", sep=',')

# Load in the train and test sets
X_train = np.array(fashion_mnist_train.iloc[:,1:])
y_train = np.array(fashion_mnist_train.iloc[:,0])
X_test = np.array(fashion_mnist_test.iloc[:,1:])
y_test = np.array(fashion_mnist_test.iloc[:,0])

# Plot one of them to make sure everything is alright
plt.imshow(X_train[1].reshape((28, 28)))
plt.title("Sample Image")
plt.show()

# Load the data into XGBoost format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set the parameters
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 5,
    "eta": 0.6,
    "gamma": 0,
    "max_delta_step": 1,
    "subsample": 1,
    "sampling_method": "uniform",
    "seed": 1337
}
task.connect(params)

# Train the XGBoost Model
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=25,
    evals=[(dtrain, "train"), (dtest, "test")],
    verbose_eval=0,
)


# Save the model
# bst.save_model("best_model")

# Make predictions for test data
y_pred = bst.predict(dtest)
predictions = [round(value) for value in y_pred]

# Evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
