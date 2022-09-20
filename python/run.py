
import matplotlib.pyplot as plt 
from estimation_functions import *

# Create Training & Testing Data
random_data = generate_linear_data(n_obs = 1000, n_covariates = 10, test_proportion = 0.2)
X_train = random_data[["X_Train"]]
y_train = random_data[["Y_Train"]]
X_test = random_data[["X_Test"]]
y_test = random_data[["Y_Test"]]

# Train and Predict with Ordinary Least Squares
ols = train(X = X_train, y = y_train, method = "OLS")
ols_pred = predict(model = ols, newdata = X_test)

# Train and Predict with Ridge Regression
ridge = train(X = X_train, y = y_train, method = "RIDGE")
ridge_pred = predict(model = ols, newdata = X_test)

# Train and Predict with Maximum Likelihood Estimation
mle = train(X = X_train, y = y_train, method = "MLE")
mle_pred = predict(model = ols, newdata = X_test)

# Train and Predict with Bayesian Linear Regression
bayes = train(X = X_train, y = y_train, method = "BAYES")
bayes_pred = predict(model = bayes, newdata = X_test)

# Plot Fitted Training Regression 
