
# Test the parameter estimation methods 

from src.estimation_functions import *
from src.generate_data import * 


# Create Training & Testing Data
X_Train, Y_Train, X_Test, Y_Test = generate_linear_data(n_obs = 1000, n_covariates = 1, test_proportion = 0.2, logistic = True)

# =============================================================
# Ordinary Least Squares - Logistic Regression
logistic_reg = regression_estimator(X = X_Train, y = Y_Train)
logistic_reg.fit_Logistic(metric = "rmse")

Y_test_pred_ols_cf = logistic_reg.predict(X = X_Test)
logistic_reg.calc_error(Y_pred = Y_test_pred_ols_cf, Y_true = Y_Test)

logistic_reg.plot_fit()
logistic_reg.plot_residual_dist(true = Y_Test, pred = Y_test_pred_ols_cf)

