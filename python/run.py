
# Test the parameter estimation methods 

from estimation_functions import *
from generate_data import * 


# Create Training & Testing Data
X_Train, Y_Train, X_Test, Y_Test = generate_linear_data(n_obs = 1000, n_covariates = 2, test_proportion = 0.2)


# =============================================================
# Ordinary Least Squares - Closed Form 
ols_reg_cf = linear_regression(X = X_Train, y = Y_Train)
ols_reg_cf.fit_OLS(metric = "rmse", closed_form = True)

Y_test_pred_ols_cf = ols_reg_cf.predict(X = X_Test)
ols_reg_cf.calc_error(Y_pred = Y_test_pred_ols_cf, Y_true = Y_Test)

ols_reg_cf.plot_fit()
ols_reg_cf.plot_residual_dist(true = Y_Test, pred = Y_test_pred_ols_cf)


# =============================================================
# Ridge Regression - Closed Form 
ridge_reg_cf = linear_regression(X = X_Train, y = Y_Train)
ridge_reg_cf.fit_Ridge(metric = "rmse", closed_form = True)

Y_test_pred_rr_cf = ridge_reg_cf.predict(X = X_Test)
ridge_reg_cf.calc_error(Y_pred = Y_test_pred_rr_cf, Y_true = Y_Test)

ridge_reg_cf.plot_fit()
ridge_reg_cf.plot_residual_dist(true = Y_Test, pred = Y_test_pred_rr_cf)


# =============================================================
# Ordinary Least Squares
ols_reg = linear_regression(X = X_Train, y = Y_Train)
ols_reg.fit_OLS(metric = "mse", closed_form = False, max_iter = 100, lr = 0.05)

Y_test_pred_ols = ols_reg.predict(X = X_Test)
ols_reg.calc_error(Y_pred = Y_test_pred_ols, Y_true = Y_Test)

ols_reg.plot_fit(x_index = 1)
ols_reg.plot_error()
ols_reg.plot_residual_dist(true = Y_Test, pred = Y_test_pred_ols)


# =============================================================
# Ridge Regression
ridge_reg = linear_regression(X = X_Train, y = Y_Train)
ridge_reg.fit_Ridge(metric = "mse", closed_form = False, L2 = 0.1)

Y_test_pred_rr = ridge_reg.predict(X = X_Test)
ridge_reg.calc_error(Y_pred = Y_test_pred_rr, Y_true = Y_Test)

ridge_reg.plot_fit()
ridge_reg.plot_error()


# =============================================================
# Maximum Likelihood Estimation
mle_reg = linear_regression(X = X_Train, y = Y_Train)
mle_reg.fit_MLE(metric = "MLE")

Y_test_pred_mle = mle_reg.predict(X = X_Test)
mle_reg.calc_error(Y_pred = Y_test_pred_mle, Y_true = Y_Test)

mle_reg.plot_fit()
mle_reg.plot_error()


# =============================================================
# Bayesian Linear Regression
bayes_reg = linear_regression(X = X_Train, y = Y_Train)
bayes_reg.fit_Bayes(metric = "BAYES")

Y_test_pred_bayes = bayes_reg.predict(X = X_Test)
bayes_reg.calc_error(Y_pred = Y_test_pred_bayes, Y_true = Y_Test)

bayes_reg.plot_fit()
bayes_reg.plot_error()

