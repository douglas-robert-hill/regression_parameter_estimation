
import matplotlib.pyplot as plt 
from estimation_functions import *
from generate_data import * 


# Create Training & Testing Data
X_Train, Y_Train, X_Test, Y_Test = generate_linear_data(n_obs = 1000, n_covariates = 1, test_proportion = 0.2)

# =============================================================
# Train and Predict with Ordinary Least Squares - Closed Form 
ols_reg_cf = linear_regression(X = X_Train, y = Y_Train)
ols_reg_cf.fit_OLS(metric = "rmse", closed_form = True)
Y_pred_ols_cf = ols_reg_cf.predict(X = X_Train)

Y_test_pred_ols_cf = ols_reg_cf.predict(X = X_Test)
ols_reg_cf.calc_error(Y_pred = Y_test_pred_ols_cf, Y_true = Y_Test)

plt.scatter(X_Train, Y_Train)
plt.plot(X_Train, Y_pred_ols_cf, '-')
plt.title(label = "Ordinary Least Squares Closed Form Solution")
plt.show() 


# =============================================================
# Train and Predict with Ridge Regression - Closed Form 
ridge_reg_cf = linear_regression(X = X_Train, y = Y_Train)
ridge_reg_cf.fit_Ridge(metric = "rmse", closed_form = True)
Y_pred_rr_cf = ridge_reg_cf.predict(X = X_Train)

Y_test_pred_rr_cf = ridge_reg_cf.predict(X = X_Test)
ridge_reg_cf.calc_error(Y_pred = Y_test_pred_ols_cf, Y_true = Y_Test)

plt.scatter(X_Train, Y_Train)
plt.plot(X_Train, Y_pred_rr_cf, '-')
plt.title(label = "Ridge Regression Closed Form Solution")
plt.show() 


# =============================================================
# Train and Predict with Ordinary Least Squares
ols_reg = linear_regression(X = X_Train, y = Y_Train)
ols_reg.fit_OLS(metric = "rmse", closed_form = False)
Y_pred_ols = ols_reg.predict(X = X_Test)


# =============================================================
# Train and Predict with Ridge Regression
ridge_reg = linear_regression(X = X_Train, y = Y_Train)
ridge_reg.fit_Ridge(metric = "", closed_form = False)
Y_pred_rr = ridge_reg.predict(X = X_Test)


# =============================================================
# Train and Predict with Maximum Likelihood Estimation
mle_reg = linear_regression(X = X_Train, y = Y_Train)
mle_reg.fit_MLE(metric = "")
Y_pred_mle = mle_reg.predict(X = X_Test)


# =============================================================
# Train and Predict with Bayesian Linear Regression
bayes_reg = linear_regression(X = X_Train, y = Y_Train)
bayes_reg.fit_Bayes(metric = "")
Y_pred_bayes = bayes_reg.predict(X = X_Test)


