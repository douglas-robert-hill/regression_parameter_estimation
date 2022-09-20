
import numpy as np
import random 

def generate_linear_data(n_obs: int, n_covariates: int, test_proportion: float = 0):
    """
    Generate random dataset with a linear relation.

    param n_obs : number of observations
    param n_covariates : number of covariates in data
    param test_proportion : proportion of data for test dataset

    return X : multi-dimensional array of covariates
    return Y : one dimensional array of target variable
    """

    generator = np.random.mtrand._rand
    X = generator.standard_normal(size = (n_obs, n_covariates))
    Y = np.sum(np.dot(X, np.random.normal()), axis = 1) + np.random.normal()

    if test_proportion == 0:
        return X, Y

    else:
        
        test_index = random.sample(population = list(range(len(X))), k = int(n_obs * test_proportion))
        test_mask = np.ones(len(X), dtype = bool)
        test_mask[test_index] = False

        X_Train = X[test_mask]
        Y_Train = Y[test_mask]
        X_Test = X[test_index]
        Y_Test = Y[test_index]

        return X_Train, Y_Train, X_Test, Y_Test


def train(X: np.array, y: np.array, method: str):
    """
    Train a regression model based on specified parameter estimation method.

    param X : set of covariates
    param y : response variable corresponding with X 
    param method : method of parameter estimation 

    return 
    """

    # Error Checks 
    valid_methods = ["OLS", "BAYES", "MLE", "RIDGE"]
    if method not in valid_methods:
        raise ValueError("Invalid method. Try: 'OLS', 'BAYES', 'MLE', 'RIDGE'")

    if len(X) != len(y):
        raise ValueError("Length of X and Y do not match.")

    # Call Estimation Method 
    if method == "OLS":
        params = fn_OLS(X = X, y = y)
    elif method == "BAYES":
        params = fn_BAYES(X = X, y = y)
    elif method == "MLE":
        params = fn_MLE(X = X, y = y)
    elif method == "RIDGE":
        params = fn_RIDGE(X = X, y = y)

    # Return 
    return params



def predict(model, newdata: np.array) -> np.array:
    """
    Make predictions based on the trained linear regression model and new data.

    param model : output of ()
    param newdata : covariates for prediction 

    return pred : one dimensional array of predictions 
    """
    pass

