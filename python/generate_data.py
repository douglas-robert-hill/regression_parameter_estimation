
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
    Y = np.sum(X, axis = 1) + np.random.normal(loc = 0, scale = 0.5, size = n_obs)
    
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

        