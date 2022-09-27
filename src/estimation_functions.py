# To Do
#   Calculate derivatives of error functions & add derivative functions to __gradientDescent
#   Test OLS & Ridge
#   Implement MLE
#   Implement bayes 
#   Calc p-value 
#   complete plotting functions for multi-dimensional data 
import numpy as np
import matplotlib.pyplot as plt 

class linear_regression():

    def __init__(self, X: np.array, y: np.array, verbose: bool = True) -> None:
        """
        Initialise a regression model class by passing train and test data.

        param X : set of covariates
        param y : response variable corresponding with X 
        param verbose : whether to report progress
        """
        if len(X) != len(y):
            raise ValueError("Length of X and Y do not match.")

        if len(y[0]) != 1:
            raise ValueError("Y is not a One-Dimensional vector.")

        if np.count_nonzero(np.isnan(y)) > 0:
            raise ValueError("Missing values found in Y.")
            
        if np.count_nonzero(np.isnan(X)) > 0:
            raise ValueError("Missing values found in X.")

        self.X = X
        self.y = y
        self.y_hat = np.zeros(len(self.y))
        self.weight = np.zeros(len(X[0]))
        self.bias = 0 
        self.error_history = [] 
        self.verbose = verbose
        self.closed_form = False
        self.precision_value = 0.0001


    def predict(self, X: np.array) -> np.array:
        """
        Predict response based on current weights and bias

        param X : covariates for prediction 

        return : array of predicted responses 
        """
        return np.sum(X * self.weight, axis = 1) + self.bias


    def fit_OLS(self, metric: str, closed_form: bool, max_iter: int = 100, lr: float = 0.01) -> None:
        """
        Ordinary Least Square parameter estimation.
        Use defined metric with gradient descent unless only one covariate then use closed form solution.
        Due to computation of inverse a high dimensional feature closed form is only used with one covariate.

        param metric : cost function  
        param closed_form : boolean on whether to use closed form solution
        param max_iter : number of iterations for gradient descent 
        param lr : learning rate  
        """
        self.metric = metric

        if self.metric not in ["rmse", "mse", "r2", "mae"]:
            raise ValueError("Invalid metric. Please try one of: 'rmse', 'mse', 'r2' or 'mae'.")

        if closed_form:
            self.__fitOLS_CF()
        else:
            self.epochs = max_iter
            self.lr = lr
            for iter in range(self.epochs):

                # Perform gradient descent 
                y_pred = self.predict(X = self.X)
                self.__gradientDescent(y_pred = y_pred)
                
                # Record error 
                err = self.calc_error(Y_pred = y_pred)
                self.error_history.append(err)
                if self.verbose:
                    print("> epoch=",iter,"; error=",format(err, '.4f'))
                
                # Check for convergence 
                if self.lr == 1000:
                    break 

        self.__update_post_train(method = "OLS") 
        self.__print_final_err()


    def fit_Ridge(self, metric: str, closed_form: bool, max_iter: int = 100, lr: float = 0.01, L2: float = 0.1) -> None:
        """
        Ridge Regression parameter estimation.
        Use defined metric with gradient descent unless only one covariate then use closed form solution.
        Due to computation of inverse a high dimensional feature closed form is only used with one covariate.

        param metric : cost function  
        param closed_form : boolean on whether to use closed form solution 
        """
        self.metric = metric
        self.alpha = L2

        if self.metric not in ["rmse", "mse", "r2", "mae"]:
            raise ValueError("Invalid metric. Please try one of: 'rmse', 'mse', 'r2' or 'mae'.")

        if self.alpha < 0 or self.alpha > 1:
            raise ValueError("L2 parameter is out of bounds. Value must be within range of 0-1.")

        if closed_form:
            self.__fitRIDGE_CF()
        else:
            self.epochs = max_iter
            self.lr = lr
            for iter in range(self.epochs):
                
                # Perform gradient descent 
                y_pred = self.predict(X = self.X)
                self.__gradientDescent(y_pred = y_pred)
                
                # Record error 
                err = self.calc_error(Y_pred = y_pred)
                self.error_history.append(err)
                if self.verbose:
                    print("> epoch=",iter,"; error=",format(err, '.4f'))

                # Check for convergence 
                if iter == 1: 
                    past_err = err
                else:
                    chng_cost = err - past_err
                    past_err = err
                    if self.precision_value > chng_cost: break 

        self.__update_post_train(method = "Ridge")
        self.__print_final_err()

    
    def fit_Bayes(self) -> None:
        """
        Bayesian Linear Regression implementation 
        """
        self.__update_post_train(method = "Bayes")
        self.__print_final_err()


    def fit_MLE(self) -> None:
        """
        Maximum Likelihood Parameter Estimation using the Expectation-Maximisation algorithm
        """

        self.__update_post_train(method = "MLE")
        self.__print_final_err()


    def __gradientDescent(self, y_pred: np.array) -> None:
        """
        Perform gradient descent on specified metric.

        param y_pred : predicted values
        param metric : method for evaluating error 
        """
        # Calculate cost function 
        cost_W, cost_B = self.__calc_error_deriv(Y_pred = y_pred)

        # Find Weight & Bias Gradients 
        if hasattr(self, 'alpha'):
            step_size_W = self.lr * (cost_W + self.alpha)
            step_size_B = self.lr * (cost_B + self.alpha)
        else:
            step_size_W = self.lr * cost_W
            step_size_B = self.lr * cost_B

        # Update Weight 
        self.weight = self.weight - step_size_W
        self.bias = self.bias - step_size_B


    def __fitOLS_CF(self) -> None:
        """
        Ordinary Least Square parameter estimation through the closed form solution
        """
        self.closed_form = True

        # Add column for bias estimation 
        intercept_array = np.ones((len(self.X), 1))
        X_Train_CF = np.concatenate([intercept_array, self.X], axis = 1)

        # Run closed form solution 
        inv_X = np.linalg.inv(np.matmul(np.transpose(X_Train_CF), X_Train_CF))
        transformed_X = np.matmul(inv_X, np.transpose(X_Train_CF))
        betas = np.matmul(transformed_X, self.y)

        # Update Weight & Biases 
        self.weight = betas[1:]
        self.bias = betas[0]


    def __fitRIDGE_CF(self) -> None:
        """
        Ridge parameter estimation through the closed form solution

        param alpha : L2 regularisation parameter
        """
        self.closed_form = True

        # Add column for bias estimation 
        intercept_array = np.ones((len(self.X), 1))
        X_Train_CF = np.concatenate([intercept_array, self.X], axis = 1)

        # Run closed form solution 
        identity_mat = np.identity(X_Train_CF.shape[1])
        inv_X = np.linalg.inv(np.matmul(np.transpose(X_Train_CF), X_Train_CF) + (self.alpha * identity_mat))        
        betas = np.matmul(np.matmul(inv_X, np.transpose(X_Train_CF)), self.y)

        # Update Weight & Biases 
        self.weight = betas[1:]
        self.bias = betas[0]


    def calc_error(self, Y_pred: np.array, Y_true: np.array = None) -> float:
        """
        Calculate error between predicted and true value by some measure.

        param Y_pred : predicted values
        param Y_true : true values 

        return err : calculated error
        """
        if Y_true is None:
            Y_true = self.y

        if self.metric == "mse":
            return (1/len(Y_pred)) * np.sum((Y_pred - Y_true)**2)
        elif self.metric == "rmse":
            return np.sqrt((1/len(Y_pred)) * np.sum((Y_pred - Y_true)**2))
        elif self.metric == "r2":
            return (1 - (np.sum(Y_true - Y_pred)**2 / np.sum(Y_pred)**2))
        elif self.metric == "mae":
            return (1/len(Y_pred)) * np.sum(abs(Y_pred - Y_true))


    def __calc_error_deriv(self, Y_pred: np.array) -> float:
        """
        Calculate the derivavtive of the error function between predicted and true value by some measure.

        param Y_pred : predicted values

        return cost_W : derivative of error function with respect to W
        return cost_B : derivative of error function with respect to B
        """
        if self.metric == "mse":
            cost_W = (-2 * np.sum(np.dot((self.y - Y_pred), self.X))) / len(Y_pred)
            cost_B = (-2 * np.sum(self.y - Y_pred)) / len(Y_pred)
        elif self.metric == "rmse":
            cost_W = 0 
            cost_B = 0 
        elif self.metric == "r2":
            cost_W = 0 
            cost_B = 0 
        elif self.metric == "mae":
            cost_W = 0 
            cost_B = 0 

        return cost_W, cost_B


    def __update_post_train(self, method: str) -> None:
        """
        Update Internals and Calculate Stats

        param method : parameter estimation method 
        """
        self.trained = True
        self.method = method
        self.y_hat= self.predict(X = self.X)
        self.__calc_std_err()
        self.__run_t_test()
        self.__calc_p_value()

    
    def __calc_std_err(self) -> None:
        """
        Calculate coefficient standard errors 
        """
        self.se = np.std(a = self.X, axis = 0) / np.sqrt(len(self.X))


    def __run_t_test(self) -> None:
        """
        Run T-Test on coefficients 
        """
        self.t_test =  self.weight / self.se


    def __calc_p_value(self) -> None:
        """
        Calculate coefficient p-values
        """
        self.p_value =  self.t_test


    def __print_final_err(self) -> None:
        """
        Print to console the prediction error 
        """
        Y_pred = self.predict(X = self.X)
        err = self.calc_error(Y_pred = Y_pred)

        print("Parameter estimation complete.")
        print("Train data (X) error (", self.metric, "): ", err)


    def plot_error(self) -> None:
        """
        Plot the error over each iteration 
        """
        if self.closed_form is True:
            raise ValueError("Can't plot error for closed form solution.")

        plt.plot(range(len(self.error_history)), self.error_history, '-')
        plt.title(label = self.method + " Solution - Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.show() 


    def plot_fit(self, x_index: int = 0) -> None:
        """
        Plot the training data fit 

        param x_index : which covariate to plot 
        """
        if len(self.X[0]) == 1: x_index = 0

        plt.scatter(self.X[:, x_index], self.y)
        plt.plot(self.X[:, x_index], self.y_hat, '-', col = "red")
        plt.title(label = self.method + " Solution Fit")
        plt.show() 


    def plot_residual_dist(self, true: np.array, pred: np.array) -> None:
        """
        Plot the distribution of residuals

        param true : true y values
        param pred : predicted y values
        """
        residuals = true - pred

        plt.hist(residuals, bins = 20)
        plt.title(label = self.method + " Residual Distribution")
        plt.show()
    
