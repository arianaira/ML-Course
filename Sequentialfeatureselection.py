import numpy as np
from sklearn import model_selection
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


"""
backward feature selection:
first we calc mse for original set with n features
then we calc mse for each n - 1 subset and choose the best of ot
then we calc mse for each n - 2 subset of the previous best mse
we continue until there is 3 features left
in each step if the best mse of subsets is not beter that the original we terminate
"""

def backward_fs(X, y, estimator):
    model = Ridge(alpha=0.01)
    if estimator == "l1":
        model = Lasso(alpha=0.01)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40)
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test)
    best_mse = mean_squared_error(y_pred, y_test)
    features_size = X.shape[1]
    selected_features = list(range(X.shape[1]))
    
    for j in range(3, features_size): 
        mses = []
        for i in range(features_size - 1):
            s_f_temp = selected_features.copy()
            s_f_temp = np.delete(s_f_temp, i)
            X_temp = X[:, s_f_temp].copy()
            X_train, X_test, y_train, y_test = train_test_split(X_temp, y, test_size=0.30, random_state=40)
            model.fit(X_train, y_train) 
            y_pred = model.predict(X_test)
            mses.append(mean_squared_error(y_pred, y_test))

        temp_mse = np.mean(mses)
        if temp_mse < best_mse:
            best_mse = temp_mse
            index = np.argmin(mses)
            selected_features = np.delete(selected_features, index)
            features_size = features_size - 1
        else:
            break
    return X[:,selected_features]



"""
forward feature selection:
first we calc mse for set with only two features
then we add each feature to calc size of 3 subsets and we choose best mse
if best mse is better than previous step we add another feature and create 4 size features , 5 and ...
we continue until we reach original set
in each step if the best mse is not better thant previous, we terminate and return the previous
"""

def forward_fs(X, y, estimator):
    model = Lasso(alpha=0.01)
    if estimator == "l2":
        model = Ridge(alpha=0.01)
    selected_features = [0, 1]
    features_size = X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X[:, selected_features], y, test_size=0.30, random_state=40)
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test)
    best_mse = mean_squared_error(y_pred, y_test)
    
    for j in range(3, features_size): 
        mses = []
        for i in range(j, features_size):
            if i not in selected_features:
                temp = selected_features.copy()
                temp.append(i)
                X_temp = X[:,temp]
                X_train, X_test, y_train, y_test = train_test_split(X_temp, y, test_size=0.30, random_state=40)
                model.fit(X_train, y_train) 
                y_pred = model.predict(X_test)
                mses.append(mean_squared_error(y_pred, y_test))
        temp_mse = np.min(mses)
        if temp_mse < best_mse:
            best_mse = temp_mse
            index = np.argmin(mses)
            print(index)
            selected_features.append(index)
        else:
            break
    return X[:,selected_features]