def backward_fs(X, y, estimator):
    model = Lasso(alpha=0.001)
    if estimator == "l2":
        model = Ridge(alpha=0.001)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test)
    best_mse = mean_squared_error(y_pred, y_test)
    features_size = X.shape[1]
    selected_features = list(range(X.shape[1]))
    
    for j in range(3, features_size): 
        mses = []
        for i in range(features_size - j):
            temp = selected_features[i]
            X_temp = np.delete(X[:,selected_features], i, axis=1)
            X_train, X_test, y_train, y_test = train_test_split(X_temp, y, test_size=0.30, random_state=40)
            model.fit(X_train, y_train) 
            y_pred = model.predict(X_test)
            mses.append(mean_squared_error(y_pred, y_test))
            selected_features = np.insert(selected_features, i, temp)

        temp_mse = np.mean(mses)
        if temp_mse < best_mse:
            best_mse = temp_mse
            index = np.argmin(mses)
            selected_features.pop(index)
        else:
            break
    return X[:,selected_features]


def forward_fs(X, y, estimator):
    model = Lasso(alpha=0.01)
    if estimator == "l2":
        model = Ridge(alpha=0.01)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test)
    best_mse = mse(y_pred, y)
    selected_features = []
    features_size = X.shape[1]
    
    for j in range(3, features_size): 
        mses = []
        for i in range(features_size):
            if i not in selected_features:
                temp = selected_features.copy()
                temp.append(i)
                X_temp = X[:,temp]
                X_train, X_test, y_train, y_test = train_test_split(X_temp, y, test_size=0.30, random_state=40)
                model.fit(X_train, y_train) 
                y_pred = model.predict(X_test)
                mses.append(mse(y_pred, y))
        temp_mse = np.min(mses)
        if temp_mse < best_mse:
            best_mse = temp_mse
            index = np.argmin(mses)
            selected_features.append(index)
        else:
            break
    return X[:,selected_features]