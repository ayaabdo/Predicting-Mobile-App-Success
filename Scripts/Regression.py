import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


# Multi-variable Linear Regression
def multi_variant_regression(data, scaled=False):
    arr = np.copy(data)
    scaled_data = arr
    if scaled is False:
        scale = StandardScaler()
        scale.fit(arr[:, :-1])
        scaled_data = scale.fit_transform(arr[:, :-1])
        scaled_data = np.append(scaled_data, np.reshape(data[:, -1], (len(data[:, -1]), 1)), axis=1)
    #scaled_data = select_features(scaled_data)
    [x_train, x_test, y_train, y_test] = train_test_split(scaled_data[:, :-1], scaled_data[:, -1], test_size=0.2,
                                                          random_state=0)

    regression = LinearRegression()
    regression.fit(x_train, y_train)
    return [regression,model_evaluation(regression, x_test, y_test)]


def polynomial_regression(data, poly_deg):
    arr = np.copy(data)
    scale = StandardScaler()
    scale.fit(arr[:, :-1])
    scaled_data = scale.fit_transform(arr[:, :-1])
    scaled_data = np.append(scaled_data, np.reshape(data[:, -1], (len(data[:, -1]), 1)), axis=1)
    features_generator = PolynomialFeatures(degree=poly_deg)
    features = features_generator.fit_transform(scaled_data[:, :-1])
    arr = np.append(features[:, 1:], np.reshape(arr[:, -1], (len(arr[:, -1]), 1)), axis=1)

    return multi_variant_regression(arr, scaled=True)


# R-Squared Method
def calc_r2(y_act, y_predict):
    SSres =sum((y_predict - y_act)**2)
    mean = np.mean(y_act)
    SStot = sum((y_act - mean)**2)
    return 1-(SSres / SStot)


# automatic features selection section

# Backward Elimination
def backward_elimination_regression(data, sl=0.05):
    features = np.copy(data[:, :-1])
    y = np.copy(data[:, -1])
    features = np.append(np.ones((len(features), 1), dtype=np.float64), features, axis=1)
    [x_train, x_test, y_train, y_test] = train_test_split(features, y, test_size=0.2, random_state=0)
    regression = sm.OLS(endog=y_train, exog=x_train).fit()
    while max(regression.pvalues) > sl:
        mx = max(regression.pvalues)
        col_idx = [idx for idx, val in enumerate(regression.pvalues) if val == mx][0]
        x_train = np.delete(x_train, col_idx, axis=1)
        x_test = np.delete(x_test, col_idx, axis=1)
        regression = sm.OLS(endog=y_train, exog=x_train).fit()
    return model_evaluation(regression, x_test, y_test)


def model_evaluation(model, x_test, y_test):
    y_predict = model.predict(x_test)
    for i, val in enumerate(y_predict):
        if val > 5:
            y_predict[i] = 5
        elif val < 0:
            y_predict[i] = 0
    SSE = sum((y_predict-y_test)**2)
    MSE = SSE / len(y_test)
    r2 = calc_r2(y_test, y_predict)
    return [r2, SSE, MSE, model, y_test, y_predict]

de = []
# Correlation Matrix
def select_features(data, threshold=0.006):
    global de
    df = pd.DataFrame(data)
    corrmat = df.corr()
    lis = [i for i, x in enumerate(corrmat[len(data[0]) - 1].values) if abs(x) >= threshold]
    # saving deleted features according to correlation matrix
    de = [i for i in range(0, len(data[0]) - 1) if i not in lis]
    ret = np.copy(data)
    ret = np.delete(ret, de, axis=1)
    return ret
