import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import string

# File shared data
day_code = {}
month_code = {}
year_code = {}


# *****Clean Data Section****** #


def numeric_checker(column_data):
    for i, ele in enumerate(column_data):
        try:
            float(ele)
        except ValueError:
            print(i, ele)


def content_rate_checker(column_data):
    for i, ele in enumerate(column_data):
        if ele not in ['Adults only 18+', 'Unrated', 'Teen',
                       'Everyone 10+', 'Mature 17+',
                       'Everyone']:
            print(i)


def trim(data_frame, func, cols=[]):
    func = list(func)
    iterator = 0
    for col in data_frame.columns if len(cols) == 0 else cols:
        data_frame.loc[:, col] = data_frame.loc[:, col].apply(func[iterator % len(func)])
        iterator += 1


def remove_commas(ele):
    if ',' in str(ele):
        ele = str(ele).replace(',', '')
    return ele


def trim_install(ele):
    if ele[-1] == '+':
        return int(ele[:-1])
    return int(ele)


def trim_size(ele):
    if str(ele)[-1] == 'M':
        return float(ele[:-1]) * (2 ** 20)
    elif str(ele)[-1] == 'k':
        return float(ele[:-1]) * (2 ** 10)


def trim_price(ele):
    ele = str(ele).replace('$', '')
    return float(ele)


def trim_min_ver(ele):
    return ele if ele != 'Varies with device' else np.nan


def date_logical_encoding(column_data):
    global month_code
    global day_code
    global year_code
    month_code = {'Jan': 'A',
                  'Feb': 'B',
                  'Mar': 'C',
                  'Apr': 'D',
                  'May': 'E',
                  'Jun': 'F',
                  'Jul': 'G',
                  'Aug': 'H',
                  'Sep': 'I',
                  'Oct': 'J',
                  'Nov': 'K',
                  'Dec': 'L'}
    day = set()
    year = set()
    for ele in column_data:
        arr = str(ele).split('-')
        day.add(int(arr[0]))
        year.add(int(arr[2]))
    code = list(string.ascii_uppercase) + list(string.ascii_lowercase)
    day_code = {str(ele): code[i] for i, ele in enumerate(list(day))}
    year_code = {str(ele): code[i] for i, ele in enumerate(list(year))}


def trim_date(ele):
    arr = str(ele).split('-')
    return year_code[arr[2]] + month_code[arr[1]] + day_code[arr[0]]


def trim_data_set(data):
    trim(data, [remove_commas])
    trim(data, [trim_min_ver])
    trim(data, [trim_install, trim_size, trim_price, trim_date], cols=['Installs', 'Size', 'Price', 'Last Updated'])


# *****Missing Value Section****** #


def impute_data(data, strategies, cols_idx=[]):
    strategies = list(strategies)
    iterator = 0
    for col in cols_idx:
        imp = SimpleImputer(missing_values=np.nan, strategy=strategies[iterator % len(strategies)])
        imp = imp.fit(data[:, col:(col + 1)])
        data[:, col:(col + 1)] = imp.transform(data[:, col:(col + 1)])
        iterator += 1


# *****Encode Data Section****** #


def data_label_encoder(data, cols_idx):
    encode = LabelEncoder()
    for col in cols_idx:
        data[:, col] = encode.fit_transform(data[:, col])


def data_hot_encoder(data, cols_idx):
    data = data.astype(np.float64)
    encode = OneHotEncoder(categorical_features=[cols_idx])
    data = encode.fit_transform(data).toarray()
    return data


def freq_encoder(data, col_idx=[]):
    for col in col_idx:
        s = set(data[:, col])
        dic = {}
        for ele in s:
            dic[ele] = 0
        for ele in data[:, col]:
            dic[ele] += 1
        total = sum(dic.values())
        for i, val in enumerate(data[:, col]):
            data[i, col] = dic[val] / total


def Pca(data, noOfSelctedFeatures):
    # data = run()
    X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.2,
                                                        random_state=0)  # , shuffle=False)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    pca = PCA(n_components=noOfSelctedFeatures)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # X_train = pca.fit_transform(X_train)
    # print(f'Ratio {pca.explained_variance_ratio_}')
    return [X_train, X_test, y_train, y_test]


def run(test_data):
    #data = pd.read_csv('D:\\Studying\\4th Year 1st Term\\Machine Learning\\Project\\Final Project\\MLProjectFinal\DataSets\\Mobile_App_Success_Milestone_2.csv')
    data = test_data
    data = data.iloc[:, 0:11]
    data.dropna(axis=0, how='any', thresh=9, subset=None, inplace=True)
    # drop noisy data
    #data.drop([6941, 12624, 18477], inplace=True)
    # print(data.shape)
    cols = data.columns.tolist()
    # cols = cols[0:2] + cols[3:] + cols[2:3]
    data = data[cols]

    # encode dates logically...
    date_logical_encoding(data['Last Updated'].values)
    # trim data frame values
    trim_data_set(data)
    # change data frame to np array to use sklearn preprocessing
    data = data.values
    # impute missing values
    impute_data(data, ['most_frequent', 'mean', 'mean', 'mean', 'mean', 'most_frequent', 'most_frequent',
                       'most_frequent'],
                [1, 2, 3, 4, 5, 6, 8, 9])
    # encode data
    data_label_encoder(data, [0, 1, 6, 7, 8, 9, 10])
    # label_encoder(data, [10])
    # print(set(data[:, 10]))
    np.delete(data, 8, axis=1)
    return data