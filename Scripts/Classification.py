import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn import svm, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from ttictoc import TicToc

models_name = ['Logistic' + '\n' + 'Regression', 'Poly' + '\n' + 'Logistic' + '\n' + 'd=2',
               'Poly' + '\n' + 'Logistic' + '\n' + 'd=3', 'Decision' + '\n' + 'Tree',
               ' AdaBoost' + '\n' + '(Decision' + '\n' + 'Tree)',
               'SVM' + '\n' + '  Gaussian' + '\n' + 'OVR', 'SVM' + '\n' + '  Gaussian' + '\n' + 'OVO', 'KNN']
AccuracyList = []
TrainingTime = []
TestingTime = []


def Logistic_Reg(data, pca=False):
    training_time = TicToc()
    if pca is False:
        arr = np.copy(data)
        scaled_data = arr
        scale = StandardScaler()
        scale.fit(arr[:, :-1])
        scaled_data = scale.fit_transform(arr[:, :-1])
        scaled_data = np.append(scaled_data, np.reshape(data[:, -1], (len(data[:, -1]), 1)), axis=1)
        [x_train, x_test, y_train, y_test] = train_test_split(scaled_data[:, :-1], scaled_data[:, -1], test_size=0.2,
                                                              random_state=0)#, shuffle=False)
        classifier = LogisticRegression(random_state=0, solver='newton-cg', multi_class='multinomial')
        # multinomial minimizes the loss ,fit across the entire prob distribution
        training_time.tic()
        classifier.fit(x_train, y_train)
        training_time.toc()
        #filename = 'LogisticRegression.sav'
        #joblib.dump(classifier, filename)

    else:
        [x_train, x_test, y_train, y_test] = data
        # x_train = pca.fit_transform(x_train)
        classifier = LogisticRegression(random_state=0, solver='newton-cg', multi_class='multinomial')
        # multinomial minimizes the loss ,fit across the entire prob distribution
        training_time.tic()
        classifier.fit(x_train, y_train)
        training_time.toc()
        #filename = 'LogisticRegressionPCA.sav'
        #joblib.dump(classifier, filename)

    return model_accuracy(classifier, x_test, y_test, training_time.elapsed)


def polynomial_classifier(data, poly_deg, pca=False):
    if pca is False:
        arr = np.copy(data)
        features_generator = PolynomialFeatures(degree=poly_deg)
        features = features_generator.fit_transform(arr[:, :-1])
        arr = np.append(features[:, 1:], np.reshape(arr[:, -1], (len(arr[:, -1]), 1)), axis=1)
        return Logistic_Reg(arr, False)
    else:
        X_train, X_test, y_train, y_test = data
        features_generator = PolynomialFeatures(degree=poly_deg)
        features = features_generator.fit_transform(X_train)
        testdata = features_generator.fit_transform(X_test)
        newFeatuers = []
        newFeatuers.append(features)
        newFeatuers.append(testdata)
        newFeatuers.append(y_train)
        newFeatuers.append(y_test)
        return Logistic_Reg(newFeatuers, True)


def Polynomial_Classifier_with_SelectiveFeature(data, poly_deg, pca=False):
    if pca is False:
        arr = np.copy(data)
        features_generator = PolynomialFeatures(degree=poly_deg)
        features = features_generator.fit_transform(arr[:, :-1])
        X_selected = Selective_feature_BestK(features[:, :-1], arr[:, -1])
        arr = np.append(features[:, 1:], np.reshape(arr[:, -1], (len(arr[:, -1]), 1)), axis=1)
        arr = np.append(X_selected[:, 1:], np.reshape(arr[:, -1], (len(arr[:, -1]), 1)), axis=1)
        return Logistic_Reg(arr, False)
    else:
        X_train, X_test, y_train, y_test = data
        features_generator = PolynomialFeatures(degree=poly_deg)
        features = features_generator.fit_transform(X_train)
        testdata = features_generator.fit_transform(X_test)
        newFeatuers = []
        newFeatuers.append(features)
        newFeatuers.append(testdata)
        newFeatuers.append(y_train)
        newFeatuers.append(y_test)
        return Logistic_Reg(newFeatuers, True)


def SVM_OneVsRest(data, pca=False):
    training_time = TicToc()
    if pca is False:
        [x_train, x_test, y_train, y_test] = train_test_split(data[:, :-1], data[:, -1], test_size=0.2, random_state=0)#, shuffle=False)
        training_time.tic()
        svm_model = OneVsRestClassifier(SVC(kernel='rbf', gamma=0.4, C=1)).fit(x_train, y_train)
        training_time.toc()
        #filename = 'SVMoneVSall.sav'
        #joblib.dump(svm_model, filename)

    else:
        [x_train, x_test, y_train, y_test] = data
        training_time.tic()
        svm_model = OneVsRestClassifier(SVC(kernel='rbf', gamma=0.4, C=1)).fit(x_train, y_train)
        training_time.toc()
        #filename = 'SVMoneVSallPCA.sav'
        #joblib.dump(svm_model, filename)

    # svm_model = OneVsRestClassifier(SVC(kernel='linear', C=1)).fit(x_train, y_train)
    # accuracy = svm_model.score(x_test, y_test)
    # print('One VS Rest SVM accuracy ( Gaussian Kernel): OLA ' + str(accuracy))

    return model_accuracy(svm_model, x_test, y_test, training_time.elapsed)


def SVM_OneVsOne(data, pca=False):
    training_time = TicToc()
    if pca is False:
        [x_train, x_test, y_train, y_test] = train_test_split(data[:, :-1], data[:, -1], test_size=0.2, random_state=0)#, shuffle=False)
        training_time.tic()
        svm_model = SVC(kernel='rbf', gamma=0.4, C=1).fit(x_train, y_train)
        # svm_model = SVC(kernel='linear', C=1).fit(x_train, y_train)
        training_time.toc()
        #filename = 'SVMoneVSone.sav'
        #joblib.dump(svm_model, filename)

    else:
        [x_train, x_test, y_train, y_test] = data
        training_time.tic()
        svm_model = SVC(kernel='rbf', gamma=0.4, C=1).fit(x_train, y_train)
        # svm_model = SVC(kernel='linear', C=1).fit(x_train, y_train)
        training_time.toc()
        #filename = 'SVMoneVSonePCA.sav'
        #joblib.dump(svm_model, filename)
    # accuracy = svm_model.score(x_test, y_test)
    # print('One VS One SVM accuracy (Gaussian Kernel): OLA ' + str(accuracy))
    # svm_model = SVC(kernel='linear', C=1).fit(x_train, y_train)

    return model_accuracy(svm_model, x_test, y_test, training_time.elapsed)


def SVM_withoutKernal(data, pca=False):
    training_time = TicToc()
    if pca is False:
        [x_train, x_test, y_train, y_test] = train_test_split(data[:, :-1], data[:, -1], test_size=0.2, random_state=0)#, shuffle=False)

    else:
        [x_train, x_test, y_train, y_test] = [data[0], data[1], data[2], data[3]]
    training_time.tic()
    svm_model = svm.LinearSVC().fit(x_train, y_train)
    training_time.toc()
    # svm_model = svm.LinearSVC(C= 1000,multi_class = 'ovr', max_iter=100000).fit(x_train, y_train)

    return model_accuracy(svm_model, x_test, y_test, training_time.elapsed)


def Decision_Tree(data, pca=False):
    training_time = TicToc()
    if pca is False:
        [x_train, x_test, y_train, y_test] = train_test_split(data[:, :-1], data[:, -1], test_size=0.2,
                                                              random_state=0)#, shuffle=False)
        dt = tree.DecisionTreeClassifier(max_depth=4)
        training_time.tic()
        dt.fit(x_train, y_train)
        training_time.toc()
        #filename = 'DecisionTree.sav'
        #joblib.dump(dt, filename)
    else:
        [x_train, x_test, y_train, y_test] = data
        dt = tree.DecisionTreeClassifier(max_depth=4)
        training_time.tic()
        dt.fit(x_train, y_train)
        training_time.toc()
        #filename = 'DecisionTreePCA.sav'
        #joblib.dump(dt, filename)

    return model_accuracy(dt, x_test, y_test, training_time.elapsed)


def AdaBoost(data, pca=False):
    training_time = TicToc()
    if pca is False:
        [x_train, x_test, y_train, y_test] = train_test_split(data[:, :-1], data[:, -1], test_size=0.2,
                                                              random_state=0)#, shuffle=False)
        bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), algorithm='SAMME.R', n_estimators=50)
        training_time.tic()
        bdt.fit(x_train, y_train)
        training_time.toc()
        #filename = 'AdaBoost.sav'
        #joblib.dump(bdt, filename)
    else:
        [x_train, x_test, y_train, y_test] = data

        bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), algorithm='SAMME.R', n_estimators=50)
        training_time.tic()
        bdt.fit(x_train, y_train)
        training_time.toc()
        #filename = 'AdaBoostPCA.sav'
        #joblib.dump(bdt, filename)

    # SAMME algo is used for discrete boosting
    # bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), algorithm="SAMME",n_estimators=200)

    # bdt = AdaBoostClassifier(SVC(kernel='rbf', gamma=0.3, C=1).fit(x_train, y_train), algorithm="SAMME",n_estimators=200)
    # bdt = AdaBoostClassifier(svm.LinearSVC(C= 1000,multi_class = 'ovr', max_iter=100000).fit(x_train, y_train), algorithm="SAMME",n_estimators=200)
    # bdt = AdaBoostClassifier(OneVsRestClassifier(SVC(kernel='rbf', gamma=0.3, C=1)).fit(x_train, y_train), algorithm="SAMME",n_estimators=200)
    # bdt = AdaBoostClassifier(SVC(kernel='linear', C=1).fit(x_train, y_train), algorithm="SAMME",n_estimators=200)

    return model_accuracy(bdt, x_test, y_test, training_time.elapsed)


def KNN(data, pca=False):
    training_time = TicToc()
    #testing_time = TicToc()
    if pca is False:
        [x_train, x_test, y_train, y_test] = train_test_split(data[:, :-1], data[:, -1], test_size=0.2, random_state=0)#, shuffle=False)
        knn = KNeighborsClassifier(n_neighbors=72)
        training_time = TicToc()
        training_time.tic()
        knn.fit(x_train, y_train)
        training_time.toc()
        #filename = 'knn.sav'
        #joblib.dump(knn, filename)

    else:
        [x_train, x_test, y_train, y_test] = data
        knn = KNeighborsClassifier(n_neighbors=72)
        training_time.tic()
        knn.fit(x_train, y_train)
        training_time.toc()
        #filename = 'knnPCA.sav'
        #joblib.dump(knn, filename)
    return model_accuracy(knn, x_test, y_test, training_time.elapsed)


def Gradient_Boosting(data):
    training_time = TicToc()
    [x_train, x_test, y_train, y_test] = train_test_split(data[:, :-1], data[:, -1], test_size=0.2,
                                                          random_state=0)#, shuffle=False)
    # alpha = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    gb_clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.75, max_features=8)
    training_time.tic()
    gb_clf.fit(x_train, y_train)
    training_time.toc()

    #filename = 'GradiantBoosting.sav'
    #joblib.dump(gb_clf, filename)
    return model_accuracy(gb_clf, x_test, y_test, training_time.elapsed)


def Kmean(data):
    [x_train, x_test, y_train, y_test] = train_test_split(data[:, :-1], data[:, -1], test_size=0.2,
                                                          random_state=0)#, shuffle=False)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(x_train)
    # predict =  kmeans.predict(X_test)
    # accuracy = np.mean(predict == y_test)
    # print('Mean Square Error For K-mean Classification : ', metrics.mean_squared_error(y_test, predict))
    # print('The achieved accuracy using K-mean is  '+str(accuracy*100))
    # print('----------------------------------------------------------')
    return model_accuracy(kmeans, x_test, y_test)


def Loading_Models(filename, x_test,y_test):
    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(x_test, y_test)
    loaded_model = joblib.load(filename)
    # result =  model_accuracy(loaded_model, x_test, y_test)
    y_pred = loaded_model.predict(x_test)
    accuracy = np.mean(y_pred == y_test)
    return accuracy


def model_accuracy(model, x_test, y_test, training_time):
    testing_time = TicToc()
    # Predicting the test set res
    testing_time.tic()
    y_pred = model.predict(x_test)
    testing_time.toc()
    accuracy = np.mean(y_pred == y_test)
    AccuracyList.append(accuracy * 100)
    TrainingTime.append(training_time)
    TestingTime.append(testing_time)
    return accuracy, training_time, testing_time.elapsed


def Selective_feature_BestK(x, y):
    # define feature selection
    fs = SelectKBest(score_func=f_classif, k=3)
    # apply feature selection
    X_selected = fs.fit_transform(x, y)
    return X_selected