import Scripts.PreprocessLib as Pre
import Scripts.Regression as Reg
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ttictoc import TicToc

if __name__ == '__main__':
    #List of names
    models_name = ['Logistic' + '\n' + 'Reg.', 'Poly' + '\n' + 'Log.' + '\n' + 'd=2',
                   'Poly' + '\n' + 'Log.' + '\n' + 'd=3', 'Decision' + '\n' + 'Tree',
                   '     AdaBoost' + '\n' + '(DT)',
                   '     SVM' + '\n' + '    Gaussian' + '\n' + 'OVR',
                   '    SVM' + '\n' + '       Gaussian' + '\n' + 'OVO', 'KNN']

    data = Pre.run('E:\\The Champion\\college\\Last year isa\\ML\\MLProjectFinal\\DataSets\\Project.csv',drop=True)
    data = data.astype(np.float64)
    t = TicToc()
    # Plotting_Accuracy_Before_PCA()
    # Plotting_Accuracy_After_PCA(6)
    # Plotting_Testing_Time_Before_PCA()
    from sklearn.preprocessing import StandardScaler

    print('#################### Regression #########################################')
    t.tic()
    res = Reg.multi_variant_regression(data)
    test_data = Pre.run('E:\\The Champion\\college\\Last year isa\\ML\\MLProjectFinal\\DataSets\\mobile_apps_regression_test.csv')
    test_data = test_data.astype(np.float64)
    scale = StandardScaler()
    scale.fit(test_data[:, :-1])
    scaled_data = scale.fit_transform(test_data[:, :-1])
    scaled_data = np.append(scaled_data, np.reshape(test_data[:, -1], (len(test_data[:, -1]), 1)), axis=1)
    model = res[0];
    values = model.predict(scaled_data[:, :-1])
    sse = 0
    print(len(values))
    for real, pred in zip(scaled_data[:,-1],values):
        sse = sse + (real-pred)**2
    print(f'linear sqaure error{sse}, mean square error {sse / len(values)}')
    from sklearn.preprocessing import PolynomialFeatures

    model = Reg.polynomial_regression(data,2)[0]
    features_generator = PolynomialFeatures(degree=2)
    features = features_generator.fit_transform(scaled_data[:, :-1])
    arr = np.append(features[:, 1:], np.reshape(scaled_data[:, -1], (len(scaled_data[:, -1]), 1)), axis=1)

    values = model.predict(arr[:, :-1])
    print(values)
    sse=0
    for real, pred in zip(arr[:, -1], values):
        sse = sse + (real - pred) ** 2
    print(f'ploy 2 sqaure error{sse}, mean square error {sse / len(values)}')

    #model.predict()
    #print(f'Multiple Linear Model R-Squared, SSE, MSE {res[0]:.3f} {res[1]:.3f} {res[2]:.3f}')
    #t.toc()
    #print('time for Multiple Linear Model is ',t.elapsed)
    #print('-----------------------------------------------------------------------')

    '''filename = 'multivar.sav'
    pickle.dump(res[3], open(filename, 'wb'))
    with open('observations1.txt', 'w') as file:
        j = 0
        for ele1, ele2 in zip(res[4], res[5]):
            file.write(f"{j}, {ele1:.5f},  {ele2:.5f}, {(ele1 - ele2):.5f}, {(ele1 - ele2) ** 2:.5f}")
            file.write('\n')
            j += 1'''
    '''
    for i in range(2, 4):
        t.tic()
        res = Reg.polynomial_regression(data, i)
        print(f'Polynomial Linear Model of degree {i} R-Squared, SSE, MSE {res[0]:.3f} {res[1]:.3f} {res[2]:.3f}')
        t.toc()
        print(f'time for Polynomial Linear Model of degree {i} is ',t.elapsed)
        print('-----------------------------------------------------------------------')
        filename = f'deg{i}poly.sav'
        pickle.dump(res[3], open(filename, 'wb'))
        with open(f'observations{i}.txt', 'w') as file:
            j = 0
            for ele1, ele2 in zip(res[4], res[5]):
                file.write(f"{j}, {ele1:.5f},  {ele2:.5f}, {(ele1 - ele2):.5f}, {(ele1 - ele2) ** 2:.5f}")
                file.write('\n')
                j += 1
    '''
    print("################### Calssification Before PCA ########################")

    ################################## LOADING FROM SAVED MODELS ##########################################

    '''x_train, x_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.2,
                                                        random_state=0)
    lr = Class.Loading_Models('LogisticRegression.sav', x_test, y_test)
    print(f'Logistic Reg Model accuracy {lr}')

    #lrdeg2 = Class.Loading_Models('LogisticRegressiondeg2.sav', x_test, y_test)
    #print(f'Polynomial Logistic Regression of degree 2 Model accuracy {lrdeg2}')

    #lrdeg3 = Class.Loading_Models('LogisticRegressiondeg3.sav', x_test, y_test)
    #print(f'Polynomial Logistic Regression of degree 3 Model accuracy {lrdeg3}')

    dt = Class.Loading_Models('DecisionTree.sav', x_test, y_test)
    print(f'Decision Tree Model accuracy {dt}')

    ada = Class.Loading_Models('AdaBoost.sav', x_test, y_test)
    print(f'Adaboost Model accuracy {ada}')

    svmall = Class.Loading_Models('SVMoneVSall.sav', x_test, y_test)
    print(f'svm one vs rest Model accuracy {svmall}')

    svmone = Class.Loading_Models('SVMoneVSone.sav', x_test, y_test)
    print(f'svm one vs one Model accuracy {svmone}')

    knn = Class.Loading_Models('knn.sav', x_test, y_test)
    print(f'knn Model accuracy {knn}')

    #gb = Class.Loading_Models('GradiantBoosting.sav', x_test, y_test)
    #print(f'Gradiant Boosting Model accuracy {gb}')'''
    ####################################### TRAINIGN & testing #################################################
    '''res,train_time,test_time = Class.Logistic_Reg(data, False)
    print(f'Logistic Regression Model accuracy {res}')
    print(' training time for Logistic Regression Model is ',train_time)
    print(' testing time for Logistic Regression Model is ',test_time)
    print('-----------------------------------------------------------------------')

    for i in range(2, 4):
        res ,train_time,test_time = Class.polynomial_classifier(data, i, False)
        print(f'Polynomial Model of degree {i} accuracy {res}')
        print(f'training time for polynomial classifier Model of degree {i} is ',train_time)  
        print(f'testing time for polynomial classifier Model of degree {i} is ',test_time)  
        print('-----------------------------------------------------------------------')

    dt,train_time,test_time = Class.Decision_Tree(data, False)
    print(f'Decision Tree accuracy is {dt}')
    print('training time for Decision Tree is ',train_time)
    print('testing time for Decision Tree is ',test_time)
    print('-----------------------------------------------------------------------')

    bdt,train_time,test_time = Class.AdaBoost(data, False)
    print(f'AdaBoost accuracy using DT is {bdt}')
    print('training time for AdaBoost is ',train_time)
    print('testing time for AdaBoost is ',test_time)
    print('-----------------------------------------------------------------------')

    #Svm,train_time,test_time = Class.SVM_withoutKernal(data, False)
    #print(f'SVM without Kernals accuracy is {Svm}' )
    #print('training time for SVM is ',train_time)
    #print('testing time for SVM is ',test_time)
    #print('-----------------------------------------------------------------------')

    OneVsRest,train_time,test_time = Class.SVM_OneVsRest(data, False)
    print(f'One VS Rest SVM accuracy ( Gaussian Kernel): is {OneVsRest}' )
    print('training time for SVM One VS Rest ( Gaussian Kernel) is ',train_time)
    print('testing time for SVM One VS Rest ( Gaussian Kernel) is ',test_time)
    print('-----------------------------------------------------------------------')

    OneVsOne,train_time,test_time = Class.SVM_OneVsOne(data, False)
    print(f'One VS One SVM accuracy (Gaussian Kernel): is {OneVsOne}')
    print('training time for SVM One VS One ( Gaussian Kernel) is ',train_time)
    print('testing time for SVM One VS One ( Gaussian Kernel) is ',test_time)
    print('-----------------------------------------------------------------------')

    knn,train_time,test_time=Class.KNN(data, False)
    print(f'KNN accuracy : is {knn} ')
    print('training time for KNN is ',train_time)
    print('testing time for KNN is ',test_time)
    print('-----------------------------------------------------------------------')

    gb,train_time,test_time = Class.Gradient_Boosting(data)
    print(f'Gradiant Boosting accuracy is {gb}')
    print('training time for Gradiant Boosting is ',train_time)
    print('testing time for Gradiant Boosting is ',test_time)'''

    print("################### Calssification After PCA #########################")

    '''noOfSelctedFeatures = 10  # 7 , 9 ,10
    PCdata = Pre.Pca(data, noOfSelctedFeatures)

    ################################## LOADING FROM SAVED MODELS ##########################################
    [x_train, x_test, y_train, y_test] = PCdata

    lr = Class.Loading_Models('LogisticRegressionPCA.sav', x_test, y_test)
    print(f'Logistic Regression Model accuracy {lr}')'''

    '''lrdeg2 = Class.Loading_Models('LogisticRegressiondeg2PCA.sav', x_test, y_test)
    print(f'Polynomial Logistic Regression degree 2 Model accuracy {lrdeg2}')

    lrdeg3 = Class.Loading_Models('LogisticRegressiondeg3PCA.sav', x_test, y_test)
    print(f'Polynomial Logistic Regression degree 3 Model accuracy {lrdeg3}')'''

    '''dt = Class.Loading_Models('DecisionTreePCA.sav', x_test, y_test)  # , True)
    print(f'Decision Tree Model accuracy {dt}')

    ada = Class.Loading_Models('AdaBoostPCA.sav', x_test, y_test)
    print(f'adaboost Model accuracy {ada}')

    svmall = Class.Loading_Models('SVMoneVSallPCA.sav', x_test, y_test)
    print(f'svm one vs rest Model accuracy {svmall}')

    svmone = Class.Loading_Models('SVMoneVSonePCA.sav', x_test, y_test)
    print(f'svm one vs one Model accuracy {svmone}')

    knn = Class.Loading_Models('knnPCA.sav', x_test, y_test)
    print(f'knn Model accuracy {knn}')'''

    ####################################### TRAINIGN #################################################
    '''res,train_time,test_time = Class.Logistic_Reg(PCdata, True)
    print(f'Logistic Regression Model accuracy with PCA is {res}')
    print('training time for Logistic Regression is ',train_time)
    print('testing time for Logistic Regression is ',test_time)
    print('-----------------------------------------------------------------------')

    for i in range(2, 4):
        res,train_time,test_time = Class.polynomial_classifier(PCdata, i, True)
        print(f'Polynomial Model of degree {i} accuracy {res}')
        print('training time for Poly Logistic Regression is ',train_time)
        print('testing time for Poly Logistic Regression is ',test_time)
        print('-----------------------------------------------------------------------')

    dt,train_time,test_time = Class.Decision_Tree(PCdata, True)
    print(f'Decision Tree accuracy with PCA is {dt}')
    print('training time for Decision Tree is ',train_time)
    print('testing time for Decision Tree is ',test_time)
    print('-----------------------------------------------------------------------')

    bdt,train_time,test_time = Class.AdaBoost(PCdata, True)
    print(f'AdaBoost accuracy using DT with PCA is {bdt}')
    print('training time for AdaBoost is ',train_time)
    print('testing time for AdaBoost is ',test_time)
    print('-----------------------------------------------------------------------')


    #Svm,train_time,test_time = Class.SVM_withoutKernal(PCdata, True)
    #print(f'SVM without Kernals accuracy with PCA is {Svm}' )
    #print('training time for SVM is ',train_time)
    #print('testing time for SVM is ',test_time)
    #print('-----------------------------------------------------------------------')


    OneVsRest,train_time,test_time = Class.SVM_OneVsRest(PCdata, True)
    print(f'One VS Rest SVM accuracy ( Gaussian Kernel): with PCA is {OneVsRest}')
    print('training time for One VS Rest SVM ( Gaussian Kernel) is ',train_time)
    print('testing time for One VS Rest SVM ( Gaussian Kernel) is ',test_time)
    print('-----------------------------------------------------------------------')

    OneVsOne,train_time,test_time = Class.SVM_OneVsOne(PCdata, True)
    print(f'One VS One SVM accuracy (Gaussian Kernel): with PCA is {OneVsOne}')
    print('trainingtime for One VS One SVM accuracy (Gaussian Kernel) is ',train_time)
    print('testing time for One VS One SVM accuracy (Gaussian Kernel) is ',test_time)
    print('-----------------------------------------------------------------------')

    knn,train_time,test_time=Class.KNN(PCdata, True)
    print(f'KNN accuracy with PCA is {knn} ')
    print('training time for KNN is ',train_time)
    print('testing time for KNN is ',test_time)'''

    # Plotting_TrainingTime(models_name,TrainingTime)

    '''
    for i in range(2, 4):
        res ,train_time,test_time = Class.Polynomial_Classifier_with_SelectiveFeature(data, i, False)
        print(f'Polynomial Model of degree {i} accuracy {res}')
        print(f'training time for polynomial classifier Model of degree {i} is ',train_time)  
        print(f'testing time for polynomial classifier Model of degree {i} is ',test_time)  
        print('-----------------------------------------------------------------------')
    '''