import Scripts.PreprocessLibCalssification as Pre
import Scripts.Classification as Class
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ttictoc import TicToc

if __name__ == '__main__':
    data = pd.read_csv('D:\\Studying\\4th Year 1st Term\\Machine Learning\\Project\\Final Project\\MLProjectFinal\DataSets\\Mobile_App_Success_Milestone_2.csv')
    data = Pre.run(data)
    data = data.astype(np.float64)
    t = TicToc()
    # Plotting_Accuracy_Before_PCA()
    # Plotting_Accuracy_After_PCA(6)
    # Plotting_Testing_Time_Before_PCA()

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
    print(f'knn Model accuracy {knn}')'''

    # gb = Class.Loading_Models('GradiantBoosting.sav', x_test, y_test)
    # print(f'Gradiant Boosting Model accuracy {gb}')
    ####################################### TRAINIGN & testing #################################################
    res, train_time, test_time = Class.Logistic_Reg(data, False)
    print(f'Logistic Regression Model accuracy {res}')
    print(' training time for Logistic Regression Model is ', train_time)
    print(' testing time for Logistic Regression Model is ', test_time)
    print('-----------------------------------------------------------------------')

    for i in range(2, 4):
        res, train_time, test_time = Class.polynomial_classifier(data, i, False)
        print(f'Polynomial Model of degree {i} accuracy {res}')
        print(f'training time for polynomial classifier Model of degree {i} is ', train_time)
        print(f'testing time for polynomial classifier Model of degree {i} is ', test_time)
        print('-----------------------------------------------------------------------')

    dt, train_time, test_time = Class.Decision_Tree(data, False)
    print(f'Decision Tree accuracy is {dt}')
    print('training time for Decision Tree is ', train_time)
    print('testing time for Decision Tree is ', test_time)
    print('-----------------------------------------------------------------------')

    bdt, train_time, test_time = Class.AdaBoost(data, False)
    print(f'AdaBoost accuracy using DT is {bdt}')
    print('training time for AdaBoost is ', train_time)
    print('testing time for AdaBoost is ', test_time)
    print('-----------------------------------------------------------------------')

    # Svm,train_time,test_time = Class.SVM_withoutKernal(data, False)
    # print(f'SVM without Kernals accuracy is {Svm}' )
    # print('training time for SVM is ',train_time)
    # print('testing time for SVM is ',test_time)
    # print('-----------------------------------------------------------------------')

    OneVsRest, train_time, test_time = Class.SVM_OneVsRest(data, False)
    print(f'One VS Rest SVM accuracy ( Gaussian Kernel): is {OneVsRest}')
    print('training time for SVM One VS Rest ( Gaussian Kernel) is ', train_time)
    print('testing time for SVM One VS Rest ( Gaussian Kernel) is ', test_time)
    print('-----------------------------------------------------------------------')

    OneVsOne, train_time, test_time = Class.SVM_OneVsOne(data, False)
    print(f'One VS One SVM accuracy (Gaussian Kernel): is {OneVsOne}')
    print('training time for SVM One VS One ( Gaussian Kernel) is ', train_time)
    print('testing time for SVM One VS One ( Gaussian Kernel) is ', test_time)
    print('-----------------------------------------------------------------------')

    knn, train_time, test_time = Class.KNN(data, False)
    print(f'KNN accuracy : is {knn} ')
    print('training time for KNN is ', train_time)
    print('testing time for KNN is ', test_time)
    print('-----------------------------------------------------------------------')

    gb, train_time, test_time = Class.Gradient_Boosting(data)
    print(f'Gradiant Boosting accuracy is {gb}')
    print('training time for Gradiant Boosting is ', train_time)
    print('testing time for Gradiant Boosting is ', test_time)

    print("################### Calssification After PCA #########################")

    noOfSelctedFeatures = 10  # 7 , 9 ,10
    PCdata = Pre.Pca(data, noOfSelctedFeatures)

    ################################## LOADING FROM SAVED MODELS ##########################################
    '''[x_train, x_test, y_train, y_test] = PCdata

    lr = Class.Loading_Models('LogisticRegressionPCA.sav', x_test, y_test)
    print(f'Logistic Regression Model accuracy {lr}')

    #lrdeg2 = Class.Loading_Models('LogisticRegressiondeg2PCA.sav', x_test, y_test)
    #print(f'Polynomial Logistic Regression degree 2 Model accuracy {lrdeg2}')

    #lrdeg3 = Class.Loading_Models('LogisticRegressiondeg3PCA.sav', x_test, y_test)
    #print(f'Polynomial Logistic Regression degree 3 Model accuracy {lrdeg3}')

    dt = Class.Loading_Models('DecisionTreePCA.sav', x_test, y_test) 
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
    res, train_time, test_time = Class.Logistic_Reg(PCdata, True)
    print(f'Logistic Regression Model accuracy with PCA is {res}')
    print('training time for Logistic Regression is ', train_time)
    print('testing time for Logistic Regression is ', test_time)
    print('-----------------------------------------------------------------------')

    for i in range(2, 4):
        res, train_time, test_time = Class.polynomial_classifier(PCdata, i, True)
        print(f'Polynomial Model of degree {i} accuracy {res}')
        print('training time for Poly Logistic Regression is ', train_time)
        print('testing time for Poly Logistic Regression is ', test_time)
        print('-----------------------------------------------------------------------')

    dt, train_time, test_time = Class.Decision_Tree(PCdata, True)
    print(f'Decision Tree accuracy with PCA is {dt}')
    print('training time for Decision Tree is ', train_time)
    print('testing time for Decision Tree is ', test_time)
    print('-----------------------------------------------------------------------')

    bdt, train_time, test_time = Class.AdaBoost(PCdata, True)
    print(f'AdaBoost accuracy using DT with PCA is {bdt}')
    print('training time for AdaBoost is ', train_time)
    print('testing time for AdaBoost is ', test_time)
    print('-----------------------------------------------------------------------')

    # Svm,train_time,test_time = Class.SVM_withoutKernal(PCdata, True)
    # print(f'SVM without Kernals accuracy with PCA is {Svm}' )
    # print('training time for SVM is ',train_time)
    # print('testing time for SVM is ',test_time)
    # print('-----------------------------------------------------------------------')

    OneVsRest, train_time, test_time = Class.SVM_OneVsRest(PCdata, True)
    print(f'One VS Rest SVM accuracy ( Gaussian Kernel): with PCA is {OneVsRest}')
    print('training time for One VS Rest SVM ( Gaussian Kernel) is ', train_time)
    print('testing time for One VS Rest SVM ( Gaussian Kernel) is ', test_time)
    print('-----------------------------------------------------------------------')

    OneVsOne, train_time, test_time = Class.SVM_OneVsOne(PCdata, True)
    print(f'One VS One SVM accuracy (Gaussian Kernel): with PCA is {OneVsOne}')
    print('trainingtime for One VS One SVM accuracy (Gaussian Kernel) is ', train_time)
    print('testing time for One VS One SVM accuracy (Gaussian Kernel) is ', test_time)
    print('-----------------------------------------------------------------------')

    knn, train_time, test_time = Class.KNN(PCdata, True)
    print(f'KNN accuracy with PCA is {knn} ')
    print('training time for KNN is ', train_time)
    print('testing time for KNN is ', test_time)

    # Plotting_TrainingTime(models_name,TrainingTime)

    '''
    for i in range(2, 4):
        res ,train_time,test_time = Class.Polynomial_Classifier_with_SelectiveFeature(data, i, False)
        print(f'Polynomial Model of degree {i} accuracy {res}')
        print(f'training time for polynomial classifier Model of degree {i} is ',train_time)  
        print(f'testing time for polynomial classifier Model of degree {i} is ',test_time)  
        print('-----------------------------------------------------------------------')
    '''