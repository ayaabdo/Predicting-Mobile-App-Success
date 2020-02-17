import numpy as np
import pandas as pd
import Scripts.Classification as Class
import Scripts.PreprocessLibCalssification as Pre
from sklearn.model_selection import train_test_split

def startLoad(data,PCAVar=False):

    if PCAVar is False:
        # THIS LINE SHOULD BE COMMENTED
        #[x_train, x_test, y_train, y_test] = train_test_split(data[:, :-1], data[:, -1], test_size=0.2,
                                                             # random_state=0)  # , shuffle=False)
        # THESE TWO LINES WILL BE UNCOMMENTED
        x_test=data[:,:-1]
        y_test=data[:,-1]

        lr = Class.Loading_Models('E:\\The Champion\\college\\Last year isa\\ML\\MLProjectFinal\\Scripts\\LogisticRegression.sav', x_test, y_test)
        print(f'Logistic Reg Model accuracy {lr}')

        dt = Class.Loading_Models('E:\\The Champion\\college\\Last year isa\\ML\\MLProjectFinal\\Scripts\\DecisionTree.sav', x_test, y_test)
        print(f'Decision Tree Model accuracy {dt}')

        ada = Class.Loading_Models('E:\\The Champion\\college\\Last year isa\\ML\\MLProjectFinal\\Scripts\\AdaBoost.sav', x_test, y_test)
        print(f'Adaboost Model accuracy {ada}')

        svmall = Class.Loading_Models('E:\\The Champion\\college\\Last year isa\\ML\\MLProjectFinal\\Scripts\\SVMoneVSall.sav', x_test, y_test)
        print(f'svm one vs rest Model accuracy {svmall}')

        svmone = Class.Loading_Models('E:\\The Champion\\college\\Last year isa\\ML\\MLProjectFinal\\Scripts\\SVMoneVSone.sav', x_test, y_test)
        print(f'svm one vs one Model accuracy {svmone}')

        knn = Class.Loading_Models('E:\\The Champion\\college\\Last year isa\\ML\\MLProjectFinal\\Scripts\\knn.sav', x_test, y_test)
        print(f'knn Model accuracy {knn}')

    else:
        noOfSelctedFeatures = 10  # 7 , 9 ,10
        PCdata = Pre.Pca(data, noOfSelctedFeatures)

        [x1, x2, y1, y2] = PCdata
        x_test=np.append(x1,x2 ,axis=0)
        y_test = np.append(y1,y2, axis=0)

        lr = Class.Loading_Models('E:\\The Champion\\college\\Last year isa\\ML\\MLProjectFinal\\Scripts\\LogisticRegressionPCA.sav', x_test, y_test)
        print(f'Logistic Regression Model accuracy {lr}')

        dt = Class.Loading_Models('E:\\The Champion\\college\\Last year isa\\ML\\MLProjectFinal\\Scripts\\DecisionTreePCA.sav', x_test, y_test)  # , True)
        print(f'Decision Tree Model accuracy {dt}')

        ada = Class.Loading_Models('E:\\The Champion\\college\\Last year isa\\ML\\MLProjectFinal\\Scripts\\AdaBoostPCA.sav', x_test, y_test)
        print(f'adaboost Model accuracy {ada}')

        svmall = Class.Loading_Models('E:\\The Champion\\college\\Last year isa\\ML\\MLProjectFinal\\Scripts\\SVMoneVSallPCA.sav', x_test, y_test)
        print(f'svm one vs rest Model accuracy {svmall}')

        svmone = Class.Loading_Models('E:\\The Champion\\college\\Last year isa\\ML\\MLProjectFinal\\Scripts\\SVMoneVSonePCA.sav', x_test, y_test)
        print(f'svm one vs one Model accuracy {svmone}')

        knn = Class.Loading_Models('E:\\The Champion\\college\\Last year isa\\ML\\MLProjectFinal\\Scripts\\knnPCA.sav', x_test, y_test)
        print(f'knn Model accuracy {knn}')
    #print(len(x_test))



data = pd.read_csv('E:\\The Champion\\college\\Last year isa\\ML\\MLProjectFinal\\Scripts\\mobile_apps_classification_test.csv')
data = Pre.run(data)
data = data.astype(np.float64)
startLoad(data,PCAVar=False)