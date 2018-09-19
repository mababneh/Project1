from perceptron import Perceptron2
from sgd import AdalineSGD
from Adaline import AdalineGD
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def main ():
    DatasetNM=input("Please Enter the name of DataSet (iris,PhishingDAta)\n")
    if DatasetNM=="iris":
        train = pd.read_csv('dataset.csv', header=None)
        X = train.iloc[0:105, [0, 1, 2, 3]].values
        #X_normalized = preprocessing.normalize(X, norm='l2')
        y0 = train.iloc[0:105, 4].values
        y = np.where(y0 == 'Iris-setosa', 1, -1)
        ytest1=y
        ytest2 = np.where(y0== 'Iris-versicolor', +1, -1)
        ytest3 = np.where(y0 == 'Iris-virginica', +1, -1)

        print("--------------------------------------------------------------------------------------------------")
        test = pd.read_csv('testdata.csv', header=None)
        Xtest = test.iloc[0:45, [0, 1, 2, 3]].values
        #X_scaled1 = preprocessing.normalize(Xtest, norm='l2')
        ytest0 = test.iloc[0:45, 4].values
        ytest = np.where(ytest0 == 'Iris-setosa', +1, -1)


    elif DatasetNM=="PhishingDAta":
        train = pd.read_csv('traindataML.csv', header=None)
        X = train.iloc[0:998, [0, 1, 2, 3,4,5,6,7,8]].values
        #X_normalized = preprocessing.normalize(X, norm='l2')
        y0 = train.iloc[0:998, 9].values
        y=np.where(y0 ==0, 1, -1)
        ytest1 = y
        ytest2 = np.where(y0 == 1, +1, -1)
        ytest3 = np.where(y0 == -1, +1, -1)

        print("--------------------------------------------------------------------------------------------------")
        test = pd.read_csv('testdataML.csv', header=None)
        Xtest = test.iloc[0:354, [0, 1, 2, 3,4,5,6,7,8]].values
        #X_scaled1 = preprocessing.normalize(Xtest, norm='l1')
        ytest0 = test.iloc[0:354, 9].values
        ytest = np.where(ytest0 ==0, +1, -1)

    def Drawing(cost):
        plt.plot(range(1, len(cost) + 1), cost ,marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Number of misclassifications')
        plt.show()


    clock=True
    def CheckingClassifer(ClassiferName):

        if ClassiferName=="Perceptron":
            counter1 = 0
            pop = Perceptron2(0.01, 10)
            print("The Error inEach iteration")
            df=pop.fit(X, y)
            print (df)
            dd = pop.predict(Xtest)
            print("The Prediction values")
            print(dd)
            Drawing(df)
            for i in range(len(ytest)):
                if dd[i] == ytest[i]:
                    counter1 = counter1 + 1
            print(counter1)
            print("The accuracy " + str(counter1 / len(dd)))
            errorrate = 1 - (counter1 / len(dd))
            print("The error rate is " + str(errorrate))

        elif ClassiferName=="AdaLine":
            p = AdalineGD(0.00001,500)
            print("The Cost in eash Iteration")
            df = p.fit(X, y)
            Drawing(df)
            print (df)
            perceptonpredict = p.predict(Xtest)
            print("The Presdiction Valuse")
            print(perceptonpredict)
            counter2 = 0
            for i in range(len(ytest)):
                if perceptonpredict[i] == ytest[i]:
                    counter2 = counter2 + 1
            print("The accuracy " + str(counter2 / len(perceptonpredict)))
            errorrate = 1 - (counter2 / len(perceptonpredict))
            print("The error rate is " + str(errorrate))
        elif ClassiferName=="SGD":
            pp = AdalineSGD(.001, 50)
            print("The Cost Values in each iteration")
            df = pp.fit(X, y)
            Drawing(df)
            print (df)
            f = pp.predict(Xtest)
            print("The Prediction Valuse of Classifer : ")
            print(f)
            counter = 0
            for i in range(len(ytest)):
                if f[i] == ytest[i]:
                    counter = counter + 1
            print("The accuracy " + str(counter / len(f)))
            errorrate = 1 - (counter / len(f))
            print("The error rate is " + str(errorrate))
        elif ClassiferName=="OneVsAll":
            counter1 = 0
            counter2=0
            counter3=0
            print("------------------------------------------------------------------------")
            pop = AdalineSGD(0.01, 15)
            pop.fit(X, ytest1)
            dd = pop.predict(Xtest)

            for i in range(len(dd)):
                if dd[i] == ytest[i]:
                    counter1 = counter1 + 1
            print("The number of instances  which classified correctly "+str(counter1))
            print("The accuracy " + str(counter1 / len(dd)))
            errorrate = 1 - (counter1 / len(dd))
            print("The error rate is " + str(errorrate))
            print("-------------------------------------------------------------------------")
            pop = AdalineSGD(0.01, 15)
            pop.fit(X, ytest2)
            dd = pop.predict(Xtest)
            for i in range(len(dd)):
                if dd[i] == ytest[i]:
                    counter2 = counter2 + 1
            print("The number of instances  which classified correctly " + str(counter2))
            print("The accuracy is " + str(counter2 / len(dd)))
            errorrate = 1 - (counter2 / len(dd))
            print("The error rate is " + str(errorrate))
            print("----------------------------------------------------------------------------")
            pop = AdalineSGD(0.01, 15)
            pop.fit(X, ytest3)
            dd = pop.predict(Xtest)
            for i in range(len(dd)):
                if dd[i] == ytest[i]:
                    counter3 = counter3+ 1
            print("The number of instances  which classified correctly " + str(counter3))
            print("The accuracy is " + str(counter3 / len(dd)))
            errorrate = 1 - (counter3 / len(dd))
            print("The error rate is " + str(errorrate))
    def ERRoRCheckingClassifer():
        while clock==True:
            ClassiferName = input("please Enter The ClassiferName {Perceptron,AdaLine,SGD,OneVsAll}:\n")
            if ((ClassiferName=="Perceptron") or (ClassiferName=="AdaLine") or (ClassiferName=="SGD") or(ClassiferName=="OneVsAll")):
                CheckingClassifer(ClassiferName)
            elif ClassiferName == str(0):
                print("Thank You for Using My program")
                break
            else:
                print("please Enter the name of the file correctly\n")

    ERRoRCheckingClassifer()





















if __name__ == "__main__":
    main()
    CheckingClassifer(ClassiferName)