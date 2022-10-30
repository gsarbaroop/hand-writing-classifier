import matplotlib.pyplot as plt
import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics, tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


gamma_list=[0.001, 0.05, 0.003, 0.0002, 0.00001]
c_list = [0.1, 0.8, 0.3, 2, 8]

digits = datasets.load_digits()

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

digits = datasets.load_digits()
data = digits.images.reshape((n_samples, -1))

val = [0.1, 0.3, 0.4, 0.6, 0.8]
sum_svm_acc=0
sum_dt_acc=0

def svm_classifier(X_train,y_train,X_test,y_test,X_dev,y_dev):
    acc_train = []
    acc_dev =[]
    acc_test =[]

    for gamma in gamma_list:
        for c in c_list:
            # Create a classifier: a support vector classifier
            clf = svm.SVC(gamma=gamma, C=c)
            
            # Learn the digits on the train subset
            clf.fit(X_train, y_train)

            # Predict the value of the digit on the test subset
            predicted_dev = clf.predict(X_dev)
            predicted_train = clf.predict(X_train)
            predicted_test = clf.predict(X_test)
            
            score_dev = accuracy_score(y_pred=predicted_dev,y_true=y_dev)
            score_train = accuracy_score(y_pred=predicted_train,y_true=y_train)
            score_test = accuracy_score(y_pred=predicted_test,y_true=y_test)
            acc_train.append(score_train)
            acc_dev.append(score_dev)
            acc_test.append(score_test)

    return acc_dev,acc_train,acc_test


def decisionTree(X_train,y_train,X_test,y_test,X_dev,y_dev):
    acc_train = []
    acc_dev =[]
    acc_test =[]
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    predicted_dev = clf.predict(X_dev)
    predicted_train = clf.predict(X_train)
    predicted_test = clf.predict(X_test)
    
    score_dev = accuracy_score(y_pred=predicted_dev,y_true=y_dev)
    score_train = accuracy_score(y_pred=predicted_train,y_true=y_train)
    score_test = accuracy_score(y_pred=predicted_test,y_true=y_test)
    acc_train.append(score_train)
    acc_dev.append(score_dev)
    acc_test.append(score_test)

    return acc_dev,acc_train,acc_test

svm_np=[]
dt_np=[]

for split in val:

    X_train, X_dev_test, y_train, y_dev_test = train_test_split(
        data, digits.target, test_size=split, shuffle=True
    )

    X_test, X_dev, y_test, y_dev = train_test_split(
        X_dev_test, y_dev_test, test_size=split, shuffle=True
    )

    acc_dev,acc_train,acc_test = svm_classifier(X_train,y_train,X_test,y_test,X_dev,y_dev)
    sum_test=0
    for i in acc_test:
        sum_test+=i
    
    # mean_dev = acc_dev.mean()
    mean_test = sum_test/len(acc_test)
    # mean_test = acc_test.mean()
    sum_svm_acc+=mean_test
    svm_np.append(mean_test)
    print("SVM: ",mean_test)

    acc_dev,acc_train,acc_test = decisionTree(X_train,y_train,X_test,y_test,X_dev,y_dev)
    sum_test_dt=0
    for i in acc_test:
        sum_test_dt+=i

    mean_test_dt = sum_test_dt/len(acc_test)
    dt_np.append(mean_test_dt)
    print("Decision Tree: ",mean_test_dt)
    sum_dt_acc+=mean_test_dt

print("Decision Tree mean: ",sum_dt_acc/5)
print("SVM mean: ",sum_svm_acc/5)
print("Decision Tree Std Dev:", np.std(dt_np))
print("SVM Std Dev:", np.std(svm_np))