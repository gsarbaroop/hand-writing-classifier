import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import pandas as pd
from PIL import Image

gamma_list=[0.001, 0.05, 0.003, 0.0002, 0.00001]
c_list = [0.1, 0.8, 0.3, 2, 8]

h_param_comb = [{'gamma':g, 'C':c} for g in gamma_list for c in c_list]

train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

digits = datasets.load_digits()
data = digits.images.reshape((n_samples, -1))
print("---Digits--")
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()
print(digits.images[0].size)

dev_test_frac = 1-train_frac
X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=True
)

X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True
)

df = pd.DataFrame()
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
        
        
df['Train Accuracy'] = acc_train
df['Dev Accuracy']= acc_dev
df['Test Accuracy'] = acc_test

print(df)

accuracy_dev = df['Dev Accuracy']
accuracy_train = df['Train Accuracy']
accuracy_test = df['Test Accuracy']
max_dev = accuracy_dev.max()
max_train = accuracy_train.max()
max_test = accuracy_test.max()
min_dev = accuracy_dev.min()
min_train = accuracy_train.min()
min_test = accuracy_test.min()
mean_dev = accuracy_dev.mean()
mean_train = accuracy_train.mean()
mean_test = accuracy_test.mean()
median_dev = accuracy_dev.median()
median_train = accuracy_train.median()
median_test = accuracy_test.median()

print("The max of dev_accuracy: ", max_dev)
print("The max of train_accuracy: ",max_train)
print("The max of test_accuracy: ",max_test)
print("The min of dev_accuracy: ",min_dev)
print("The min of train_accuracy: ",min_train)
print("The min of test_accuracy: ",min_test)
print("The mean of dev_accuracy: ",mean_dev)
print("The mean of train_accuracy: ",mean_train)
print("The mean of test_accuracy: ",mean_test)
print("The median of dev_accuracy: ",median_dev)
print("The median of train_accuracy: ",median_train)
print("The median of test_accuracy: ",median_test)

plt.show()

