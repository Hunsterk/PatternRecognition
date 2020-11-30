import pandas as pd
mnist_data = pd.read_csv(r'C:\Users\isabu\OneDrive\Documenten\EERSTEJAAR GAME AND MEDIA TECHNOLOGY\Pattern recognition\mnist.csv').values

import matplotlib.pyplot as plt
labels = mnist_data[:, 0]
digits = mnist_data[:, 1:]
#img_size = 28
#plt.imshow(digits[5].reshape(img_size, img_size))
#plt.show()

#df = pd.DataFrame(mnist_data)
#dflabels = pd.DataFrame(labels)
#print(dflabels)

import numpy as np
ink = np.array([sum(row) for row in digits])
# compute mean for each digit class
ink_mean = [np.mean(ink[labels == i]) for i in range(10)]
# compute standard deviation for each digit class
ink_std = [np.std(ink[labels == i]) for i in range(10)]

#scale your feature 
from sklearn.preprocessing import scale
ink = scale(ink).reshape(-1, 1)

from sklearn import metrics

#logistic regression, create array of predictions of ink
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(ink, labels)
predictions = logisticRegr.predict(ink)

#confusion matrix
#from sklearn import metrics
#cm = metrics.confusion_matrix(labels, predictions)
#print(cm)

#split the digits in 4 quarters vertically
first_quarter = digits[:, 0:195]
second_quarter = digits[:, 195:391]
third_quarter = digits[:, 392:587]
fourth_quarter = digits[:, 588:]

#ink in each quarter
ink_first_quarter = np.array([sum(row) for row in first_quarter])
ink_second_quarter = np.array([sum(row) for row in second_quarter])
ink_third_quarter = np.array([sum(row) for row in third_quarter])
ink_fourth_quarter = np.array([sum(row) for row in fourth_quarter])

#ink means of each row
#ink_mean_firstrow = [np.mean(ink_firstrows[labels == i]) for i in range(10)]
#ink_mean_secondrow = [np.mean(ink_secondrows[labels == i]) for i in range(10)]
#ink_mean_thirdrow = [np.mean(ink_thirdrows[labels == i]) for i in range(10)]
#ink_mean_fourthrow = [np.mean(ink_fourthrows[labels == i]) for i in range(10)]

#scale each quarter feature
ink_first_quarter = scale(ink_first_quarter).reshape(-1, 1)
ink_second_quarter = scale(ink_second_quarter).reshape(-1, 1)
ink_third_quarter = scale(ink_third_quarter).reshape(-1, 1)
ink_fourth_quarter = scale(ink_fourth_quarter).reshape(-1, 1)

#make one array that contains the ink in each quarter
first_half = np.concatenate((ink_first_quarter, ink_second_quarter),axis=1)
second_half = np.concatenate((ink_third_quarter, ink_fourth_quarter),axis=1)
all_ink = np.concatenate((first_half, second_half), axis=1)

#Logistic regression
logisticRegrAll = LogisticRegression()
logisticRegrAll.fit(all_ink, labels)
predictions_all_ink = logisticRegrAll.predict(all_ink)

#confusion matrix
#cm_all_ink = metrics.confusion_matrix(labels, predictions_all_ink)
#print(cm_all_ink)
#score_all_ink = logisticRegrAll.score(all_ink, labels)
#print(score_all_ink)

import seaborn as sns
#plt.figure(figsize=(10,10))
#sns.heatmap(cm_all_ink, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
#plt.ylabel('Actual label')
#plt.xlabel('Predicted label');
#all_sample_title = 'Accuracy Score: {0}'.format(score_all_ink)
#plt.title(all_sample_title, size = 15);

both_features = np.concatenate((ink, all_ink), axis=1)
logisticRegrBoth = LogisticRegression()
logisticRegrBoth.fit(both_features, labels)
predictions_both = logisticRegrBoth.predict(both_features)

#cm_both = metrics.confusion_matrix(labels, predictions_both)
#print(cm_both)
#score_both = logisticRegrBoth.score(both_features, labels)
#print(score_both)
