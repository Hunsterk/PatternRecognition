import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import cv2
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score, RepeatedStratifiedKFold
from sklearn.svm import SVC

##read data
mnist_data = pd.read_csv(r'C:\Users\isabu\OneDrive\Documenten\EERSTEJAAR GAME AND MEDIA TECHNOLOGY\Pattern recognition\mnist.csv').values

##split the data in labels and digits
labels = mnist_data[:, 0]
digits = mnist_data[:, 1:]
#img_size = 28
#plt.imshow(digits[117].reshape(img_size, img_size))
#plt.show()

#df = pd.DataFrame(mnist_data)
#dflabels = pd.DataFrame(labels)
#print(dflabels)

##EXERCISE 2: ink feature
##compute ink for each digit
ink = np.array([sum(row) for row in digits])
##compute mean for each digit class
ink_mean = [np.mean(ink[labels == i]) for i in range(10)]
##compute standard deviation for each digit class
ink_std = [np.std(ink[labels == i]) for i in range(10)]

#Plot mean and standard deviation of the ink feature 
#plt.errorbar(range(10), ink_mean, yerr = ink_std, fmt = 'o')
#plt.show()

#scale your ink feature 
ink = scale(ink).reshape(-1, 1)

##logistic regression
logisticRegr = LogisticRegression()

##fit ink on labels and predict 
#logisticRegr.fit(ink, labels)
#predictions = logisticRegr.predict(ink)

##confusion matrix and score for ink feature
#cm = metrics.confusion_matrix(labels, predictions)
#score = logisticRegr.score (ink, labels)
#percentage_score = round(score*100, 2)
#print(cm)

##nice display of confusion matrix
#plt.figure(figsize=(9,9))
#sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r')
#plt.ylabel('Actual label')
#plt.xlabel('Predicted label');
#all_sample_title = 'Accuracy Score: {0}'.format(percentage_score)
#plt.title(all_sample_title, size = 15);

##EXERCISE 3: Create own feature
##split the digits in 4 quarters vertically 
first_quarter = digits[:, 0:196]
second_quarter = digits[:, 196:392]
third_quarter = digits[:, 392:588]
fourth_quarter = digits[:, 588:]

##ink in each quarter
ink_first_quarter = np.array([sum(row) for row in first_quarter])
ink_second_quarter = np.array([sum(row) for row in second_quarter])
ink_third_quarter = np.array([sum(row) for row in third_quarter])
ink_fourth_quarter = np.array([sum(row) for row in fourth_quarter])

##ink means of each row
#ink_mean_firstrow = [np.mean(ink_first_quarter[labels == i]) for i in range(10)]
#ink_mean_secondrow = [np.mean(ink_second_quarter[labels == i]) for i in range(10)]
#ink_mean_thirdrow = [np.mean(ink_third_quarter[labels == i]) for i in range(10)]
#ink_mean_fourthrow = [np.mean(ink_fourth_quarter[labels == i]) for i in range(10)]


#Plot mean and standard deviation of the ink in a region of a digit
#ink_std_second = [np.std(ink_second_quarter[labels == i]) for i in range(10)]
#plt.errorbar(range(10), ink_mean_secondrow, yerr = ink_std_second, fmt = 'o')
#plt.ylabel('Amount of ink')
#plt.xlabel('Digit class')


##scale each ink in each quarter
ink_first_quarter = scale(ink_first_quarter).reshape(-1, 1)
ink_second_quarter = scale(ink_second_quarter).reshape(-1, 1)
ink_third_quarter = scale(ink_third_quarter).reshape(-1, 1)
ink_fourth_quarter = scale(ink_fourth_quarter).reshape(-1, 1)

##make one array that contains the ink in each quarter
first_half = np.concatenate((ink_first_quarter, ink_second_quarter),axis=1)
second_half = np.concatenate((ink_third_quarter, ink_fourth_quarter),axis=1)
own_feature = np.concatenate((first_half, second_half), axis=1)

#k_fold = KFold(n_splits = 3)
#hyperparameters = {'C': [1, 2, 5, 10, 50]}
#grid = GridSearchCV(logisticRegr, hyperparameters, cv=k_fold, verbose=0)
#best_model = grid.fit(own_feature, labels)
#print('Best C:', best_model.best_estimator_.get_params()['C'])

##Logistic regression on own feature
#logisticRegr.fit(own_feature, labels)
#predictions_own_feature = logisticRegr.predict(own_feature)

##confusion matrix and score from own feature
#cm_own_feature = metrics.confusion_matrix(labels, predictions_own_feature)
#print(cm_own_feature)
#score_own_feature = logisticRegr.score(own_feature, labels)
#percentage_score_own_feature = round(score_own_feature*100,3)
#print(score_own_feature)

##nice display of confusion matrix
#plt.figure(figsize=(9,9))
#sns.heatmap(cm_own_feature, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r')
#plt.ylabel('Actual label')
#plt.xlabel('Predicted label');
#all_sample_title = 'Accuracy Score: {0}'.format(percentage_score_own_feature)
#plt.title(all_sample_title, size = 15);

##EXERCISE 4: use both features
#concatenate own feature array and ink array
#both_features = np.concatenate((ink, own_feature), axis=1)

##logistic regression on both features
#logisticRegr.fit(both_features, labels)
#predictions_both_features = logisticRegr.predict(both_features)

##confusion matrix and score both features 
#cm_both_features = metrics.confusion_matrix(labels, predictions_both_features)
#print(cm_both_features)
#score_both_features = logisticRegr.score(both_features, labels)
#percentage_score_both_features = round(score_both_features*100, 3)
#print(score_both_features)

##nice display of confusion matrix
#plt.figure(figsize=(9,9))
#sns.heatmap(cm_both_features, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r')
#plt.ylabel('Actual label')
#plt.xlabel('Predicted label');
#all_sample_title = 'Accuracy Score: {0}'.format(percentage_score_both_features)
#plt.title(all_sample_title, size = 15);

##EXERCISE 5: pixel as feature and different classifiers
##Resize data
resized_digits = np.empty((len(digits), 14, 14))  # create empty array for resized digits
for i in range(len(digits)):
    digit = digits[i].reshape((28, 28)).astype('float32')
    resized_digit = cv2.resize(digit, (14, 14), interpolation=cv2.INTER_AREA)
    resized_digits[i] = resized_digit
    
##Reshape training data from 3D array to 2D array
nsamples, nx, ny = resized_digits.shape
resized_digits = resized_digits.reshape((nsamples, nx * ny))

##Split data in trainingset of 5000 datapoints and testset of 37000 datapoints
X_train, X_test, Y_train, Y_test = train_test_split(resized_digits, labels, test_size=0.88095, random_state=1)

##EXERCISE 5.1
##Decide on best complexity parameter C, using cross validation with 3 folds on the train set, result C = 1.0 
#model = LogisticRegression(penalty = 'l1', solver = 'saga')
#k_fold = KFold(n_splits = 3, random_state=1)
#hyperparameters = {'C': [1, 2, 5, 10, 50]}
#grid = GridSearchCV(model, hyperparameters, cv=k_fold, verbose=0)
#best_model = grid.fit(X_train, Y_train)
#print('Best C:', best_model.best_estimator_.get_params()['C'])

##Logistic regression on pixels with complexity parameter C=1 and Lasso penalty and predict test set
#logisticRegrWithC = LogisticRegression(C = 50, penalty = 'l1', solver = 'saga')
#logisticRegrWithC.fit(X_train, Y_train)
#predictions_pixels_with_C = logisticRegrWithC.predict(X_test)

##confusion matrix and score pixels with tuned complexity parameter
#cm_pixels_with_C = metrics.confusion_matrix(Y_test, predictions_pixels_with_C)
#print(cm_pixels_with_C)
#score_pixels_with_C = logisticRegrWithC.score(X_test, Y_test)
#percentage_score_with_C = round(score_pixels_with_C*100, 2)
#print(score_pixels_with_C)

##nice display of confusion matrix
#plt.figure(figsize=(9,9))
#sns.heatmap(cm_pixels_with_C, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r')
#plt.ylabel('Actual label')
#plt.xlabel('Predicted label');
#all_sample_title = 'Accuracy Score: {0}'.format(percentage_score_with_C)
#plt.title(all_sample_title, size = 15);

##EXERCISE 5.2
#Tune parameters of SVM
#supportVectorMachine = SVC()
#k_fold = KFold(n_splits = 3, random_state=1)
#hyperparameters = {'C': [1, 5, 10, 50],'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
#grid = GridSearchCV(supportVectorMachine, hyperparameters, cv = k_fold)
#best_model = grid.fit(X_train, Y_train)
#print(grid.best_params_)
#print(grid.best_estimator_)

##Support vector machine tuned parameters
#supportVectorMachineTuned = SVC(C = 10.0, kernel = 'rbf')
#supportVectorMachineTuned.fit(X_train, Y_train)
#predictions_SVC_Tuned = supportVectorMachineTuned.predict(X_test)

#Confusion matrix and score pixels, with tuned parameter
#score_SVC_Tuned = supportVectorMachineTuned.score(X_test, Y_test)
#percentage_score_SVC_Tuned = round(score_SVC_Tuned*100, 2)
#cm_SVC_tuned = metrics.confusion_matrix(Y_test, predictions_SVC_Tuned)
#print(cm_SVC_tuned)
#print(score_SVC_Tuned)

##nice display of confusion matrix on pixels with svc and tuned parameter
#plt.figure(figsize=(9,9))
#sns.heatmap(cm_SVC_tuned, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r')
#plt.ylabel('Actual label')
#plt.xlabel('Predicted label');
#all_sample_title = 'Accuracy Score: {0}'.format(percentage_score_SVC_Tuned)
#plt.title(all_sample_title, size = 15);
