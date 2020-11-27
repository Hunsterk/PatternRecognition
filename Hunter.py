import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn import preprocessing
import seaborn as sns

mnist_data = pd.read_csv('mnist.csv').values


def plot_label_frequency():
    possible_labels = np.arange(0, 10)
    y = np.bincount(labels)
    plt.bar(possible_labels, y, width=0.8)
    plt.xlabel('Labels')
    plt.xticks(possible_labels)
    plt.ylabel('Number of records')
    plt.title('Digit Dataset')
    total = sum(y)
    for i, v in enumerate(y):
        plt.text(possible_labels[i] - 0.4, v + 50, str(round((v/total)*100, 2))+'%',
                 fontsize=9, color='dimgrey')
    plt.show()


def create_confusion_matrix(treu_values, pred_values, accuracy):
    cm = metrics.confusion_matrix(treu_values, pred_values)
    plt.figure(figsize=(12, 12))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual digit', size=14)
    plt.xlabel('Predicted digit', size=14)
    all_sample_title = 'Accuracy Score: {:.2%}'.format(accuracy)
    plt.title(all_sample_title, size=15)
    plt.show()


def analyse_feature(feature):
    # Standardize the training data
    standardized_feature = preprocessing.scale(feature)

    # The reshape is necessary to call LogisticRegression() with a single feature
    if feature.ndim == 1:  # check if array is one dimensional (single feature)
        standardized_feature = scale(standardized_feature).reshape(-1, 1)

    log = LogisticRegression()
    log.fit(standardized_feature, labels)
    predictions = log.predict(standardized_feature)
    accuracy = accuracy_score(labels, predictions)
    print("accuracy:", accuracy)

    # create_confusion_matrix(labels, predictions, accuracy)


def split(array, nrows, ncols):
    """Split a matrix into 4 sub-matrices."""

    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))


def create_zoning_feature(digits):
    feature = np.empty((len(digits), 4, 14, 14))

    for i, digit in enumerate(digits):
        # Reshape to a 2D array
        digits_2D = np.reshape(digit, (-1, 28))
        A, B, C, D,  = split(digits_2D, 14, 14)
        feature[i] = [A, B, C, D]

    return feature



if __name__ == '__main__':
    # Preparing data
    labels = mnist_data[:, 0]
    digits = mnist_data[:, 1:]

    # Question 1: Exploratory Data Analysis
    print("\nQuestion 1: Exploratory Data Analysis")
    plot_label_frequency()

    # Question 2: Ink feature Analysis
    print("\nQuestion 2: Ink feature Analysis")
    ink = np.array([sum(row) for row in digits])
    analyse_feature(ink)

    # Question 3: Own feature
    print("\nQuestion 3: Own feature")
    zoning = create_zoning_feature(digits)

    ink_zoning = np.empty((len(digits), 4))  # create empty array for ink sums of the zones
    for i, digit in enumerate(zoning):
        digit_ink_zones = []
        for zone in digit:
            ink_zone = np.sum(zone)
            digit_ink_zones.append(ink_zone)
        ink_zoning[i] = digit_ink_zones

    analyse_feature(ink_zoning)
