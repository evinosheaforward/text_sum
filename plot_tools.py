import itertools

import matplotlib.pyplot
import numpy
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, ShuffleSplit, learning_curve
from sklearn.metrics import confusion_matrix

#for the recordbooks plt.savefig("matplot1.jpg")
def plot_confusion(true_labels, predictions, classes, title="Confusion Matrix"):
    """Plotting function for a confusion matrix, only need to pass predictions,
    correct labels, and classes
    """

    cm = confusion_matrix(true_labels, predictions)
    cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]

    matplotlib.pyplot.imshow(cm, interpolation='nearest', cmap=matplotlib.pyplot.cm.Blues)
    matplotlib.pyplot.colorbar()
    tick_marks = numpy.arange(len(classes))
    matplotlib.pyplot.xticks(tick_marks, classes, rotation=45)
    matplotlib.pyplot.yticks(tick_marks, classes)


    thresh = cm.max() / 2. 
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        matplotlib.pyplot.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.ylabel('True label')
    matplotlib.pyplot.xlabel('Predicted label')
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.show()


    # # add some text for labels, title and axes ticks
    # ax.set_xticks(ind + width / 2)
    # ax.set_xticklabels([key for key in numbers["spam"]])

    # ax.legend((rects1[0], rects2[0]), ('ham', 'spam'))
    # matplotlib.pyplot.show()

def plot_learning(features_matrix, labels, classifier=XGBClassifier(), n_splits=10, 
                  test_size=0.2, random_state=0, n_jobs=1, train_sizes=None, title=None, percent=False):
    """Plot the learning curve for a 
    """
    cv = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=random_state)
    n_jobs = 1
    if train_sizes is None:
        train_sizes = numpy.linspace(.1, 1.0, 10)

    train_sizes, train_scores, test_scores = learning_curve(classifier, features_matrix, labels,
                                        cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = numpy.mean(train_scores, axis=1)
    train_scores_std = numpy.std(train_scores, axis=1)
    test_scores_mean = numpy.mean(test_scores, axis=1)
    test_scores_std = numpy.std(test_scores, axis=1)
    matplotlib.pyplot.grid()

    if percent:
        try:
            num_data = len(labels)
        except TypeError:
            num_data = labels.shape[0]

        train_sizes = [size / num_data for size in train_sizes]
        xlabel = "Data Used in Training (%)"
    else:
        xlabel = "Number of Training Data Points"

    matplotlib.pyplot.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    matplotlib.pyplot.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    matplotlib.pyplot.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training Score")
    matplotlib.pyplot.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-Validation Score")

    if title is None:
        title = "Model Training Curve"
    matplotlib.pyplot.xlabel(xlabel)
    matplotlib.pyplot.ylabel("Score (%)")
    matplotlib.pyplot.title(title)

    matplotlib.pyplot.legend(loc="best")
    matplotlib.pyplot.show()