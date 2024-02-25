import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_corr(corr, annot=True, cmap='RdYlGn', center=0, square=True):
    """
    Plot a heatmap correlation plot.

    Parameters:
    - corr: Correlation matrix
    - annot: Whether to annotate the heatmap with correlation values (default is True)
    - cmap: Colormap for the heatmap (default is 'RdYlGn')
    - center: Center value for the colormap (default is 0)
    - square: Whether to force the plot to be square (default is True)

    Returns:
    - None
    """
    # Creating an upper triangle mask
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Creating a heatmap correlation plot
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=annot, center=center, square=square)
    plt.tight_layout()
    plt.show()

def plot_roc_curve(fpr_list, tpr_list, roc_auc_list, labels):
    """
    Plot ROC curves for multiple models on the same graph.

    Parameters:
    - fpr_list: List of false positive rates for each model
    - tpr_list: List of true positive rates for each model
    - roc_auc_list: List of ROC AUC scores for each model
    - labels: List of labels for each model

    Returns:
    - None
    """
    plt.clf()

    for fpr, tpr, roc_auc, label in zip(fpr_list, tpr_list, roc_auc_list, labels):
        plt.plot(fpr, tpr, label=f'{label} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_histograms(df,column, figsize=(12, 10)):
    """
    Plot histograms for each column in the DataFrame.

    Parameters:
    - df: DataFrame
    - column: list
    - figsize: Tuple, optional, default is (12, 10)

    Returns:
    - None
    """
    df[column].hist(figsize=(12,10))
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Plot a confusion matrix.

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - labels: List of class labels (default is None)

    Returns:
    - None
    """
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    display_labels = labels if labels is not None else ['True', 'False']

    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    cm_display.plot()

    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.linear_model import LinearRegression

def visualize_cross_validation(model, X, y, n_splits=10, test_size=0.2, random_state=0):
    """
    Visualize cross-validation scores for a given model using a boxplot.

    Parameters:
    - model: The machine learning model to evaluate.
    - X: The feature matrix.
    - y: The target variable.
    - n_splits: Number of shuffle splits in cross-validation.
    - test_size: Proportion of the dataset to include in the test split.
    - random_state: Seed for reproducibility.

    Returns:
    - None (Displays boxplot and prints summary statistics).
    """
    # Creating a ShuffleSplit object
    cross_validation = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    # Performing cross-validation and getting the scores
    cv_scores = cross_val_score(model, X, y, cv=cross_validation)

    # Displaying summary statistics
    print(f'We ran {n_splits} shuffle splits')
    print('Minimum cross-validation score is', min(cv_scores))
    print('Mean cross-validation score is', cv_scores.mean())
    print('Maximum cross-validation score is', max(cv_scores))

    # Creating a boxplot to visualize the cross-validation scores
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=cv_scores)
    plt.title(f'{model.__class__.__name__} Cross-Validation Scores')
    plt.xlabel('Cross-Validation Splits')
    plt.ylabel('R^2 Score')  # Adjust the label based on your evaluation metric
    plt.show()
