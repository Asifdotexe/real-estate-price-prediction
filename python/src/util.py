import re
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

def convert_sqft(value: str) -> float | None:
    """Convert various area representations to square feet

    Args:
        value (str): Area string, possibly a range or with units.

    Returns:
        float | None: Converted area in square feet, or None if unconvertible.
    """
    # Non string or empty inputs can't be processed meaningfully.
    if not isinstance(value, str) or not value.strip():
        return None

    value = value.strip()

    # If the value is a range like "2100 - 2850",
    # we approximate by averaging the two numbers
    if '-' in value:
        tokens = value.strip('-')
        if len(tokens) == 2:
            first = float(tokens[0].strip())
            second = float(tokens[1].strip())
            return (first + second) / 2

    # Extract the first numeric part to interpret the value, even if it includes a unit.
    numeric_match = re.match(r"([\d\.]+)", value)
    if not numeric_match:
        # If there is no number, we cannot proceed with conversions.
        return None

    try:
        # Defensive check in case float conversion fails
        numeric_value = float(numeric_match.group(1))
    except ValueError:
        return None

    value_lower = value.lower()

    # Define unit to sqft conversion mapping
    unit_to_sqft = {
        'sq. meter': 10.7639,
        'sq meter': 10.7639,
        'sqm': 10.7639,
        'sq. yard': 9,
        'sq yards': 9,
        'sq yard': 9,
        # fallback for less specific yard mentions
        'yard': 9,
        'acre': 45560,
        'ground': 2400,
        'guntha': 1089,
        'cent': 439.6,
        # fallback for less specific meter mentions
        'meter': 10.7639
    }

    # Checking if any known unit is in the string and convert accordingly
    for unit, factor in unit_to_sqft.items():
        if unit in value_lower:
            return numeric_value * factor

    # Handling 'perch' specifically (damn this was confusing).
    # Skip conversion as it's unclear
    if 'perch' in value_lower:
        return None

    # If no unit matched, assume it is already in sqft.
    return numeric_value

def check_imbalance(df, class_column='class'):
    no_of_true = len(df.loc[df[class_column] == True])
    no_of_false = len(df.loc[df[class_column] == False])

    true_ratio = (no_of_true / (no_of_true + no_of_false))
    false_ratio = (no_of_false / (no_of_false + no_of_true))

    print(f"Number of true: {no_of_true} ({round(true_ratio, 4) * 100}%)")
    print(f"Number of false: {no_of_false} ({round(false_ratio, 5) * 100}%)")

def random_forest_tuning(x_train, x_test, y_train, y_test, n, d, l, seed):

    """
    Evaluate the performance of a Random Forest model with different hyperparameters.

    Parameters:
    - X_train: Training features
    - X_test: Testing features
    - y_train: Training labels
    - y_test: Testing labels
    - n: List of the number of trees in the forest
    - d: List of the maximum depth of the tree
    - l: List of the minimum samples required to be at a leaf node
    - seed: Random seed for reproducibility

    Returns:
    - model_performance_df: DataFrame containing the model performance metrics
    """
    model_performance = []

    for i in n:
        for j in d:
            for k in l:
                # Create and train the Random Forest model
                random_forest = RandomForestClassifier(
                    n_estimators=i,
                    max_depth=j,
                    min_samples_leaf=k,
                    random_state=seed
                )
                random_forest.fit(x_train, y_train.ravel())

                # Predict probabilities on the training and testing sets
                train_pred = random_forest.predict_proba(x_train)
                test_pred = random_forest.predict_proba(x_test)

                # Create a unique identifier for the current set of hyperparameters
                t1 = f'trees{i}_maxDepth{j}_minLeaf{k}'

                # Calculate and store AUC-ROC scores for training and testing sets
                t2 = [t1, round(roc_auc_score(y_train, train_pred[:, 1]), 4), round(roc_auc_score(y_test, test_pred[:, 1]), 4)]
                model_performance.append(t2)

    # Create a DataFrame from the collected performance metrics
    model_performance_df = pd.DataFrame(model_performance)
    model_performance_df.rename(columns={0: 'parameter', 1: 'train_auc', 2: 'test_auc'}, inplace=True)

    return model_performance_df

def get_feature_importance(X_train, model, top_n=5):
    """
    Get feature importances from a trained model and return the top N features.

    Parameters:
    - X_train: DataFrame containing the training features
    - model: Trained model with a `feature_importances_` attribute
    - top_n: Number of top features to retrieve (default is 5)

    Returns:
    - imp_feat_df: DataFrame with the top N features and their importances
    """
    feature_importances = model.feature_importances_

    t1 = []
    for i in range(len(X_train.columns)):
        t2 = [X_train.columns[i], feature_importances[i]]
        t1.append(t2)

    imp_feat_df = pd.DataFrame(t1, columns=['name', 'importance'])
    imp_feat_df = imp_feat_df.sort_values(by=['importance'], ascending=False).head(top_n)

    return imp_feat_df

def print_categorical_value_counts(df, cat_cols):
    """
    Print value counts for each categorical column in the DataFrame.

    Parameters:
    - df: DataFrame
    - cat_cols: List of categorical column names

    Returns:
    - None
    """
    for col in cat_cols:
        print("=" * 15)
        print(col)
        print("--" * 5)
        print(df[col].value_counts())