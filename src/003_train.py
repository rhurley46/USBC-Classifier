import argparse
from typing import Text
import yaml
from src.utils.logs import get_logger
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
import numpy as np
from sklearn.pipeline import Pipeline

# Evaluation
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold


# Evaluate model
def evaluate_model(X, y, model):
    """
    *Reference sklearn.metrics.SCORERS.keys() for list of available scorers*
    """
    # Define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=143)
    # Evaluate model
    scores = cross_validate(model, X, y, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'], cv=cv, n_jobs=4)
    return scores

# Models dictionary
def get_models():
    models_dict = {
        'XGB': XGBClassifier(n_estimators=1000)
        }
    return models_dict


def train(config_path: Text) -> None:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('TRAINING', log_level=config['base']['log_level'])

    logger.info('Get dataset')
    feat_df = pd.read_csv(config['featurised_data']['train'])

    #train test split
    X = feat_df.iloc[:, :-1]
    y = feat_df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

    # Define models
    models = get_models()

    # Evaluate models
    results = {}

    for k, v in models.items():
        # Build pipeline
        clf = Pipeline(steps=[
            ('scaler', RobustScaler()),
            ('classifier', v)
        ])
        
        # Evaluate model
        scores = evaluate_model(X_train, y_train, clf)
        results[k] = scores
        name = k
        print(f"{name}\n{'-' * len(name)}")
        print('Mean Accuracy: %.3f (%.3f)' % (np.mean(scores['test_accuracy']), np.std(scores['test_accuracy'])))
        print('Mean F1: %.3f (%.3f)' % (np.mean(scores['test_f1']), np.std(scores['test_f1'])))
        print('Mean Recall: %.3f (%.3f)' % (np.mean(scores['test_recall']), np.std(scores['test_recall'])))
        print('Test Precision: %.3f (%.3f)' % (np.mean(scores['test_precision']), np.std(scores['test_precision'])))
        print('Mean ROC-AUC: %.3f (%.3f)\n' % (np.mean(scores['test_roc_auc']), np.std(scores['test_roc_auc'])))
    
    filename = config['models']['model']
    pickle.dump(clf, open(filename, 'wb'))

if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train(config_path=args.config)
