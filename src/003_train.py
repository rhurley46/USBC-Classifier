import argparse
from typing import Text
import yaml
from src.utils.logs import get_logger
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


# Evaluation
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold


# load dataset 
def load_dataset(full_path):
    dataframe = pd.read_csv(full_path, na_values=' ?')
    dataframe = dataframe.dropna()
    X, y= dataframe.iloc[:, :-1], dataframe.iloc[:, -1]
    cat_ix = X.select_dtypes(include=['object', 'bool']).columns
    num_ix = X.select_dtypes(include=['int64', 'float64']).columns
    y = LabelEncoder().fit_transform(y)
    return X, y, cat_ix, num_ix

def evaluate_model(X, y, model):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores

def get_models():
    models, names = list(), list()
    models.append(DecisionTreeClassifier())
    names.append("CART")
    # models.append(SVC(gamma='scale'))
    # names.append("SVM")
    # models.append(BaggingClassifier(n_estimators=100))
    # names.append("BAG")
    models.append(RandomForestClassifier(n_estimators=100))
    names.append("RF")
    models.append(GradientBoostingClassifier(n_estimators=100))
    names.append("GBM")
    return models, names


def train(config_path: Text) -> None:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('TRAINING', log_level=config['base']['log_level'])

    logger.info('Load dataset')
    X, y, cat_ix, num_ix = load_dataset(full_path=config['preprocessed_data']['train'])

    models, names = get_models()

    results = list()
    for i in range(len(models)):
        steps = [
            ('c', OneHotEncoder(handle_unknown='ignore'), cat_ix),
            ('n', MinMaxScaler(), num_ix)
        ]
        ct = ColumnTransformer(steps)
        pipeline = Pipeline(
            steps=[
                ('t', ct),
                ('m', models[i])]
        )
        scores = evaluate_model(X, y, pipeline)
        results.append(scores)
        print(f"{names[i]}: {np.mean(scores)} ({np.std(scores)})")

    plot_path = config['outputs']['plots'] + 'model_selection.png'
    plt.boxplot(results, labels=names, showmeans=True)
    plt.show()
    # plt.savefig(plot_path)


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train(config_path=args.config)
