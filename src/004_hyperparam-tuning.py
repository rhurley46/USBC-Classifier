import argparse
from typing import Text
import yaml
from src.utils.logs import get_logger
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
import joblib


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

def tuning(config_path: Text) -> None:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('TUNING', log_level=config['base']['log_level'])

    logger.info('Load dataset')
    X, y, cat_ix, num_ix = load_dataset(full_path=config['preprocessed_data']['train'])

    results = list()

    steps = [
        ('c', OneHotEncoder(handle_unknown='ignore'), cat_ix),
        ('n', MinMaxScaler(), num_ix)
    ]
    ct = ColumnTransformer(steps)
    pipeline = Pipeline(
        steps=[
            ('t', ct),
            ('m', GradientBoostingClassifier())]
    )

    grid_params = {
        'm__learning_rate': config['hyperparams']['learning_rate'],
        'm__n_estimators': config['hyperparams']['n_estimators'],
        'm__max_depth': config['hyperparams']['max_depth']
    }

    clf = GridSearchCV(pipeline, grid_params, verbose=5)
    clf.fit(X, y)

    print("Best Score: ", clf.best_score_)
    print("Best Params: ", clf.best_params_)

    # save best estimator
    logger.info('Saving best model')
    best_clf = clf.best_estimator_
    joblib.dump(best_clf, config['pipeline']['tuned_pipeline'])


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    tuning(config_path=args.config)