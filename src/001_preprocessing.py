import argparse
from typing import Text
import yaml
from src.utils.logs import get_logger
import pandas as pd
import numpy as np

def convert_dtypes(df, mapping):
# Convert object dtype to category
    for k in mapping:
        if mapping[k] == "continuous":
            df[k] = df[k].astype('int64')
        elif mapping[k] == "nominal":
            df[k] = df[k].astype('category')
    return df

def distribution_impute(df, col):
    # Record distribution
    attribute_dist = df[col].value_counts(normalize=True)

    # Impute NaN
    nulls = df[col].isna()
    df.loc[nulls, col] = np.random.choice(attribute_dist.index, size=len(df[nulls]), p=attribute_dist.values)

def preprocess(config_path: Text) -> None:
    """Load raw data and preprocess.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('Preprocess', log_level=config['base']['log_level'])

    # with open(config['col_types']) as file:
    #     cols = [line.rstrip() for line in file]

    cols = config['col_names']

    logger.info('Get dataset')
    train = pd.read_csv(config['raw_data']['train'], names=cols, na_values=' ?')
    test = pd.read_csv(config['raw_data']['test'], names=cols, na_values=' ?')

    col_types = config['col_types']

    # print(col_types)

    train = convert_dtypes(train, mapping=col_types)
    test = convert_dtypes(test, mapping=col_types)

    # remove `instance weight` column
    train = train.drop(columns=['instance weight'])
    test = test.drop(columns=['instance weight'])

    # find where there are duplicates (excluding 'income' column), check it matches expected value from metadata file
    train_dupes = train.loc[:, train.columns != 'income'].duplicated()
    test_dupes = test.loc[:, test.columns != 'income'].duplicated()
    # remove duplicates
    train = train[~train_dupes]
    test = test[~test_dupes]

    print(train.head())

    # convert income to binary
    train['income']=train['income'].map({' - 50000.': '< $50k', ' 50000+.': '> $50k'})
    test['income']=test['income'].map({' - 50000.': '< $50k', ' 50000+.': '> $50k'})

    # check nas and proportions
    isna_df = pd.DataFrame(
        {"train_nas": train.isna().sum(), 
        "test_nas": test.isna().sum(),
        "prop_train_na": (train.isna().sum().values ) / len(train) * 100,
        "prop_test_na": (test.isna().sum().values ) / len(test) * 100 
        })
    isna_df = isna_df[(isna_df['train_nas'] > 0) | (isna_df['test_nas'] > 0)]

    isna_df_cols = list(isna_df.index)
    # Impute Nan
    for col in isna_df_cols:
        distribution_impute(train, col)
        distribution_impute(test, col)

    # save preprocessed data
    train.to_csv(config['preprocessed_data']['train'], index=None)
    test.to_csv(config['preprocessed_data']['test'], index=None)


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    preprocess(config_path=args.config)
