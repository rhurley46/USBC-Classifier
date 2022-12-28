import argparse
from typing import Text
import yaml
from src.utils.logs import get_logger
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def featurise(config_path: Text) -> None:
    """Feature engineering.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('FEATURE ENGINEERING', log_level=config['base']['log_level'])

    logger.info('Get dataset')
    train_pp = pd.read_csv(config['preprocessed_data']['train'])
    test_pp = pd.read_csv(config['preprocessed_data']['test'])

    #keep only certain features
    keep_features = config['keep_features']
    train_pp = train_pp[keep_features]
    test_pp = test_pp[keep_features]

    # convert income to binary
    train_pp['income']=train_pp['income'].map({'< $50k': 0, '> $50k': 1})
    test_pp['income']=test_pp['income'].map({'< $50k': 0, '> $50k': 1})

    # harmonise 'education' category
    train_pp['education'].replace([' Children', ' Less than 1st grade', ' 1st 2nd 3rd or 4th grade', ' 5th or 6th grade', ' 7th and 8th grade', ' 9th grade', ' 10th grade', ' 11th grade', ' 12th grade no diploma'], 'School', inplace=True)
    train_pp['education'].replace(' High school graduate', "High School", inplace=True)
    train_pp['education'].replace([' Some college but no degree', ' Associates degree-occup /vocational', ' Associates degree-academic program'], "Higher", inplace=True)
    train_pp['education'].replace(" Bachelors degree(BA AB BS)", "Undergraduate", inplace=True)
    train_pp['education'].replace(" Masters degree(MA MS MEng MEd MSW MBA)", "Graduate", inplace=True)
    train_pp['education'].replace(' Doctorate degree(PhD EdD)', "Doctorate", inplace=True)
    train_pp['education'].replace(" Prof school degree (MD DDS DVM LLB JD)", "PostDoctorate", inplace=True)

    # harmonise 'education' category
    test_pp['education'].replace([' Children', ' Less than 1st grade', ' 1st 2nd 3rd or 4th grade', ' 5th or 6th grade', ' 7th and 8th grade', ' 9th grade', ' 10th grade', ' 11th grade', ' 12th grade no diploma'], 'School', inplace=True)
    test_pp['education'].replace(' High school graduate', "High School", inplace=True)
    test_pp['education'].replace([' Some college but no degree', ' Associates degree-occup /vocational', ' Associates degree-academic program'], "Higher", inplace=True)
    test_pp['education'].replace(" Bachelors degree(BA AB BS)", "Undergraduate", inplace=True)
    test_pp['education'].replace(" Masters degree(MA MS MEng MEd MSW MBA)", "Graduate", inplace=True)
    test_pp['education'].replace(' Doctorate degree(PhD EdD)', "Doctorate", inplace=True)
    test_pp['education'].replace(" Prof school degree (MD DDS DVM LLB JD)", "PostDoctorate", inplace=True);

    # harmonise 'marital stat' category
    train_pp['marital stat'].replace([' Married-spouse absent', ' Married-A F spouse present', ' Married-civilian spouse present'], 'Married', inplace=True)
    test_pp['marital stat'].replace([' Married-spouse absent', ' Married-A F spouse present', ' Married-civilian spouse present'], 'Married', inplace=True)


    # harmonise 'citizen stat' category
    train_pp['citizenship'].replace([' Native- Born abroad of American Parent(s)', ' Native- Born in Puerto Rico or U S Outlying', ' Native- Born in the United States'], 'US Native', inplace=True)
    train_pp['citizenship'].replace([' Foreign born- Not a citizen of U S ', ' Foreign born- U S citizen by naturalization'], 'Non-US Native', inplace=True)
    # harmonise 'citizen stat' category
    test_pp['citizenship'].replace([' Native- Born abroad of American Parent(s)', ' Native- Born in Puerto Rico or U S Outlying', ' Native- Born in the United States'], 'US Native', inplace=True)
    test_pp['citizenship'].replace([' Foreign born- Not a citizen of U S ', ' Foreign born- U S citizen by naturalization'], 'Non-US Native', inplace=True)

    #drop country of birth
    train_pp = train_pp.drop(columns=['country of birth self'])
    #drop country of birth
    test_pp = test_pp.drop(columns=['country of birth self'])

    # encode labels
    train_pp = train_pp.apply(LabelEncoder().fit_transform)
    test_pp = test_pp.apply(LabelEncoder().fit_transform)

    #save featurised dataset
    train_pp.to_csv(config['featurised_data']['train'], index=None)
    test_pp.to_csv(config['featurised_data']['test'], index=None)

if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    featurise(config_path=args.config)
