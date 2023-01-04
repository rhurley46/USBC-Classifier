import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yaml
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

with open('/Users/rory.hurley/Documents/GitHub/uscb_classifier/params.yaml') as conf_file:
    config = yaml.safe_load(conf_file)

st.title('Income Classifier Tool 🤑')
st.write("Use this tool to predict whether an individual earns more or less than $50,000")

pipeline = joblib.load(open(config['pipeline']['tuned_pipeline'], "rb"))

# load dataset 
def load_dataset(full_path):
    dataframe = pd.read_csv(full_path, na_values=' ?')
    dataframe = dataframe.dropna()
    X, y= dataframe.iloc[:, :-1], dataframe.iloc[:, -1]
    cat_ix = X.select_dtypes(include=['object', 'bool']).columns
    num_ix = X.select_dtypes(include=['int64', 'float64']).columns
    y = LabelEncoder().fit_transform(y)
    return X, y, cat_ix, num_ix

X, y, cat_ix, num_ix = load_dataset(full_path=config['preprocessed_data']['test'])
    
options_dict = {}
for i in cat_ix:
    options_dict[i] = X[i].unique()

col_age = st.slider('age', 0, 100, 25)
col_class_of_worker = st.selectbox('class of worker', options_dict['class of worker'])
col_education = st.selectbox('education', options_dict['education'])
col_marital_stat = st.selectbox('marital stat', options_dict['marital stat'])
col_major_industry_code = st.selectbox('major industry code', options_dict['major industry code'])
col_major_occupation_code = st.selectbox('major occupation code', options_dict['major occupation code'])
col_race = st.selectbox('race', options_dict['race'])
col_sex = st.selectbox('sex', options_dict['sex'])
col_capital_gains = st.slider('capital gains', 0, 10000000, 25)
col_capital_losses = st.slider('capital losses', 0, 10000000, 25)
col_dividends_from_stocks = st.slider('dividends from stocks', 0, 10000000, 25)
col_country_of_birth_self = st.selectbox('country of birth self', options_dict['country of birth self'])
col_citizenship = st.selectbox('citizenship', options_dict['citizenship'])
col_own_business_or_self_employed = st.selectbox('own business or self employed', [0,1,2])
col_weeks_worked_in_year = st.slider('weeks worked in year', 0, 52, 25)
col_wage_per_hour = st.slider('wage per hour', 0, 10000000, 25)


keys = list(X.columns)
values = [
    col_age,
    col_class_of_worker,
    col_education,
    col_marital_stat,
    col_major_industry_code,
    col_major_occupation_code,
    col_race,
    col_sex,
    col_capital_gains,
    col_capital_losses,
    col_dividends_from_stocks,
    col_country_of_birth_self,
    col_citizenship,
    col_own_business_or_self_employed,
    col_weeks_worked_in_year,
    col_wage_per_hour,
    ]


df = pd.DataFrame(dict(zip(keys, values)), index=[0])
st.write("Your selections are...")
st.write(df)

y_preds = pipeline.predict(df)

st.markdown("#### It is predicted that you earn...")
if y_preds == 0:
    st.markdown("## Less than $50,0000.")
else:
    st.markdown("## More than $50,0000!")

