import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title('Predict Income')

col_names = [
  "age",
  "class of worker",
  "detailed industry recode",
  "detailed occupation recode",
  "education",
  "wage per hour",
  "enroll in edu inst last wk",
  "marital stat",
  "major industry code",
  "major occupation code",
  "race",
  "hispanic origin",
  "sex",
  "member of a labor union",
  "reason for unemployment",
  "full or part time employment stat",
  "capital gains",
  "capital losses",
  "dividends from stocks",
  "tax filer stat",
  "region of previous residence",
  "state of previous residence",
  "detailed household and family stat",
  "detailed household summary in household",
  "instance weight",
  "migration code-change in msa",
  "migration code-change in reg",
  "migration code-move within reg",
  "live in this house 1 year ago",
  "migration prev res in sunbelt",
  "num persons worked for employer",
  "family members under 18",
  "country of birth father",
  "country of birth mother",
  "country of birth self",
  "citizenship",
  "own business or self employed",
  "fill inc questionnaire for veteran's admin",
  "veterans benefits",
  "weeks worked in year",
  "year",
  "income"]

pipeline = joblib.load(open("/Users/rory.hurley/Documents/GitHub/uscb_classifier/pipelines/pipeline.pkl", "rb"))

data = pd.read_csv("/Users/rory.hurley/Documents/GitHub/uscb_classifier/data/raw/census_income_test.csv", names=col_names, na_values=' ?')

cat_cols = ['class of worker', 'education', 'enroll in edu inst last wk',
       'marital stat', 'major industry code', 'major occupation code', 'race',
       'hispanic origin', 'sex', 'member of a labor union',
       'reason for unemployment', 'full or part time employment stat',
       'tax filer stat', 'region of previous residence',
       'state of previous residence', 'detailed household and family stat',
       'detailed household summary in household',
       'migration code-change in msa', 'migration code-change in reg',
       'migration code-move within reg', 'live in this house 1 year ago',
       'migration prev res in sunbelt', 'family members under 18',
       'country of birth father', 'country of birth mother',
       'country of birth self', 'citizenship',
       "fill inc questionnaire for veteran's admin"]

num_cols = ['age', 'detailed industry recode', 'detailed occupation recode',
       'wage per hour', 'capital gains', 'capital losses',
       'dividends from stocks', 'instance weight',
       'num persons worked for employer', 'own business or self employed',
       'veterans benefits', 'weeks worked in year', 'year']
    
options_dict = {}
for i in cat_cols:
    options_dict[i] = data[i].unique()
    

['class of worker', 'education', 'enroll in edu inst last wk',
       'marital stat', 'major industry code', 'major occupation code', 'race',
       'hispanic origin', 'sex', 'member of a labor union',
       'reason for unemployment', 'full or part time employment stat',
       'tax filer stat', 'region of previous residence',
       'state of previous residence', 'detailed household and family stat',
       'detailed household summary in household',
       'migration code-change in msa', 'migration code-change in reg',
       'migration code-move within reg', 'live in this house 1 year ago',
       'migration prev res in sunbelt', 'family members under 18',
       'country of birth father', 'country of birth mother',
       'country of birth self', 'citizenship',
       "fill inc questionnaire for veteran's admin"]

# options_dict = {}
# for i in cat_cols:
#     options_dict[i] = st.selectbox(i,data[i].unique())
#     option = st.selectbox(
#         i,
#         data[i].unique())
