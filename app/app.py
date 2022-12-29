import streamlit as st
import pandas as pd
import numpy as np

st.title('Uber pickups in NYC')

data = pd.read_csv("/Users/rory.hurley/Documents/GitHub/uscb_classifier/data/preprocessed/processed_data.csv")
category_cols = list(data.select_dtypes(['category']).columns)
coldict = {}
for i in category_cols:
    coldict[i] = data[i].unique()

option = st.selectbox(
    "class of worker",
    ["a", "b", "c"])
    # ('Email', 'Home phone', 'Mobile phone'))

st.write('You selected:', option)