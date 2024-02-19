import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#streamlit run "c:/Users/loris/Desktop/Project ER/Presentation.py"

st.write("Dataset source: [click link](https://databank.worldbank.org/source/world-development-indicators)")

# Load the dataset after the merge and some clean operations
@st.cache_data()

def load_data(cc):
    data = pd.read_csv(cc)
    return data

st.header("Analysis of economic indices and GDP forecast")

Glob_df = load_data("C:/Users/loris/Desktop/Project ER/Glob_df.csv")
Glob_df2 = load_data("C:/Users/loris/Desktop/Project ER/Glob_df2.csv")

###### Sidebar with controls ######
st.sidebar.subheader("Controls")

# GLOB DF structure
show_raw_data_Glob_df = st.sidebar.checkbox("Show data for Glob_df")
# If the checkbox for Glob_df is selected, show sub-controls
if show_raw_data_Glob_df:
    st.sidebar.subheader("Subcontrols for Glob_df")
    show_raw_data1 = st.sidebar.checkbox("Missing Values Summary")
    show_raw_data2 = st.sidebar.checkbox("Show raw data 2")

# GLOB DF2 structure
show_raw_data_Glob_df2 = st.sidebar.checkbox("Show data for Glob_df2")
# If the checkbox for Glob_df2 is selected, show sub-controls
if show_raw_data_Glob_df2:
    st.sidebar.subheader("Subcontrols for Glob_df2")
    show_raw_data3 = st.sidebar.checkbox("Missing Values Summary")
    show_raw_data4 = st.sidebar.checkbox("Show raw data 2")

#####################################
# Glob_df
if show_raw_data_Glob_df:
    st.subheader("Glob_df")
    st.write(Glob_df)

    # If the checkbox for missing values summary is selected, show the summary
    if show_raw_data1:
        st.subheader("Missing Values Summary for Glob_df")
        st.write(Glob_df.isna().sum())

# Glob_df2
if show_raw_data_Glob_df2:
    st.subheader("Glob_df2")
    st.write(Glob_df2)

    # If the checkbox for missing values summary is selected, show the summary
    if show_raw_data3:
        st.subheader("Missing Values Summary for Glob_df2")
        st.write(Glob_df2.isna().sum())
####################################
