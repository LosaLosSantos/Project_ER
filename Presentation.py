import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from PIL import Image

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
Glob_df3 = load_data("C:/Users/loris/Desktop/Project ER/Glob_df3.csv")
Top_gdp = load_data("C:/Users/loris/Desktop/Project ER/top_gdp_countries.csv")
Glob_df4_reset = load_data("C:/Users/loris/Desktop/Project ER/Glob_df4_reset.csv")
Glob_df2 = Glob_df2.set_index(["Code", "Year"])
Glob_df3 = Glob_df3.set_index(["Code", "Year"])
Top_gdp = Top_gdp.set_index(["Code", "Year"])

###############################
# Sidebar with controls
st.sidebar.header("Controls")

############# Group for Glob_df
show_data_Glob_df = st.sidebar.checkbox("Glob_df", key="df")

if show_data_Glob_df:
    st.subheader("Glob_df")
    st.dataframe(Glob_df)
    st.sidebar.subheader("Subcontrols")
    show_data1 = st.sidebar.checkbox("Missing Values for Glob_df1", key="1_1")
    if show_data1:
        st.subheader("Missing Values for Glob_df")
        st.write(Glob_df.isna().sum())

############## Group for Glob_df2
show_data_Glob_df2 = st.sidebar.checkbox("Glob_df2", key="df2")

if show_data_Glob_df2:
    st.subheader("Glob_df2")
    st.dataframe(Glob_df2)
    st.sidebar.subheader("Subcontrols")
    show_data2_1 = st.sidebar.checkbox("Missing Values for Glob_df2", key="2_1")
    if show_data2_1:
        st.subheader("Missing Values for Glob_df2")
        st.write(Glob_df2.isna().sum())
    show_data2_2 = st.sidebar.checkbox("Missing Values per Year for Glob_df2", key="2_2")
    if show_data2_2:
        st.subheader("Missing Values per Year for Glob_df2")
        st.write(Glob_df2.isna().groupby("Year").sum())
    show_data2_3 = st.sidebar.checkbox("Index", key="2_3")
    if show_data2_3:
        st.subheader("Index for Glob_df2")
        st.write(Glob_df2.index)

    
############# Group for Glob_df3
show_data_Glob_df3 = st.sidebar.checkbox("Glob_df3", key="df3")

if show_data_Glob_df3:
    st.subheader("Glob_df3")
    st.dataframe(Glob_df3)
    st.sidebar.subheader("Subcontrols")
    show_data3_1 = st.sidebar.checkbox("Missing Values for Glob_df3", key="3_1")
    if show_data3_1:
        st.subheader("Missing Values for Glob_df3")
        st.write(Glob_df3.isna().sum())
    show_data3_2 = st.sidebar.checkbox("Columns", key="3_2")
    if show_data3_2:
        st.subheader("Columns for Glob_df3")
        st.write(Glob_df3.columns)
    show_data3_3 = st.sidebar.checkbox("The best 10 GDP per capita", key="3_3")
    if show_data3_3:
        st.subheader("The best 10 GDP per capita")
        st.write(Glob_df3.loc[Glob_df3.index.get_level_values("Year") == 2021, "GDP_pc"].nlargest(10))
    

show_matrix = st.sidebar.checkbox("Correlation Matrix of columns", key="corr")
if show_matrix:
    correlation_matrix = Glob_df3.corr()
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(20, 16))
    sns.heatmap(correlation_matrix, annot=True, ax=ax,cmap="coolwarm", fmt=".2f")
    st.pyplot(fig)

show_top_gdp = st.sidebar.checkbox("Highest GDP (c. 2015 $) in 2021", key="top")
if show_top_gdp:
    st.subheader("Highest GDP (c. 2015 $) in 2021")
    st.write(Top_gdp["GDP (constant 2015 US$)"])

###################################GRAPHS 1############################################

image_files = ["The top 9 of highest GDP (constant 2015 US$)", "The top 9 for Total natural resources rents",
               "The top 9 for Military Expenditure","The top 9 for Research and development expenditure",
               "The top 9 for Current health expenditure", "The best 15 GDP in the world (2021)"]

# Drop-down menu to select an image
selected_image = st.selectbox("Plots using the 15 highest GDP (c. 2015 $) in the world", image_files, index=None)

image_paths = {"The top 9 of highest GDP (constant 2015 US$)": "C:\\Users\\loris\Desktop\\Project ER\\Graphs\\The top 9 of highest GDP (constant 2015 US$).png",
    "The top 9 for Total natural resources rents": "C:\\Users\\loris\Desktop\\Project ER\\Graphs\\The top 9 for Total natural resources rents.png",
    "The top 9 for Military Expenditure": "C:\\Users\\loris\Desktop\\Project ER\\Graphs\\The top 9 for Military Expenditure.png",
    "The top 9 for Research and development expenditure": "C:\\Users\\loris\Desktop\\Project ER\\Graphs\\The top 9 for Research and development expenditure.png",
    "The top 9 for Current health expenditure": "C:\\Users\\loris\Desktop\\Project ER\\Graphs\\The top 9 for Current health expenditure.png",
    "The best 15 GDP in the world (2021)": "C:\\Users\\loris\Desktop\\Project ER\\Graphs\\The best 15 GDP in the world (2021).png"}

#Load and visualize the image
if selected_image:
    image_path = image_paths[selected_image]
    image = Image.open(image_path)
    st.image(image, caption=selected_image)

###################################GRAPHS 2############################################

image_files2 = ["Population Distribution by Development Class (1998)(2021)", 
                "Number of countries per development class (1998)(2021)",
                "Average GDP from Natural Resources Rents by Development Class (1998 vs 2021)"]

# Drop-down menu to select an image
selected_image2 = st.selectbox("Some Macrotrends", image_files2, index=None)

image_paths2 = {"Population Distribution by Development Class (1998)(2021)": "C:\\Users\\loris\Desktop\\Project ER\\Graphs\\Population Distribution by Development Class (1998)(2021).png",
    "Number of countries per development class (1998)(2021)": "C:\\Users\\loris\Desktop\\Project ER\\Graphs\\Number of countries per development class (1998)(2021).png",
    "Average GDP from Natural Resources Rents by Development Class (1998 vs 2021)": "C:\\Users\\loris\Desktop\\Project ER\\Graphs\\Average GDP from Natural Resources Rents by Development Class (1998 vs 2021).png"}

#Load and visualize the image
if selected_image2:
    image_path2 = image_paths2[selected_image2]
    image2 = Image.open(image_path2)
    st.image(image2, caption=selected_image2)

############################# MODEL #################################

features = ["GDP (current US$)", "Merchandise imports (current US$)",
                         "Mineral rents (% of GDP)", "Natural gas rents (% of GDP)",
                         "Population, total", "Total natural resources rents (% of GDP)",
                         "Human Development Index", "Forest rents (% of GDP)"]

dependent_variable = "GDP (constant 2015 US$)"

X = Glob_df4_reset[features]
y = Glob_df4_reset[dependent_variable]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

show_model = st.sidebar.checkbox("Model", key="mod")
if show_model:
    #Print the model metrics and coefficients
    st.subheader("Model Evaluation")
    st.write("Mean Squared Error:", mse)
    st.write("R^2 Score:", r2)
    st.write("Coefficients:")
    for feature, coef in zip(features, model.coef_):
        st.write(f"{feature}: {coef}")

    selected_country_code = st.selectbox("Select Country Code", Glob_df4_reset["Code"].unique())
    selected_country_features = Glob_df4_reset[Glob_df4_reset["Code"] == selected_country_code][features]
    #Predict for the selected country
    predicted_gdp = model.predict(selected_country_features)

    st.subheader(f"Predicted GDP for {selected_country_code}:")
    predicted_df = pd.DataFrame(predicted_gdp, columns=["Predicted GDP"])
    st.dataframe(predicted_df.style.format({"Predicted GDP": "{:.0f}"}))