import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
Dev_df = pd.read_csv("C:/Users/loris/Desktop/Project ER/Develop Ind.csv")
HDI_df = pd.read_csv("C:/Users/loris/Desktop/Project ER/human-development-index.csv")

HDI_df
HDI_df.shape
#Observing the structure of the dataset
HDI_df.head()
print(HDI_df.tail())
HDI_df[5890:5923]
HDI_df.info()
#Observing the structure of the dataset
Dev_df.head()
Dev_df.info()
print(Dev_df.shape)
Dev_df.tail()
Glob_df = Dev_df.copy()
Glob_df.rename(columns={"Country Code": "Code"}, inplace=True)
Glob_df.rename(columns={"Time": "Year"}, inplace=True)
Glob_df.isnull().sum()
Glob_df[6645:6655]
Glob_df[5424:6656]
#Upon inspecting NaN data, it has been observed that there are also aggregates of geographic areas.
#Therefore, deleting all rows below the last country in the dataset
Glob_df.drop(Glob_df.index[5425:6656], inplace=True)
print(Glob_df.shape)
print(HDI_df.shape)
#Now changing the type of Year in Glob_df to match the type of HDI_df.
Glob_df["Year"] = Glob_df["Year"].astype(int)

print(f"{Glob_df.columns}\n{HDI_df.columns}")
#Setting index
Glob_df = Glob_df.set_index(["Code", "Year"])
HDI_df = HDI_df.set_index(["Code", "Year"])
print(HDI_df)
print(Glob_df)
#Checking for null values before the merge.
print(f"{Glob_df.isnull().sum()}\n{HDI_df.isnull().sum()}")
print(HDI_df[5855:5940])
#Performing the merge using how="left" to utilize only keys from the left frame and using the index of both datasets for the merge.
Glob_df = Glob_df.merge(HDI_df["Human Development Index"], left_index=True, right_index=True, how="left")
#Checking for NaN values created by the merge
Glob_df["Human Development Index"].isnull().value_counts()

Glob_df.info()
Glob_df.to_csv("Glob_df.csv", index= True)
HDI_nan_countries = Glob_df.groupby("Code")["Human Development Index"].agg(lambda x : x.isnull().sum())
HDI_nan_countries
#Observing the number of countries with no data for the HDI
HDI_nan_countries[HDI_nan_countries == 25].index

#This country code was not found in HDI_df. Confirmed by checking examples: HDI_df.loc["GUM"] and HDI_df.loc["SXM"]
#The other countries with NaN values are due to a mismatch between years in the two merged datasets.
#(Every country has at least one because the year 2022 was not present in the HDI_df).
#The decision has been made to exclude the year 2022 and retain only countries with less than 5 NaN values.

HDI_df.loc["AGO"]
HDI_df.loc["GNB"]
HDI_df.loc["AND"]
#Countries with more than 4 NaN values in the HDI column have been identified
HDI_too_nan = HDI_nan_countries[(HDI_nan_countries > 5)]

#Checking the countries with too many NaN values in the HDI column
HDI_too_nan.index
#Creating the mask for the countries to retain.
HDI_few_nan = HDI_nan_countries[HDI_nan_countries <= 4]
HDI_few_nan
#Applying the boolean mask to filter the desired countries
Glob_df2 = Glob_df[Glob_df.index.get_level_values("Code").isin(HDI_few_nan.index)]

Glob_df2.isna().sum()
#Replacing dots with NaN throughout the dataframe to estimate the real number of NaN values in the columns
Glob_df2.replace("..", np.nan, inplace=True)

Glob_df2.isna().sum()
#The year 2022 contains too many NaN values in the columns essential for my analysis. 
#Additionally, the HDI does not have values for any country in this year.
Glob_df2.isna().groupby("Year").sum()
Glob_df2.drop(index=2022, level="Year", inplace = True) 
print(Glob_df2.columns)
#Dropping columns that are not essential for the analysis and contain too many NaN values
columns_to_drop = ["Country Name", "Time Code", "Arms exports (SIPRI trend indicator values) [MS.MIL.XPRT.KD]",
"Arms imports (SIPRI trend indicator values) [MS.MIL.MPRT.KD]", "Consumer price index (2010 = 100) [FP.CPI.TOTL]",
"Cost of business start-up procedures, female (% of GNI per capita) [IC.REG.COST.PC.FE.ZS]",
"Coverage of social insurance programs (% of population) [per_si_allsi.cov_pop_tot]",
"Current education expenditure, total (% of total expenditure in public institutions) [SE.XPD.CTOT.ZS]",
"Domestic credit to private sector (% of GDP) [FS.AST.PRVT.GD.ZS]", "Gross savings (% of GDP) [NY.GNS.ICTR.ZS]", 
"Human capital index (HCI) (scale 0-1) [HD.HCI.OVRL]", "Manufacturing, value added (% of GDP) [NV.IND.MANF.ZS]", 
"Services, value added (% of GDP) [NV.SRV.TOTL.ZS]", "Literacy rate, adult total (% of people ages 15 and above) [SE.ADT.LITR.ZS]",
"Central government debt, total (% of GDP) [GC.DOD.TOTL.GD.ZS]", "Interest payments (% of expense) [GC.XPN.INTP.ZS]",
"Researchers in R&D (per million people) [SP.POP.SCIE.RD.P6]", "Expense (% of GDP) [GC.XPN.TOTL.GD.ZS]",
"Labor force, total [SL.TLF.TOTL.IN]", "Land area (sq. km) [AG.LND.TOTL.K2]",
"Energy imports, net (% of energy use) [EG.IMP.CONS.ZS]", "Fossil fuel energy consumption (% of total) [EG.USE.COMM.FO.ZS]",
"Food exports (% of merchandise exports) [TX.VAL.FOOD.ZS.UN]", "Food imports (% of merchandise imports) [TM.VAL.FOOD.ZS.UN]",
"Population, female (% of total population) [SP.POP.TOTL.FE.ZS]", "Population, male (% of total population) [SP.POP.TOTL.MA.ZS]",
"Agricultural raw materials imports (% of merchandise imports) [TM.VAL.AGRI.ZS.UN]", 
"Agricultural raw materials exports (% of merchandise exports) [TX.VAL.AGRI.ZS.UN]", "Surface area (sq. km) [AG.SRF.TOTL.K2]"]

Glob_df2 = Glob_df2.drop(columns=columns_to_drop)
Glob_df2.info()
print(Glob_df2.columns)
Glob_df2.to_csv("Glob_df2.csv", index= True)
#Observing countries with NaN values in the GDP (current US$) column.
GDPcurrent_mask = Glob_df2.groupby("Code")["GDP (current US$) [NY.GDP.MKTP.CD]"].agg(lambda x : x.isnull().sum())
GDPcurrent_mask[GDPcurrent_mask != 0]
#Obtaining an overview of the countries with missing values in the key indices for my analysis
GDP2015_mask = Glob_df2.groupby("Code")["GDP (constant 2015 US$) [NY.GDP.MKTP.KD]"].agg(lambda x : x.isnull().sum())
GDP2015_mask[GDP2015_mask != 0]
#Selecting countries with less than 5 NaN values for the GDP (constant 2015 US$) column.
GDP2015_few_nan = GDP2015_mask[GDP2015_mask < 5]

#Filtering the dataframe to include only the selected countries
Glob_df2 = Glob_df2[Glob_df2.index.get_level_values("Code").isin(GDP2015_few_nan.index)]
Glob_df2.isna().sum()
Glob_df3 = Glob_df2.copy()
#Selecting columns with data type "object" and applying the pd.to_numeric function to convert to numeric format.
#This transformation assists in converting non-numeric data to numeric, replacing non-convertible values with NaN.

columns_to_convert = Glob_df3.columns[Glob_df3.dtypes == "object"]
Glob_df3[columns_to_convert] = Glob_df3[columns_to_convert].apply(pd.to_numeric)
Glob_df3.isna().sum() == Glob_df2.isna().sum()
#Function to interpolate NaN data for each country. This method is chosen for its ease of use and effectiveness with time series data(work better than mean or other methods)
#The parameter limit_direction = "both" ensures interpolation works on both sides of the values in the column.
def interpolate_country_nan(df, column_name, limit=None):

    for code in df.index.get_level_values("Code").unique():
        country_data = df.loc[code][column_name]
        
        country_data.interpolate(method="linear", limit=limit, limit_direction = "both", inplace =True)

interpolate_country_nan(Glob_df3,"Human Development Index", limit= None)
interpolate_country_nan(Glob_df3,"GDP (constant 2015 US$) [NY.GDP.MKTP.KD]", limit=None)
interpolate_country_nan(Glob_df3,"GDP (current US$) [NY.GDP.MKTP.CD]", limit=None)
Glob_df3.isna().sum()
Glob_df3.index
#Plotting all countries to display the values during the specified period for each country, providing an overview of the distribution.
def plot_all_countries(dataframe, column_name):
    data = dataframe[column_name]
    plt.figure(figsize=(15, 8))

    for code, country_data in data.groupby("Code"):
        plt.plot(country_data.index.get_level_values("Year"), country_data.values, label=code)

    plt.xlabel("Years")
    plt.ylabel(column_name)
    plt.title(f"{column_name} (1998-2022)")
    plt.grid()

    plt.show()
plot_all_countries(Glob_df3, "GDP (constant 2015 US$) [NY.GDP.MKTP.KD]")
plot_all_countries(Glob_df3, "Human Development Index")
#Creating a column for GDP per capita, adjusted for inflation.

Glob_df3["GDP_pc"] = Glob_df3["GDP (constant 2015 US$) [NY.GDP.MKTP.KD]"] / Glob_df3["Population, total [SP.POP.TOTL]"]
plot_all_countries(Glob_df3, "GDP_pc")
#Observing the 10 highest GDP per capita values in the world for the year 2021
Glob_df3.loc[Glob_df3.index.get_level_values("Year") == 2021, "GDP_pc"].nlargest(10)
Glob_df3.columns
#Modifying column titles for shorter names.
columns = ["Coal rents (% of GDP) [NY.GDP.COAL.RT.ZS]","Current health expenditure (% of GDP) [SH.XPD.CHEX.GD.ZS]",
           "Exports of goods and services (% of GDP) [NE.EXP.GNFS.ZS]","Forest rents (% of GDP) [NY.GDP.FRST.RT.ZS]",
           "Fuel exports (% of merchandise exports) [TX.VAL.FUEL.ZS.UN]","Fuel imports (% of merchandise imports) [TM.VAL.FUEL.ZS.UN]",
           "GDP (constant 2015 US$) [NY.GDP.MKTP.KD]","GDP (current US$) [NY.GDP.MKTP.CD]", "Gini index [SI.POV.GINI]",
           "Government expenditure on education, total (% of GDP) [SE.XPD.TOTL.GD.ZS]","Imports of goods and services (% of GDP) [NE.IMP.GNFS.ZS]",
           "Inflation, consumer prices (annual %) [FP.CPI.TOTL.ZG]","Manufactures exports (% of merchandise exports) [TX.VAL.MANF.ZS.UN]",
           "Manufactures imports (% of merchandise imports) [TM.VAL.MANF.ZS.UN]","Merchandise exports (current US$) [TX.VAL.MRCH.CD.WT]",
           "Merchandise imports (current US$) [TM.VAL.MRCH.CD.WT]","Merchandise trade (% of GDP) [TG.VAL.TOTL.GD.ZS]",
           "Military expenditure (% of GDP) [MS.MIL.XPND.GD.ZS]","Mineral rents (% of GDP) [NY.GDP.MINR.RT.ZS]",
           "Natural gas rents (% of GDP) [NY.GDP.NGAS.RT.ZS]","Oil rents (% of GDP) [NY.GDP.PETR.RT.ZS]",
           "Population, total [SP.POP.TOTL]","Research and development expenditure (% of GDP) [GB.XPD.RSDV.GD.ZS]",
           "Ores and metals exports (% of merchandise exports) [TX.VAL.MMTL.ZS.UN]","Ores and metals imports (% of merchandise imports) [TM.VAL.MMTL.ZS.UN]",
           "Total natural resources rents (% of GDP) [NY.GDP.TOTL.RT.ZS]","Human Development Index","GDP_pc"]

cleaned_columns = [col[:col.find("[")].strip() if "[" in col else col for col in columns]
Glob_df3.columns = cleaned_columns
Glob_df3.to_csv("Glob_df3.csv", index= True)
#Visualizing the correlation matrix of the columns.
import seaborn as sns

correlation_matrix = Glob_df3.corr()

plt.figure(figsize=(25, 25))
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Correlation matrix of the columns")
plt.show()
#Creating a new dataset to interpolate the other columns useful for the analysis
Glob_df4 = Glob_df3
Glob_df4.isna().groupby("Code").sum()
Glob_df4.isna().sum()
Glob_df4.groupby("Code").apply(lambda x: x.isnull().sum())
interpolate_country_nan(Glob_df4,"Total natural resources rents (% of GDP)", limit=None)
#Selecting the first 15 countries based on GDP (constant 2015 US$) values in 2021
sorted_21_gdp_df = Glob_df4.loc[Glob_df4.index.get_level_values("Year") == 2021].sort_values(by="GDP (constant 2015 US$)", ascending=False)
top_gdp9_countries = sorted_21_gdp_df.head(9) #for the line graph of the gdp performance
top_gdp_countries = sorted_21_gdp_df.head(15) #Selecting the top 15 countries usefulfor the dataset in the futures graph
print(top_gdp_countries.index.get_level_values("Code"))
print(sorted_21_gdp_df)
top_gdp_countries["GDP (constant 2015 US$)"]
top_gdp_countries.to_csv("top_gdp_countries.csv", index= True)
#The new dataset with all the values and columns for each country
GDP_cons_df = Glob_df4.loc[Glob_df4.index.get_level_values("Code").isin(top_gdp_countries.index.get_level_values("Code"))]
GDP_cons_df

#Plotting function to visualize data for selected countries over time.
#Dataframe: Containing the data.
#Countries: Containing country codes as index.
#Column_name: The name of the column to plot.
#ylabel: Label for the y-axis.
#Title: Title for the plot.
def plot_country_data(dataframe, countries, column_name, ylabel, title):
    plt.figure(figsize=(10, 6))

    for country_code in countries.index.get_level_values("Code"):
        country_data = dataframe.loc[country_code, column_name]
        years = country_data.index.get_level_values("Year")
        plt.plot(years, country_data, label=country_code)

    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title="Country Codes", loc="upper left")
    plt.grid()
    plt.show()
#Creating a line graph to visualize the GDP over time for the top 9 countries with the highest GDP (constant 2015 US$) in 2021
plot_country_data(GDP_cons_df, top_gdp9_countries, "GDP (constant 2015 US$)", 
                  "GDP (constant 2015 US$)", "The top 9 of highest GDP (constant 2015 US$)")
GDP_cons_df_21 = GDP_cons_df.loc[GDP_cons_df.index.get_level_values("Year") == 2021]
GDP_cons_df_21

from matplotlib.cm import ScalarMappable

#Show the GDP per capita and Human Development Index for the best 15 economies in the world in the 2021
plt.figure(figsize=(8, 6))

#Scatter plot with GDP, GDP per capita, and Human Development Index
scatter = plt.scatter(GDP_cons_df_21["GDP (constant 2015 US$)"], GDP_cons_df_21["GDP_pc"],
                      s=600, c=GDP_cons_df_21["Human Development Index"], cmap="RdYlGn", alpha=0.7, label="GDP")

#Labels with country names for each point
for i, code in enumerate(GDP_cons_df_21.index.get_level_values("Code")):
    plt.annotate(code,(GDP_cons_df_21["GDP (constant 2015 US$)"].iloc[i], GDP_cons_df_21["GDP_pc"].iloc[i]),
                 fontsize=10, ha="center",va="center",xytext=(0, 0), textcoords="offset points") 

plt.xlabel("GDP (constant 2015 US$)")
plt.ylabel("GDP per capita")
plt.title("The best 15 economies in the world (2021): HDI between GDP and GDP per capita")

#Color Bar
sm = ScalarMappable(cmap="RdYlGn")
sm.set_array(GDP_cons_df_21["Human Development Index"])

#Position of the bar
cbar = plt.colorbar(sm, label="Human Development Index", orientation="vertical", ax=plt.gca())
cbar.set_label("Human Development Index")

plt.show()
GDP_cons_df.index
GDP_cons_df.isna().sum()
#Selecting the first 9 countries with the highest Military expenditure (% of GDP) in 2021 among the top 15 countries in the world based on GDP in 2021.
top_mil_countries= GDP_cons_df.loc[GDP_cons_df.index.get_level_values("Year") == 2021].sort_values(by="Military expenditure (% of GDP)", ascending=False).head(9)
print(top_mil_countries.index.get_level_values("Code"))
plot_country_data(GDP_cons_df, top_mil_countries, "Military expenditure (% of GDP)", 
                  "Military expenditure (% of GDP)", "The top 9 for Military Expenditure")
#Counting null values ​​of the column Research and development expenditure (% of GDP) for each country
null_rs_by_country = GDP_cons_df.groupby("Code")["Research and development expenditure (% of GDP)"].apply(lambda x: x.isna().sum())
print(null_rs_by_country)
GDP_cons_df.loc["AUS"]["Research and development expenditure (% of GDP)"]
#Nan for AUS present a distribution pretty spread out over the period, make sense interpolate on its too 
interpolate_country_nan(GDP_cons_df,"Research and development expenditure (% of GDP)", limit=None)
#Selecting the first 9 countries with the highest Research and development expenditure (% of GDP) in 2021 among the top 15 countries in the world based on GDP in 2021.
top_rs_countries = GDP_cons_df.loc[GDP_cons_df.index.get_level_values("Year") == 2021].sort_values(by="Research and development expenditure (% of GDP)", ascending=False).head(9)
print(top_rs_countries.index.get_level_values("Code"))
plot_country_data(GDP_cons_df, top_rs_countries, "Research and development expenditure (% of GDP)", 
                  "Research and development expenditure (% of GDP)", "The top 9 for Research and development expenditure")
#Counting null values ​​of the column Research and development expenditure (% of GDP) for each country
GDP_cons_df.groupby("Code")["Total natural resources rents (% of GDP)"].apply(lambda x: x.isna().sum())
#Selecting the first 9 countries with the highest Total natural resources rents (% of GDP) in 2021 among the top 15 countries in the world based on GDP in 2021
top_nr_countries  = GDP_cons_df.loc[GDP_cons_df.index.get_level_values("Year") == 2021].sort_values(by="Total natural resources rents (% of GDP)", ascending=False).head(9)
print(top_nr_countries.index.get_level_values("Code"))
plot_country_data(GDP_cons_df, top_nr_countries, "Total natural resources rents (% of GDP)", 
                  "Total natural resources rents (% of GDP)", "The top 9 for Total natural resources rents")
null_he_by_country = GDP_cons_df.groupby("Code")["Current health expenditure (% of GDP)"].apply(lambda x: x.isna().sum())
print(null_he_by_country)
print(GDP_cons_df.loc["USA"]["Current health expenditure (% of GDP)"])
GDP_cons_df.loc["IND"]["Current health expenditure (% of GDP)"]

interpolate_country_nan(GDP_cons_df,"Current health expenditure (% of GDP)", limit=None)
top_he_countries = GDP_cons_df.loc[GDP_cons_df.index.get_level_values("Year") == 2021].sort_values(by="Current health expenditure (% of GDP)", ascending=False).head(9)
print(top_he_countries.index.get_level_values("Code"))
plot_country_data(GDP_cons_df, top_he_countries, "Current health expenditure (% of GDP)", 
                  "Current health expenditure (% of GDP)", "The top 9 for Current health expenditure")
Glob_21_df = sorted_21_gdp_df
Glob_98_df = Glob_df4.loc[Glob_df4.index.get_level_values("Year") == 1998].sort_values(by="GDP (constant 2015 US$)", ascending=False)
print(Glob_21_df)
Glob_98_df

#Creating a function to classify countries based on their own performance using GDP per capita and Human Development Index.
def development_classes(row):
    if row['Human Development Index'] >= 0.8 and row['GDP_pc'] >= 11000:
        return "Developed"
    elif row["GDP (constant 2015 US$)"] >= 1.5e+12 or ((row["Human Development Index"] >= 0.8 and row["GDP_pc"] < 11000) or \
        (0.5 <= row["Human Development Index"] < 0.8 and row["GDP_pc"] >= 11000)  or \
            (0.5 <= row["Human Development Index"] < 0.8 and 3255 <= row["GDP_pc"] <= 11000)  or \
                (row["Human Development Index"] <= 0.5 and row["GDP_pc"] > 11000) or\
                    (row["Human Development Index"] > 0.8 and row["GDP_pc"] < 3255)):
        return "Developing"
    else:
        return "Underdeveloped"

Glob_21_df["Development_Class"] = Glob_21_df.apply(development_classes, axis=1)
Glob_98_df["Development_Class"] = Glob_98_df.apply(development_classes, axis=1)


Glob_21_df["Development_Class"].head(25)
print(Glob_21_df.isna().sum())
Glob_98_df.isna().sum()
#Noting: 42 countries were deleted during the cleaning process due to a lack of data in the HDI and GDP columns. 
#It's likely that many of them could be classified in the Underdeveloped category.
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].bar(Glob_98_df["Development_Class"].value_counts().reindex(["Developed", "Developing", "Underdeveloped"]).index, 
           Glob_98_df["Development_Class"].value_counts().reindex(["Developed", "Developing", "Underdeveloped"]).values, color="lightgreen")
axs[0].set_title("Number of countries per development class (1998)")
axs[0].set_ylabel("Number of countries")

axs[1].bar(Glob_21_df["Development_Class"].value_counts().reindex(["Developed", "Developing", "Underdeveloped"]).index, 
           Glob_21_df["Development_Class"].value_counts().reindex(["Developed", "Developing", "Underdeveloped"]).values, color="skyblue")
axs[1].set_title("Number of countries per development class (2021)")

plt.show()

print(Glob_21_df.groupby("Development_Class")["Population, total"].sum())
Glob_98_df.groupby("Development_Class")["Population, total"].sum()
population_sum_1998 = {"Developed": 884951495,"Developing": 2249849761,"Underdeveloped": 2619712696}
population_sum_2021 = {"Developed": 1364608769, "Developing": 4591342874,"Underdeveloped": 1578135042}

fig, axs = plt.subplots(1, 2, figsize=(9, 7))

axs[0].pie(population_sum_1998.values(), labels=population_sum_1998.keys(), autopct='%1.1f%%', colors=["skyblue", "lightgreen", "lightcoral"])
axs[0].set_title("Population Distribution by Development Class (1998)")

axs[1].pie(population_sum_2021.values(), labels=population_sum_2021.keys(), autopct='%1.1f%%', colors=["skyblue", "lightgreen", "lightcoral"])
axs[1].set_title("Population Distribution by Development Class (2021)")

plt.tight_layout()
plt.show()
#Displaying the average percentage of GDP from natural resources rents for countries in 1998 and 2021.
print(f"1998:\n{Glob_98_df.groupby("Development_Class")["Total natural resources rents (% of GDP)"].mean()}\n2021:\n{Glob_21_df.groupby("Development_Class")["Total natural resources rents (% of GDP)"].mean()}")

development_classes = ["Developed", "Developing", "Underdeveloped"]
nrr_1998 = [1.016800, 4.350060, 7.954888]
nrr_2021 = [4.896486, 7.405499, 9.680971]

plt.figure(figsize=(10, 6))
bar_width = 0.4
index = (range(0,3))

bars1 = plt.bar(index, nrr_1998, bar_width, color = "lightcoral",label="1998")
bars2 = plt.bar([i + bar_width for i in index], nrr_2021, bar_width,color = "skyblue", label="2021")

plt.ylabel("Average GDP from Natural Resources Rents (%)")
plt.title("Average GDP from Natural Resources Rents by Development Class (1998 vs 2021)")
plt.xticks([i + bar_width/2 for i in index], development_classes)
plt.legend()

plt.tight_layout()
plt.show()

#Beginning the preparation of the dataset for applying the Linear Regression Model to predict GDP values.
#NaN values in columns intended for use as independent variables in the model will be filled.

Glob_df4.isna().sum()
Glob_df4.columns
m_rent_mask = Glob_df4.groupby("Code")["Mineral rents (% of GDP)"].apply(lambda x: x.isna().sum())
m_rent_mask[m_rent_mask != 0]
interpolate_country_nan(Glob_df4,"Mineral rents (% of GDP)", limit=None)
ng_rent_mask = Glob_df4.groupby("Code")["Natural gas rents (% of GDP)"].apply(lambda x: x.isna().sum())
ng_rent_mask[m_rent_mask != 0]
Glob_df4.drop("PLW", level=0, inplace=True)
interpolate_country_nan(Glob_df4,"Natural gas rents (% of GDP)", limit=None)
Glob_df4.isna().sum()
print(Glob_df4[Glob_df4["Natural gas rents (% of GDP)"].isna()].index)

Glob_df4.drop("SLE", level=0, inplace=True)
mi_rent_mask = Glob_df4.groupby("Code")["Merchandise imports (current US$)"].apply(lambda x: x.isna().sum())
mi_rent_mask[mi_rent_mask != 0]

Glob_df4 = Glob_df4.drop("AND",level=0)
Glob_df4 = Glob_df4.drop("SRB",level=0)
interpolate_country_nan(Glob_df4,"Merchandise imports (current US$)", limit=None)
fo_rent_mask = Glob_df4.groupby("Code")["Forest rents (% of GDP)"].apply(lambda x: x.isna().sum())
fo_rent_mask[fo_rent_mask != 0]
interpolate_country_nan(Glob_df4,"Forest rents (% of GDP)", limit=None)
Glob_df4.isna().sum()
#Selected these columns like indipendent variables
"""GDP (current US$),Merchandise imports (current US$)
Mineral rents (% of GDP)
Natural gas rents (% of GDP)
Population, total
Total natural resources rents (% of GDP)
Human Development Index, Forest rents (% of GDP)"""
#Resetting the index to training and testing the model
Glob_df4_reset = Glob_df4.reset_index()
Glob_df4_reset.to_csv("Glob_df4_reset.csv", index= True)
Glob_df4_reset.columns[Glob_df4_reset.isna().sum() == 0]
from sklearn.linear_model import LinearRegression
features = ["GDP (current US$)","Merchandise imports (current US$)","Mineral rents (% of GDP)",
            "Natural gas rents (% of GDP)","Population, total","Total natural resources rents (% of GDP)",
            "Human Development Index","Forest rents (% of GDP)"]

dipendent_variable = "GDP (constant 2015 US$)"

X = Glob_df4_reset[features]
y = Glob_df4_reset[dipendent_variable]
from sklearn.model_selection import train_test_split

#Subdivision of the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

print("Training set size:", X_train.shape, y_train.shape)
print("Test set dimensions:", X_test.shape, y_test.shape)

#Initialize the linear regression model
model = LinearRegression()

#Training the model on the training set
model.fit(X_train, y_train)
#Making predictions about the test set
y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score

#Some metrics to evaluate the model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2:", r2_score(y_test, y_pred))
print("Intercept: ",model.intercept_)
print("Coefficient for each feature: ", model.coef_)
#Doing some prediction to observing the predicted values by the model and the real data from the GDP column
X_ita = Glob_df4_reset[Glob_df4_reset["Code"] == "ITA"][features]
model.predict(X_ita)
Glob_df4.loc["ITA"]["GDP (current US$)"]
X_usa = Glob_df4_reset[Glob_df4_reset["Code"] == "USA"][features]
model.predict(X_usa)
Glob_df4.loc["USA"]["GDP (current US$)"]
X_chn = Glob_df4_reset[Glob_df4_reset["Code"] == "CHN"][features]
model.predict(X_chn)
Glob_df4.loc["CHN"]["GDP (current US$)"]