import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
Dev_df = pd.read_csv("C:/Users/loris/Desktop/Project ER/Develop Ind.csv")
HDI_df = pd.read_csv("C:/Users/loris/Desktop/Project ER/human-development-index.csv")

HDI_df
HDI_df.shape
#i'm watching the stucture of the dataset
HDI_df.head()
print(HDI_df.tail())
HDI_df[5890:5923]
HDI_df.info()
#i'm watching the stucture of the dataset
Dev_df.head()
Dev_df.info()
print(Dev_df.shape)
Dev_df.tail()
Glob_df = Dev_df.copy()
Glob_df.rename(columns={"Country Code": "Code"}, inplace=True)
Glob_df.rename(columns={"Time": "Year"}, inplace=True)
#I tried change type of Year column before the merge, but i can't for Nan
Glob_df.isnull().sum()
Glob_df[6645:6655]
Glob_df[5424:6656]
#watching Nan Data i noticed that there are also some aggregate of geographics areas
#so i'm deleting all the rows under the last country in the dataset
Glob_df.drop(Glob_df.index[5425:6656], inplace=True)
print(Glob_df.shape)
print(HDI_df.shape)
#now i change the type of Year in Glob_df to have the same type of HDI_df
Glob_df["Year"] = Glob_df["Year"].astype(int)

print(f"{Glob_df.columns}\n{HDI_df.columns}")
Glob_df = Glob_df.set_index(["Code", "Year"])
HDI_df = HDI_df.set_index(["Code", "Year"])
print(HDI_df)
print(Glob_df)
#I controll the null values before the merge
print(f"{Glob_df.isnull().sum()}\n{HDI_df.isnull().sum()}")
print(HDI_df[5855:5940])
#I do the merge, how="left" to use only keys from left frame and using index of both datasets for the merge
Glob_df = Glob_df.merge(HDI_df["Human Development Index"], left_index=True, right_index=True, how="left")

#I controll the Nan created with the merge
Glob_df["Human Development Index"].isnull().value_counts()

Glob_df.info()
HDI_nan_countries = Glob_df.groupby("Code")["Human Development Index"].agg(lambda x : x.isnull().sum())
HDI_nan_countries
#i'm watching how many countries have no data for the HDI
HDI_nan_countries[HDI_nan_countries == 25].index

#This country code there were not in HDI_df. I controlled with ex. HDI_df.loc["GUM"] HDI_df.loc["SXM"]
#The others country with Nan Values is for dismatch beetween years in the two merged dataset
#(every country have at least one because the year 2022 there was not in the HDI_df ), 
#so i decided to delete the year 2022 and mantain only country with <5
HDI_df.loc["AGO"]
HDI_df.loc["GNB"]
HDI_df.loc["AND"]
#i decided to delete countries with more than 4 Nan so i control which they are
HDI_too_nan = HDI_nan_countries[(HDI_nan_countries > 5)]

#i control the countris with too many nan values in the HDI column
HDI_too_nan.index
#i create the mask of the countries that i want mantain
HDI_few_nan = HDI_nan_countries[HDI_nan_countries <= 4]
HDI_few_nan
#i take these countries applying the boolean mask
Glob_df2 = Glob_df[Glob_df.index.get_level_values("Code").isin(HDI_few_nan.index)]

Glob_df2.isna().sum()
#Replace dots with NaN throughout the dataframe, too estimate the real number of Nan in the others columns
Glob_df2.replace("..", np.nan, inplace=True)

Glob_df2.isna().sum()
#the year 2022 present too many Nan in the columns useful for my analises and moreover the HDI don't have values for any country
Glob_df2.isna().groupby("Year").sum()
Glob_df2.drop(index=2022, level="Year", inplace = True) 
print(Glob_df2.columns)
#i drop columns that i'm sure to don't use
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
"Agricultural raw materials exports (% of merchandise exports) [TX.VAL.AGRI.ZS.UN]", "Surface area (sq. km) [AG.SRF.TOTL.K2]",
"GDP (current US$) [NY.GDP.MKTP.CD]"]

Glob_df2 = Glob_df2.drop(columns=columns_to_drop)
Glob_df2.info()
print(Glob_df2.columns)
#I'm getting an idea of ​​the countries with missing values ​​in the key indices for my analysis
GDP2015_mask = Glob_df2.groupby("Code")["GDP (constant 2015 US$) [NY.GDP.MKTP.KD]"].agg(lambda x : x.isnull().sum())
GDP2015_mask[GDP2015_mask != 0]
#I choose less than 5 Nan for country
GDP2015_few_nan = GDP2015_mask[GDP2015_mask < 5]

Glob_df2 = Glob_df2[Glob_df2.index.get_level_values("Code").isin(GDP2015_few_nan.index)]
Glob_df2.isna().sum()
Glob_df3 = Glob_df2.copy()
#The code selects columns with data type "object" and then applies the pd.to_numeric function to convert to numeric format.
#This helps in transforming non-numeric data to numeric, replacing non-convertible values with NaN.

columns_to_convert = Glob_df3.columns[Glob_df3.dtypes == "object"]
Glob_df3[columns_to_convert] = Glob_df3[columns_to_convert].apply(pd.to_numeric)
Glob_df3.isna().sum() == Glob_df2.isna().sum()
#A function to interpolate the nan data for each country, i want use this comand because it's easy to use and with time series 
#work better than mean or other way.
#Filling Nan using a linear method, with a determinated limit , it's like a crossed control, 
#because before use this function, maybe i could left some country with some nan.
#Moreover limit_direction = "both" so the interpolation work in both the side of the values in the coloumn
def interpolate_country_nan(df, column_name, limit=None):

    for code in df.index.get_level_values("Code").unique():
        country_data = df.loc[code][column_name]
        
        country_data.interpolate(method="linear", limit=limit, limit_direction = "both", inplace =True)

interpolate_country_nan(Glob_df3,"Human Development Index", limit= None)
interpolate_country_nan(Glob_df3,"GDP (constant 2015 US$) [NY.GDP.MKTP.KD]", limit=None)
Glob_df3.isna().sum()
Glob_df3.index
#I'm plotting all countries, for each code (coutry) i show the values during the period
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
#I want create a column GDP pro capita, but taking inflation into account

Glob_df3["GDP_pc"] = Glob_df3["GDP (constant 2015 US$) [NY.GDP.MKTP.KD]"] / Glob_df3["Population, total [SP.POP.TOTL]"]
plot_all_countries(Glob_df3, "GDP_pc")
#watching the 10 highest gdp per capita in the world (year 2021)
Glob_df3.loc[Glob_df3.index.get_level_values("Year") == 2021, "GDP_pc"].nlargest(10)
#I'm visualizing the correlation matrix of the coloumns
import seaborn as sns

correlation_matrix = Glob_df3.corr()

plt.figure(figsize=(25, 25))
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Correlation matrix of the columns")
plt.show()
#The new dataset for interpolate the others columns useful for the analises
Glob_df4 = Glob_df3
Glob_df4.isna().groupby("Code").sum()
Glob_df4.isna().sum()
Glob_df4.groupby("Code").apply(lambda x: x.isnull().sum())
interpolate_country_nan(Glob_df4,"Total natural resources rents (% of GDP) [NY.GDP.TOTL.RT.ZS]", limit=None)
#I want select the first 13 country of GDP (constant 2015 US$), i'm selecting them usng the values at 2021.
sorted_21_gdp_df = Glob_df4.loc[Glob_df4.index.get_level_values("Year") == 2021].sort_values(by="GDP (constant 2015 US$) [NY.GDP.MKTP.KD]", ascending=False)
top_gdp9_countries = sorted_21_gdp_df.head(9) #i will use this for the line graph of the gdp performance
top_gdp_countries = sorted_21_gdp_df.head(13) #i will use it for the dataset useful for the other graphs
print(top_gdp_countries.index.get_level_values("Code"))
print(sorted_21_gdp_df)

#The new dataset with all the values and coloumns for each country
GDP_cons_df = Glob_df4.loc[Glob_df4.index.get_level_values("Code").isin(top_gdp_countries.index.get_level_values("Code"))]
GDP_cons_df

def plot_country_data(dataframe, countries, column_name, ylabel, title):
    plt.figure(figsize=(10, 6))

    for country_code in countries.index.get_level_values("Code").unique():
        country_data = dataframe.loc[country_code, column_name]
        years = country_data.index.get_level_values("Year")
        plt.plot(years, country_data, label=country_code)

    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title='Country Codes', loc='upper left')
    plt.grid()
    plt.show()
#Create line graph for GDP over time for the top 9 countries with highest GDP (constant 2015 US$) in 2021 among the biggest
#13 country considering the gdp in 2021
plot_country_data(GDP_cons_df, top_gdp9_countries, "GDP (constant 2015 US$) [NY.GDP.MKTP.KD]", 
                  "GDP (constant 2015 US$)", "The top 9 of highest GDP (constant 2015 US$)")
GDP_cons_df_21 = GDP_cons_df.loc[GDP_cons_df.index.get_level_values("Year") == 2021]
GDP_cons_df_21

from matplotlib.cm import ScalarMappable

#Show the GDP per capita and Human Development Index for the best 10 economies in the world in the 2021
plt.figure(figsize=(8, 6))

scatter = plt.scatter(GDP_cons_df_21["GDP (constant 2015 US$) [NY.GDP.MKTP.KD]"], GDP_cons_df_21["GDP_pc"],
                      s=600, c=GDP_cons_df_21["Human Development Index"],
                      cmap="RdYlGn", alpha=0.7, label="GDP")

#labels with country names for each point
for i, code in enumerate(GDP_cons_df_21.index.get_level_values("Code")):
    plt.annotate(code,(GDP_cons_df_21["GDP (constant 2015 US$) [NY.GDP.MKTP.KD]"].iloc[i], GDP_cons_df_21["GDP_pc"].iloc[i]),
                 fontsize=10, ha="center",va="center",xytext=(0, 0), textcoords="offset points") 

plt.xlabel("GDP (constant 2015 US$)")
plt.ylabel("GDP per capita")
plt.title("The best 10 economies in the world (2022): HDI between GDP and GDP per capita")

#Color Bar
sm = ScalarMappable(cmap="RdYlGn")
sm.set_array(GDP_cons_df_21["Human Development Index"])

#Position of the bar
cbar = plt.colorbar(sm, label="Human Development Index", orientation="vertical", ax=plt.gca())
cbar.set_label("Human Development Index")

plt.show()
GDP_cons_df.index
GDP_cons_df.isna().sum()
#I want select the first 9 country of Military expenditure (% of GDP) in 2021 among the biggest 13 countries in the world for gdp in 2021"
top_mil_countries= GDP_cons_df.loc[GDP_cons_df.index.get_level_values("Year") == 2021].sort_values(by="Military expenditure (% of GDP) [MS.MIL.XPND.GD.ZS]", ascending=False).head(9)
print(top_mil_countries.index.get_level_values("Code"))
plot_country_data(GDP_cons_df, top_mil_countries, "Military expenditure (% of GDP) [MS.MIL.XPND.GD.ZS]", 
                  "Military expenditure (% of GDP)", "Military Expenditure of the 9 Countries with highest GDP in 2022")
# Count of null values ​​of the column Research and development expenditure (% of GDP) for each country
null_rs_by_country = GDP_cons_df.groupby("Code")["Research and development expenditure (% of GDP) [GB.XPD.RSDV.GD.ZS]"].apply(lambda x: x.isna().sum())
print(null_rs_by_country)
GDP_cons_df.loc["AUS"]["Research and development expenditure (% of GDP) [GB.XPD.RSDV.GD.ZS]"]
#Nan for AUS present a distribution pretty spread out over the period, so I'll use interpolate on it too 
interpolate_country_nan(GDP_cons_df,"Research and development expenditure (% of GDP) [GB.XPD.RSDV.GD.ZS]", limit=None)
#I want select the first 9 country of "Research and development expenditure (% of GDP)" in 2021 among the biggest 13 countries in the world for gdp in 2021"
top_rs_countries = GDP_cons_df.loc[GDP_cons_df.index.get_level_values("Year") == 2021].sort_values(by="Research and development expenditure (% of GDP) [GB.XPD.RSDV.GD.ZS]", ascending=False).head(9)
print(top_rs_countries.index.get_level_values("Code"))
# Create line graph for military spending over time for the top 9 countries with highest military expenditure (% of GDP) in 2021 among the biggest
#country considering the gdp in 2021
plot_country_data(GDP_cons_df, top_rs_countries, "Research and development expenditure (% of GDP) [GB.XPD.RSDV.GD.ZS]", 
                  "Research and development expenditure (% of GDP)", "Research and development expenditure of the 9 Countries with highest GDP in 2022")
# Count of null values ​​of the column Research and development expenditure (% of GDP) for each country
GDP_cons_df.groupby("Code")["Total natural resources rents (% of GDP) [NY.GDP.TOTL.RT.ZS]"].apply(lambda x: x.isna().sum())
#I want select the first 9 country of "Total natural resources rents (% of GDP) [NY.GDP.TOTL.RT.ZS]" in 2021 among the biggest 9 countries in the world for gdp in 2021"
top_nr_countries  = GDP_cons_df.loc[GDP_cons_df.index.get_level_values("Year") == 2021].sort_values(by="Total natural resources rents (% of GDP) [NY.GDP.TOTL.RT.ZS]", ascending=False).head(9)
print(top_nr_countries.index.get_level_values("Code"))
# Create line graph for military spending over time for the top 9 countries with highest military expenditure (% of GDP) in 2021 among the biggest
#country considering the gdp in 2021
plot_country_data(GDP_cons_df, top_nr_countries, "Total natural resources rents (% of GDP) [NY.GDP.TOTL.RT.ZS]", 
                  "Total natural resources rents (% of GDP)", "Total natural resources rents of the 9 Countries with highest GDP in 2022")
null_he_by_country = GDP_cons_df.groupby("Code")["Current health expenditure (% of GDP) [SH.XPD.CHEX.GD.ZS]"].apply(lambda x: x.isna().sum())
print(null_he_by_country)
print(GDP_cons_df.loc["USA"]["Current health expenditure (% of GDP) [SH.XPD.CHEX.GD.ZS]"])
GDP_cons_df.loc["IND"]["Current health expenditure (% of GDP) [SH.XPD.CHEX.GD.ZS]"]

interpolate_country_nan(GDP_cons_df,"Current health expenditure (% of GDP) [SH.XPD.CHEX.GD.ZS]", limit=None)
top_he_countries = GDP_cons_df.loc[GDP_cons_df.index.get_level_values("Year") == 2021].sort_values(by="Current health expenditure (% of GDP) [SH.XPD.CHEX.GD.ZS]", ascending=False).head(9)
print(top_he_countries.index.get_level_values("Code"))
plot_country_data(GDP_cons_df, top_he_countries, "Current health expenditure (% of GDP) [SH.XPD.CHEX.GD.ZS]", 
                  "Current health expenditure (% of GDP)", "Current health expenditure of the 9 Countries with highest GDP in 2022")
Glob_21_df = sorted_21_gdp_df
Glob_98_df = Glob_df4.loc[Glob_df4.index.get_level_values("Year") == 1998].sort_values(by="GDP (constant 2015 US$) [NY.GDP.MKTP.KD]", ascending=False)
print(Glob_21_df)
Glob_98_df

def development_classes(row):
    if row['Human Development Index'] >= 0.8 and row['GDP_pc'] >= 11000:
        return "Developed"
    elif row['GDP (constant 2015 US$) [NY.GDP.MKTP.KD]'] >= 1.5e+12 or ((row['Human Development Index'] >= 0.8 and row['GDP_pc'] < 11000) or \
        (0.5 <= row['Human Development Index'] < 0.8 and row['GDP_pc'] >= 11000)  or \
            (0.5 <= row['Human Development Index'] < 0.8 and 3255 <= row['GDP_pc'] <= 11000)  or \
                (row['Human Development Index'] <= 0.5 and row['GDP_pc'] > 11000) or\
                    (row['Human Development Index'] > 0.8 and row['GDP_pc'] < 3255)):
        return "Developing"
    else:
        return "Underdeveloped"

Glob_21_df["Development_Class"] = Glob_21_df.apply(development_classes, axis=1)
Glob_98_df["Development_Class"] = Glob_98_df.apply(development_classes, axis=1)


Glob_21_df["Development_Class"].head(25)
print(Glob_21_df.isna().sum())
Glob_98_df.isna().sum()
counts_2021 = Glob_21_df["Development_Class"].value_counts()
counts_1998 = Glob_98_df["Development_Class"].value_counts()
#Remeber: deleted 42 countries during the cleaning for lack of too many data in the coloumns HDI and GDP. 
#Probably a lot of them could be in the Underdeveloped and Developing classes
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].bar(counts_1998.reindex(["Developed", "Developing", "Underdeveloped"]).index, counts_1998.reindex(["Developed", "Developing", "Underdeveloped"]).values, color="lightgreen")
axs[0].set_title("Number of countries per development class (1998)")
axs[0].set_ylabel("Number of countries")

axs[1].bar(counts_2021.reindex(["Developed", "Developing", "Underdeveloped"]).index, counts_2021.reindex(["Developed", "Developing", "Underdeveloped"]).values, color="skyblue")
axs[1].set_title("Number of countries per development class (2021)")

plt.show()

print(Glob_21_df.groupby("Development_Class")["Population, total [SP.POP.TOTL]"].sum())
Glob_98_df.groupby("Development_Class")["Population, total [SP.POP.TOTL]"].sum()
population_sum_1998 = {"Developed": 884951495,"Developing": 2249849761,"Underdeveloped": 2619712696}
population_sum_2021 = {"Developed": 1364608769, "Developing": 4591342874,"Underdeveloped": 1578135042}

fig, axs = plt.subplots(1, 2, figsize=(9, 7))

axs[0].pie(population_sum_1998.values(), labels=population_sum_1998.keys(), autopct='%1.1f%%', colors=["skyblue", "lightgreen", "lightcoral"])
axs[0].set_title('Population Distribution by Development Class (1998)')

axs[1].pie(population_sum_2021.values(), labels=population_sum_2021.keys(), autopct='%1.1f%%', colors=["skyblue", "lightgreen", "lightcoral"])
axs[1].set_title('Population Distribution by Development Class (2021)')

plt.tight_layout()
plt.show()
#show that:
#1)Developed countries have less % of GDP from natural resources rents
#2)during the period, exporter countries of natural resources increase their condition
print(f"1998:\n{Glob_98_df.groupby("Development_Class")["Total natural resources rents (% of GDP) [NY.GDP.TOTL.RT.ZS]"].mean()}\n2021:\n{Glob_21_df.groupby("Development_Class")["Total natural resources rents (% of GDP) [NY.GDP.TOTL.RT.ZS]"].mean()}")


