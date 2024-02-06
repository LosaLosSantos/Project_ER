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
#now i change the type of Year in Glob_df so for have the same type of HDI_df
Glob_df["Year"] = Glob_df["Year"].astype(int)

print(f"{Glob_df.columns}\n{HDI_df.columns}")
Glob_df = Glob_df.set_index(["Code", "Year"])
HDI_df = HDI_df.set_index(["Code", "Year"])
print(HDI_df)
print(Glob_df)
#I controll the null values before the merge
print(f"{Glob_df.isnull().sum()}\n{HDI_df.isnull().sum()}")
print(HDI_df[5855:5940])
#I do the merge and lose the HDI_df's rows without correspondence with Glob_df
Glob_df = Glob_df.merge(HDI_df["Human Development Index"], left_index=True, right_index=True, how='left')

#I controll the Nan created with the merge
Glob_df["Human Development Index"].isnull().value_counts()
HDI_nan_countries = Glob_df.groupby("Code")["Human Development Index"].agg(lambda x : x.isnull().sum())
HDI_nan_countries
#i'm watching how many countries have no data for the HDI
HDI_nan_countries[HDI_nan_countries == 25].index
#This country code there were not in HDI_df.I controlled with ex. HDI_df.loc["SXM"], HDI_df.loc["GUM"]
#The others country with Nan Values is for dismatch beetween years in the two merged dataset
#(every country have at least one because the year 2022 there was not in the HDI_df ), 
#so i decided to mantain only country with <4 Nan that i can manage sobstituing Nan while a prediction of the values
HDI_df.loc["AGO"]
HDI_df.loc["GNB"]
HDI_df.loc["AND"]
#i decided to delete countries with more than 3 Nan so i control which they are
HDI_too_nan = HDI_nan_countries[(HDI_nan_countries > 3)]

#i control the countris with too many nan values in the HDI column
HDI_too_nan.index
#i create the mask of the countries that i want mantain
HDI_few_nan = HDI_nan_countries[HDI_nan_countries <= 3]
HDI_few_nan
#i take these countries applying the boolean mask
Glob_df2 = Glob_df[Glob_df.index.get_level_values("Code").isin(HDI_few_nan.index)]

Glob_df2.isna().sum()
#A function to interpolate the nan data for each country
#Filling Nan using a linear method, with a determinated limit , it's like a crossed control, 
#because before use this function, maybe i left some country with some amount of nan.
#Moreover limit_direction = "both" so the interpolation work in both the side of the interpolation
def interpolate_country_nan(df, column_name, limit=None):

    for code in df.index.get_level_values("Code").unique():
        country_data = df.loc[code][column_name]
        
        country_data.interpolate(method="linear", limit=limit, limit_direction = "both", inplace =True)
interpolate_country_nan(Glob_df2,"Human Development Index", limit=3)
Glob_df2.isna().sum()
#Replace dots with NaN throughout the dataframe, too estimate the real number of Nan in the others columns
Glob_df2.replace("..", np.nan, inplace=True)

Glob_df2.isna().sum()
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
#I putted this command after the command for change data from type 'object' to numbers
interpolate_country_nan(Glob_df3,"GDP (constant 2015 US$) [NY.GDP.MKTP.KD]", limit=4)
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
Glob_df3["Population, total [SP.POP.TOTL]"].isna().sum()
#i'm removing countries with population too small
average_pop = Glob_df3.groupby("Code")["Population, total [SP.POP.TOTL]"].mean()

enough_pop_mask = average_pop[average_pop > 300000]

average_pop[average_pop < 300000]

#I selected the countries with more than 300000 of population
Glob_df3 = Glob_df3[Glob_df3.index.get_level_values("Code").isin(enough_pop_mask.index)]
#I want create a column GDP pro capita, but taking inflation into account

Glob_df3["GDP_pc"] = Glob_df3["GDP (constant 2015 US$) [NY.GDP.MKTP.KD]"] / Glob_df3["Population, total [SP.POP.TOTL]"]
plot_all_countries(Glob_df3, "GDP_pc")
#watching the 10 highest gdp pc in the world (year 2022)
gdp_pc_2022 = Glob_df3.loc[Glob_df3.index.get_level_values("Year") == 2022, "GDP_pc"]
top_gdp_pc_2022 = gdp_pc_2022.nlargest(10)

print(top_gdp_pc_2022)
#I'm visualizing the correlation matrix of the coloumns
import seaborn as sns

correlation_matrix = Glob_df3.corr()

plt.figure(figsize=(25, 25))
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=True, fmt=".2f", linewidths=.5)
plt.title("Correlation matrix of the columns")
plt.show()

#I want select the first 10 country of GDP (constant 2015 US$)"
sorted_gdp_df = Glob_df3.loc[Glob_df3.index.get_level_values("Year") == 2022].sort_values(by="GDP (constant 2015 US$) [NY.GDP.MKTP.KD]", ascending=False)

# Selecting the countries
top_gdp_countries = sorted_gdp_df.head(10)

print(top_gdp_countries.index.get_level_values("Code"))

#The new dataset with all the values and coloumns for each country
GDP_cons_df = Glob_df3.loc[Glob_df3.index.get_level_values("Code").isin(top_gdp_countries.index.get_level_values("Code"))]
GDP_cons_df

GDP_cons_df_2022 = GDP_cons_df.loc[GDP_cons_df.index.get_level_values("Year") == 2022]

from matplotlib.cm import ScalarMappable

#Show the GDP per capita and Human Development Index for the best 10 economies in the world in the 2022
plt.figure(figsize=(8, 6))

scatter = plt.scatter(GDP_cons_df_2022["GDP (constant 2015 US$) [NY.GDP.MKTP.KD]"], GDP_cons_df_2022["GDP_pc"],
                      s=500, c=GDP_cons_df_2022["Human Development Index"],
                      cmap='RdYlGn', alpha=0.7, label='GDP')

#labels with country names for each point
for i, code in enumerate(GDP_cons_df_2022.index.get_level_values("Code")):
    plt.annotate(code,(GDP_cons_df_2022["GDP (constant 2015 US$) [NY.GDP.MKTP.KD]"].iloc[i], GDP_cons_df_2022["GDP_pc"].iloc[i]),
                 fontsize=10, ha='center',va='center',xytext=(0, 0), textcoords='offset points') 

plt.xlabel("GDP (constant 2015 US$)")
plt.ylabel("GDP per capita")
plt.title("The best 10 economies in the world (2022): HDI between GDP and GDP per capita")
#Color Bar
sm = ScalarMappable(cmap='RdYlGn')
sm.set_array(GDP_cons_df_2022["Human Development Index"])
#Position of the bar
cbar = plt.colorbar(sm, label='Human Development Index', orientation='vertical', ax=plt.gca())
cbar.set_label('Human Development Index')

plt.show()