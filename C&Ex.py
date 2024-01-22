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

#i do the merge and lose the HDI_df's rows without correspondence with Glob_df
Glob_df = Glob_df.merge(HDI_df["Human Development Index"], left_index=True, right_index=True, how='left')

#i controll the Nan created with the merge
Glob_df["Human Development Index"].isnull().value_counts()
HDI_Nan_countries = Glob_df.groupby("Code")["Human Development Index"].agg(lambda x : x.isnull().sum())
HDI_Nan_countries

#i'm watching how many countries have no data for the HDI
HDI_Nan_countries[HDI_Nan_countries == 25].index
#This country code there were not in HDI_df.I controlled with ex. HDI_df.loc["SXM"], HDI_df.loc["GUM"]
#The others country with Nan Values is for dismatch beetween years in the two merged dataset
#(every country have at least one because the year 2022 there was not in the HDI_df ), 
#so i dcided to mantain only country with <4 Nan that i can manage sobstituing Nan while a prediction of the values
HDI_df.loc["AGO"]
HDI_df.loc["GNB"]
HDI_df.loc["AND"] 

#i decided to delete countries with more than 3 Nan so i control which they are
HDI_Null_too_null = HDI_Nan_countries[(HDI_Nan_countries > 3)]

#i control the countris with too many nan values in the HDI column
HDI_Null_too_null.index

#i create the mask of the countries that i want mantain
HDI_few_Null = HDI_Nan_countries[HDI_Nan_countries <= 3]
HDI_few_Null

#i take these countries applying the boolean mask
Glob_df2 = Glob_df[Glob_df.index.get_level_values("Code").isin(HDI_few_Null.index)]
Glob_df2.isna().sum()

#Replace dots with NaN throughout the dataframe, too estimate the real number of Nan
Glob_df2.replace('..', np.nan, inplace=True)
Glob_df2.isna().sum()

#I'm getting an idea of ​​the countries with missing values ​​in the key indices for my analysis
Gmask = Glob_df2.groupby('Code')["GDP (constant 2015 US$) [NY.GDP.MKTP.KD]"].agg(lambda x : x.isnull().sum())
Gmask[Gmask != 0]
Gmask2 = Glob_df2.groupby("Code")["GDP (current US$) [NY.GDP.MKTP.CD]"].agg(lambda x : x.isnull().sum())
Gmask2[Gmask2 != 0]
Glob_df2.columns

#i drop columns that i'm sure to don't use
columns_to_drop = ["Country Name", "Time Code",
"Arms exports (SIPRI trend indicator values) [MS.MIL.XPRT.KD]",
"Arms imports (SIPRI trend indicator values) [MS.MIL.MPRT.KD]",
"Consumer price index (2010 = 100) [FP.CPI.TOTL]",
"Cost of business start-up procedures, female (% of GNI per capita) [IC.REG.COST.PC.FE.ZS]",
"Coverage of social insurance programs (% of population) [per_si_allsi.cov_pop_tot]",
"Current education expenditure, total (% of total expenditure in public institutions) [SE.XPD.CTOT.ZS]",
"Domestic credit to private sector (% of GDP) [FS.AST.PRVT.GD.ZS]",
"Gross savings (% of GDP) [NY.GNS.ICTR.ZS]",
"Human capital index (HCI) (scale 0-1) [HD.HCI.OVRL]",
"Manufacturing, value added (% of GDP) [NV.IND.MANF.ZS]",
"Services, value added (% of GDP) [NV.SRV.TOTL.ZS]",
"Literacy rate, adult total (% of people ages 15 and above) [SE.ADT.LITR.ZS]",
"Central government debt, total (% of GDP) [GC.DOD.TOTL.GD.ZS]",
"Interest payments (% of expense) [GC.XPN.INTP.ZS]",
"Researchers in R&D (per million people) [SP.POP.SCIE.RD.P6]"]

Glob_df2 = Glob_df2.drop(columns=columns_to_drop)
Glob_df2.info()
#The code selects columns with data type 'object' and then applies the pd.to_numeric function to convert to numeric format.
#This helps in transforming non-numeric data to numeric, replacing non-convertible values with NaN.
columns_to_convert = Glob_df2.columns[Glob_df2.dtypes == "object"]
Glob_df2[columns_to_convert] = Glob_df2[columns_to_convert].apply(pd.to_numeric)
Glob_df3 = Glob_df2.copy()
Glob_df3.isna().sum() == Glob_df2.isna().sum()
Glob_df3.index
