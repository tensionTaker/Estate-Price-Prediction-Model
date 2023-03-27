import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn import metrics

def convert_sqft_to_num(x):
    q = x.split('-')
    if len(q) == 2:
        return (float(q[0]) + float(q[1]))/2
    try:
        return float(x)
    except:
        return None

# This Function maintains that Price Per Sq. Ft. is not way to low or High
# The price of a particular location can't be less than (mean-std) and greater than (mean+std)
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key,SubDf in df.groupby('location'):
        mean = np.mean(SubDf['price_perSqft'])
        st = np.std(SubDf['price_perSqft'])
        df_new = SubDf[(SubDf['price_perSqft']>=(mean-st)) & (SubDf['price_perSqft']<=(mean+st))]
        df_out = pd.concat([df_out,df_new], ignore_index=True)
    return df_out

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_perSqft),
                'std': np.std(bhk_df.price_perSqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_perSqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

df1 = pd.read_csv("bhd.csv")

# Dropping Unnecessary Coloumn
df2 = df1.drop(['area_type','society','balcony','availability'], axis=1)

#Since the number of rows with null value is quite low we can drop them
df3 = df2.dropna()

#Adding a feature
df3['bhk'] = df3['size'].apply(lambda x:int(x.split(' ')[0]))

# Correcting the total area coloumn
df4 = df3.copy()
df4['total_sqft'] = df3['total_sqft'].apply(convert_sqft_to_num)
df4 = df4.dropna()

# Adding a coloumn price_perSqft
df5 = df4.copy()
df5['price_perSqft'] = (df4['price']*100000)/df4['total_sqft']

# Reducing the dimension by placing area with count less than 10
df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5['location'].value_counts(ascending=False)
location_stats_less_than_10 = location_stats[location_stats<=10]
df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

# Removing the outliers such that a single a room can't be less than 300 sqft
df6 = df5.copy()
df6 = df5[~((df5['total_sqft']/df5['bhk'])<300)]

df7 = remove_pps_outliers(df6)
df8 = remove_bhk_outliers(df7)

df9 = df8[df8['bath']<(df8['bhk']+2)]

# Removing the unnecessary but previously used coloumn
df10 = df9.drop(['size', 'price_perSqft'], axis = 1)

# One Hot Encodeing
dummies = pd.get_dummies(df10.location)
df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df12 = df11.drop('location',axis='columns')

# Data Cleaning finish
X = df12.drop('price', axis = 'columns') # We are segregating data and target(price)
Y = df12.price

def Transfer_Data():
    return X

def Transfer_Target():
    return Y
