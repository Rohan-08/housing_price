import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.show()


df = pd.read_csv(r'F:\Python Notebooks\Housing Prices\train.csv')

df.head()
df.shape

df['SalePrice'].head()

sns.histplot(data=df, x="SalePrice")

plt.hist(df['SalePrice'], bins=100)
plt.show()

df.columns

df.describe()

# As there are a large number  of variables, we will first look at the numeric variables and see how correlated are they with the target variable SalePrice
# This is done to choose the most important numeric variables.
df.dtypes.unique()

numeric = [x for x in df.columns if df.dtypes[x]!= 'object']

numeric.remove('Id')
numeric.remove('SalePrice')

df.isnull().sum()

missing_numeric = df.loc[:,numeric].isnull().sum()
missing_numeric[missing_numeric>0]

numeric_df = df.loc[:,numeric]

numeric_df.shape

numeric_df.corr()


corr = numeric_df.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(numeric_df.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(numeric_df.columns)
ax.set_yticklabels(numeric_df.columns)
plt.show()

sns.heatmap(numeric_df.corr(), cmap='viridis')
plt.show()

# Through the correlation graph, we can see that SalePrice is strongly correlted with OverallQual and GrLivArea

numeric_df[['SalePrice','OverallQual']].corr()
numeric_df[['SalePrice','GrLivArea']].corr()

numeric_df.corr().loc[:,['SalePrice']]>0.5

numeric_df.loc[numeric_df.corr().loc[:,['SalePrice']]>0.5]



numeric_df.loc[numeric_df.corr().loc[:,['SalePrice']]>0.5==True]

corr_df=numeric_df.corr().loc[:,['SalePrice']]>0.5

type(corr_df)

corr_df.sort_index()

# OverallQual         True
# YearBuilt           True
# YearRemodAdd        True
# TotalBsmtSF         True
# 1stFlrSF            True
# GrLivArea           True
# FullBath            True
# TotRmsAbvGrd        True
# GarageCars          True
# GarageArea          True

# Plot between Overall Quality and Sale Price. Because Overall Quality is categorical, we are creating a box plot
sns.boxplot(data=df, x='OverallQual', y='SalePrice')
plt.show()

# Plot between GrLivArea and Sale Price. As both are numerical, we will make a scatter plot

sns.scatterplot(data=df, x='GrLivArea', y='SalePrice')
plt.show()

sns.lmplot(data=df, x='GrLivArea', y='SalePrice')
plt.show()

missing=df.isnull().sum()
missing=missing[missing>0]
missing.sort_values(inplace=True)
missing.plot.bar()
plt.show()

missing

df['PoolQC'].value_counts()

df['PoolQC'].isnull()==True

df.loc[df['PoolQC'].isnull() ==True, 'PoolQC'] = 'NA'

df['PoolQC'].value_counts()

# Label Encoding the variable

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

le.fit(df['PoolQC'])
le.classes_

keys = le.classes_
values = le.transform(le.classes_)
poolqc_enc = dict(zip(keys, values))
print(poolqc_enc)

df['PoolQC_enc'] =le.transform(df['PoolQC'])

df['PoolQC_enc'].value_counts()
df['PoolQC'].value_counts()

df['PoolArea'].value_counts()

df[(df['PoolArea']>0)]['PoolQC_enc'].value_counts()

df.columns

df[df['MiscFeature'].isnull()==True]['MiscFeature']='NA'

df.loc[df['MiscFeature'].isnull()==True, ['MiscFeature']]='NA'

df['MiscFeature'].value_counts()

df.groupby(['MiscFeature'])['SalePrice'].mean().plot(kind='bar')
plt.show()

df['Alley'].value_counts()

df.loc[df['Alley'].isnull()==True, ['Alley']]='NA'
df['Alley'] = df['Alley'].astype(object)
type(df['Alley'][1])

df['Fence'].value_counts()
df['Fence'].isnull().sum()
df.loc[df['Fence'].isnull()==True, ['Fence']]='NA'

df['Fireplaces'].isnull().sum()
df['Fireplaces'].value_counts()

df['FireplaceQu'].isnull().sum()
df['FireplaceQu'].value_counts()

df.loc[df['FireplaceQu'].isnull()==True, ['FireplaceQu']]='NA'

df.groupby(['FireplaceQu'])['SalePrice'].mean().plot(kind='bar')
plt.show()

# Because Fireplace Quality is an ordinal variable, we will encode this to get results

len(df['FireplaceQu'].unique())

le= preprocessing.LabelEncoder()
df['FireplaceQu']=le.fit_transform(df['FireplaceQu'])

df['FireplaceQu'].value_counts()

le.classes_
le.transform(le.classes_)

fp_enc= dict(zip(le.classes_, le.transform(le.classes_)))

df['LotFrontage'].isnull().sum()
sns.histplot(data=df, x='LotFrontage')
plt.show()

df.columns

df['Neighborhood'].isnull().sum()

df.groupby(['Neighborhood'])['LotFrontage'].median()

df.groupby(['Neighborhood'])['LotFrontage'].median().sort_values().plot(kind='bar')
plt.show()

df.groupby(['Neighborhood'])['LotFrontage'].mean().sort_values().plot(kind='bar')
plt.show()

# Imputing LotFrontage values with the median of frontage area grouped by neighborhood feature

df['LotFrontage_bkp']=df['LotFrontage']
df['LotFrontage']=df['LotFrontage_bkp']

df['LotFrontage'].isnull().sum()

df["LotFrontage_1"] = df.groupby("Neighborhood")['LotFrontage']

# QC
df.loc[df['LotFrontage_bkp'].isnull()==True, ['Neighborhood','Id','LotFrontage_bkp','LotFrontage','LotFrontage_1']].head()


df['LotFrontage'] = df['LotFrontage'].fillna(df.groupby('Neighborhood')['LotFrontage'].transform('median'))


df.groupby(['Neighborhood'])['LotFrontage'].median()

# for i in range(len(df)):
#     if df['LotFrontage'][i].isnull()==True:
#         df.loc[i,['LotFrontage_2']] = df.groupby("Neighborhood")['LotFrontage'].median()[df.loc[i, ['Neighborhood']]]

df.columns

df.drop(['LotFrontage_bkp','LotFrontage_1'], inplace=True, axis=1)

df.groupby('LotShape')['SalePrice'].mean().plot(kind='bar')
plt.show()

df.groupby('LotShape')['SalePrice'].median().plot(kind='bar')
plt.show()

df['LotShape'].isnull().sum()

le=preprocessing.LabelEncoder()

df['LotShape_enc'] = le.fit_transform(df['LotShape'])

lotshape_enc= dict(zip(le.classes_, le.transform(le.classes_)))

df['LotConfig'].isnull().sum()

df.groupby('LotConfig')['SalePrice'].mean().plot(kind='bar')
plt.show()

df['GarageType'].isnull().sum()
df['GarageType'].value_counts()

df.loc[df['GarageType'].isnull()==True, ['GarageType']]='NA'

df['GarageType'].value_counts()

df.groupby('GarageType')['SalePrice'].mean().plot(kind='bar')
plt.show()

le=preprocessing.LabelEncoder()
df['GarageType_enc'] = le.fit_transform(df['GarageType'])

grgtype_enc = dict(zip(le.classes_, le.transform(le.classes_)))

# GrgYrBlt
df['GarageYrBlt'].isnull().sum()
df['YearBuilt'].isnull().sum()

df['GarageYrBlt'].fillna(lambda x: x in df['YearBuilt'], inplace=True)

# 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
#        'GarageCond'

df['GarageType_enc'].isnull().sum()
df['GarageYrBlt'].isnull().sum()

df['GarageFinish'].value_counts()

df.loc[df['GarageFinish'].isnull()==True, ['GarageFinish']]='NA'

df.groupby('GarageFinish')['SalePrice'].mean().plot(kind='bar')
plt.show()

le=preprocessing.LabelEncoder()
df['GarageFinish_enc'] = le.fit_transform(df['GarageFinish'])

grgfnsh_enc = dict(zip(le.classes_, le.transform(le.classes_)))

df['GarageCars'].isnull().sum()

df['GarageArea'].isnull().sum()

df['GarageQual'].isnull().sum()

df.loc[df['GarageQual'].isnull()==True, ['GarageQual']]='NA'

df.groupby('GarageQual')['SalePrice'].mean().plot(kind='bar')
plt.show()

le=preprocessing.LabelEncoder()
df['GarageQual_enc'] = le.fit_transform(df['GarageQual'])

grgqlty_enc = dict(zip(le.classes_, le.transform(le.classes_)))


# 

df['GarageCond'].isnull().sum()

df.loc[df['GarageCond'].isnull()==True, ['GarageCond']]='NA'

df.groupby('GarageCond')['SalePrice'].mean().plot(kind='bar')
plt.show()

le=preprocessing.LabelEncoder()
df['GarageCond_enc'] = le.fit_transform(df['GarageCond'])

grgcond_enc = dict(zip(le.classes_, le.transform(le.classes_)))


# Basement

df['BsmtQual'].isnull().sum()

df.loc[df['BsmtQual'].isnull()==True, ['BsmtQual']]='NA'

df.groupby('BsmtQual')['SalePrice'].mean().plot(kind='bar')
plt.show()

le=preprocessing.LabelEncoder()
df['BsmtQual_enc'] = le.fit_transform(df['BsmtQual'])

bsmtQual_enc = dict(zip(le.classes_, le.transform(le.classes_)))

# Condition
df['BsmtCond'].isnull().sum()

df.loc[df['BsmtCond'].isnull()==True, ['BsmtCond']]='NA'

df.groupby('BsmtCond')['SalePrice'].mean().plot(kind='bar')
plt.show()

le=preprocessing.LabelEncoder()
df['BsmtCond_enc'] = le.fit_transform(df['BsmtCond'])

bsmtCond_enc = dict(zip(le.classes_, le.transform(le.classes_)))

#Exposure 
df['BsmtExposure'].isnull().sum()

df.loc[df['BsmtExposure'].isnull()==True, ['BsmtExposure']]='NA'

df.groupby('BsmtExposure')['SalePrice'].mean().plot(kind='bar')
plt.show()

le=preprocessing.LabelEncoder()
df['BsmtExp_enc'] = le.fit_transform(df['BsmtExposure'])

bsmtExp_enc = dict(zip(le.classes_, le.transform(le.classes_)))


#FinType1 
df['BsmtFinType1'].isnull().sum()

df.loc[df['BsmtFinType1'].isnull()==True, ['BsmtFinType1']]='NA'

df.groupby('BsmtFinType1')['SalePrice'].mean().plot(kind='bar')
plt.show()

le=preprocessing.LabelEncoder()
df['BsmtFinType1_enc'] = le.fit_transform(df['BsmtFinType1'])

bsmtfin_enc = dict(zip(le.classes_, le.transform(le.classes_)))


#FinType2 
df['BsmtFinType2'].isnull().sum()

df.loc[df['BsmtFinType2'].isnull()==True, ['BsmtFinType2']]='NA'

df.groupby('BsmtFinType2')['SalePrice'].mean().plot(kind='bar')
plt.show()

le=preprocessing.LabelEncoder()
df['BsmtFinType2_enc'] = le.fit_transform(df['BsmtFinType2'])

bsmtfin2_enc = dict(zip(le.classes_, le.transform(le.classes_)))

#Basement
 
df['BsmtFinSF1'].isnull().sum()
df['BsmtFinSF1'].head()

df['BsmtFinSF2'].isnull().sum()
df['BsmtFinSF2'].head()

df['BsmtUnfSF'].isnull().sum()
df['BsmtUnfSF'].head()

df['TotalBsmtSF'].isnull().sum()
df['TotalBsmtSF'].head()

df['BsmtFullBath'].isnull().sum()
df['BsmtFullBath'].head()
df['BsmtFullBath'].value_counts()

df['BsmtHalfBath'].isnull().sum()
df['BsmtHalfBath'].head()
df['BsmtHalfBath'].value_counts()

df['MasVnrType'].head()
df['MasVnrType'].value_counts()
df['MasVnrType'].isnull().sum()

df['MasVnrArea'].head()
df['MasVnrArea'].isnull().sum()
len(df.loc[df['MasVnrArea']==0])

df[(df['MasVnrType']=='None') & (df['MasVnrArea']!=0)][['MasVnrType','MasVnrArea']]

# Masonry Veneer Area was 1.0 sqft for 2 houses. this would most likely be a data glitch as area 1sqft is way too less. Hence, making them as zero

df.loc[df['MasVnrArea']==1,['MasVnrArea']]=0.0
                               
df[df['MasVnrType'].isnull()][['MasVnrType','MasVnrArea']]

df['MasVnrType'].value_counts()

df['MasVnrType'].isnull().sum()

df.groupby(['Neighborhood','MasVnrType'])['Id'].count()

df.loc[(df['MasVnrType']=='None') & (df['MasVnrArea']!=0), ['MasVnrArea']]=0

# There were three rows where type was None but area was given. I have made them null

df.groupby('MasVnrType')['SalePrice'].mean().plot(kind='bar')
plt.show()

le = preprocessing.LabelEncoder()
df['MasVnrType_enc'] = le.fit_transform(df['MasVnrType'])

masvnrtype= dict(zip(le.classes_, le.transform(le.classes_)))


# MS Zoning

df['MSZoning'].value_counts()
df['MSZoning'].isnull().sum()

df.groupby('MSZoning')['SalePrice'].mean().plot(kind='bar')
plt.show()

le = preprocessing.LabelEncoder()
df['MSZoning_enc'] = le.fit_transform(df['MSZoning'])
mszoning= dict(zip(le.classes_, le.transform(le.classes_)))


# Kitchens
df['KitchenQual'].value_counts()
df['KitchenQual'].isnull().sum()

df.groupby('KitchenQual')['SalePrice'].mean().plot(kind='bar')
plt.show()

le = preprocessing.LabelEncoder()
df['KitchenQual_enc'] = le.fit_transform(df['KitchenQual'])
kitchenqual= dict(zip(le.classes_, le.transform(le.classes_)))

df['KitchenAbvGr'].head()
df['KitchenAbvGr'].isnull().sum()

# Utilities

df['Utilities'].head()
df['Utilities'].isnull().sum()
df['Utilities'].value_counts()


df.groupby('Utilities')['SalePrice'].mean().plot(kind='bar')
plt.show()

# Functional
df['Functional'].head()
df['Functional'].value_counts()
df['Functional'].isnull().sum()

df.groupby('Functional')['SalePrice'].mean().plot(kind='bar')
plt.show()

le = preprocessing.LabelEncoder()
df['Functional_enc'] = le.fit_transform(df['Functional'])
functional= dict(zip(le.classes_, le.transform(le.classes_)))

# Exterior

df['Exterior1st'].value_counts()
df['Exterior1st'].isnull().sum()

df.groupby('Exterior1st')['SalePrice'].mean().plot(kind='bar')
plt.show()

le = preprocessing.LabelEncoder()
df['Exterior1st_enc'] = le.fit_transform(df['Exterior1st'])
exterior1st_= dict(zip(le.classes_, le.transform(le.classes_)))

# 
df['Exterior2nd'].value_counts()
df['Exterior2nd'].isnull().sum()

df.groupby('Exterior2nd')['SalePrice'].mean().plot(kind='bar')
plt.show()

le = preprocessing.LabelEncoder()
df['Exterior2nd_enc'] = le.fit_transform(df['Exterior2nd'])
exterior2_= dict(zip(le.classes_, le.transform(le.classes_)))

# 
df['ExterQual'].value_counts()
df['ExterQual'].isnull().sum()

df.groupby('ExterQual')['SalePrice'].mean().plot(kind='bar')
plt.show()

le = preprocessing.LabelEncoder()
df['ExterQual_enc'] = le.fit_transform(df['ExterQual'])
exterQual= dict(zip(le.classes_, le.transform(le.classes_)))

# 
df['ExterCond'].value_counts()
df['ExterCond'].isnull().sum()

df.groupby('ExterCond')['SalePrice'].mean().plot(kind='bar')
plt.show()

le = preprocessing.LabelEncoder()
df['ExterCond_enc'] = le.fit_transform(df['ExterCond'])
exterCond= dict(zip(le.classes_, le.transform(le.classes_)))


# Electrical

df['Electrical'].head()
df['Electrical'].value_counts()
df['Electrical'].isnull().sum()


# replacing one null value with the most common electrical type
df.loc[df['Electrical'].isnull(), ['Electrical']]='SBrkr'

df.groupby('Electrical')['SalePrice'].mean().plot(kind='bar')
plt.show()

le = preprocessing.LabelEncoder()
df['Electrical_enc'] = le.fit_transform(df['Electrical'])
Electrical= dict(zip(le.classes_, le.transform(le.classes_)))

# SaleType

df['SaleType'].value_counts()
df['SaleType'].isnull().sum()

df.groupby('SaleType')['SalePrice'].mean().plot(kind='bar')
plt.show()

le = preprocessing.LabelEncoder()
df['SaleType_enc'] = le.fit_transform(df['SaleType'])
saletype= dict(zip(le.classes_, le.transform(le.classes_)))

# SaleCondition

df['SaleCondition'].value_counts()
df['SaleCondition'].isnull().sum()

df.groupby('SaleCondition')['SalePrice'].mean().plot(kind='bar')
plt.show()

le = preprocessing.LabelEncoder()
df['SaleCondition_enc'] = le.fit_transform(df['SaleCondition'])
saleCondition= dict(zip(le.classes_, le.transform(le.classes_)))

df.isnull().sum().plot(kind='bar')
plt.show()

# Foundation

df['Foundation'].head()
df['Foundation'].value_counts()

df['Foundation'].isnull().sum()

df.groupby('Foundation')['SalePrice'].mean().plot(kind='bar')
plt.show()

le = preprocessing.LabelEncoder()
df['Foundation_enc'] = le.fit_transform(df['Foundation'])
foundation= dict(zip(le.classes_, le.transform(le.classes_)))


# Heating
df['Heating'].head()
df['Heating'].value_counts()

df['Heating'].isnull().sum()

df.groupby('Heating')['SalePrice'].mean().plot(kind='bar')
plt.show()

le = preprocessing.LabelEncoder()
df['Heating_enc'] = le.fit_transform(df['Heating'])
heating= dict(zip(le.classes_, le.transform(le.classes_)))

# Heating QC
df['HeatingQC'].head()
df['HeatingQC'].value_counts()

df['HeatingQC'].isnull().sum()

df.groupby('HeatingQC')['SalePrice'].mean().plot(kind='bar')
plt.show()

le = preprocessing.LabelEncoder()
df['HeatingQC_enc'] = le.fit_transform(df['HeatingQC'])
heatingQC= dict(zip(le.classes_, le.transform(le.classes_)))

# central Air

df['CentralAir'].head()
df['CentralAir'].value_counts()

df['CentralAir'].isnull().sum()

df.groupby('CentralAir')['SalePrice'].mean().plot(kind='bar')
plt.show()

le = preprocessing.LabelEncoder()
df['CentralAir_enc'] = le.fit_transform(df['CentralAir'])
centralAir= dict(zip(le.classes_, le.transform(le.classes_)))

# Roof
df['RoofStyle'].head()
df['RoofStyle'].value_counts()

df['RoofStyle'].isnull().sum()

df.groupby('RoofStyle')['SalePrice'].mean().plot(kind='bar')
plt.show()

le = preprocessing.LabelEncoder()
df['RoofStyle_enc'] = le.fit_transform(df['RoofStyle'])
roofstyle= dict(zip(le.classes_, le.transform(le.classes_)))

# Roof Matl

df['RoofMatl'].head()
df['RoofMatl'].value_counts()

df['RoofMatl'].isnull().sum()

df.groupby('RoofMatl')['SalePrice'].mean().plot(kind='bar')
plt.show()

le = preprocessing.LabelEncoder()
df['RoofMatl_enc'] = le.fit_transform(df['RoofMatl'])
roofmatl= dict(zip(le.classes_, le.transform(le.classes_)))



# LandContour


df['LandContour'].head()
df['LandContour'].value_counts()

df['LandContour'].isnull().sum()

df.groupby('LandContour')['SalePrice'].mean().plot(kind='bar')
plt.show()

le = preprocessing.LabelEncoder()
df['LandContour_enc'] = le.fit_transform(df['LandContour'])
LandContour= dict(zip(le.classes_, le.transform(le.classes_)))


# LandSlope

df['LandSlope'].head()
df['LandSlope'].value_counts()

df['LandSlope'].isnull().sum()

df.groupby('LandSlope')['SalePrice'].mean().plot(kind='bar')
plt.show()

le = preprocessing.LabelEncoder()
df['LandSlope_enc'] = le.fit_transform(df['LandSlope'])
Landslope= dict(zip(le.classes_, le.transform(le.classes_)))


# BldgType

df['BldgType'].head()
df['BldgType'].value_counts()

df['BldgType'].isnull().sum()

df.groupby('BldgType')['SalePrice'].mean().plot(kind='bar')
plt.show()

le = preprocessing.LabelEncoder()
df['BldgType_enc'] = le.fit_transform(df['BldgType'])
BldgType= dict(zip(le.classes_, le.transform(le.classes_)))

# HouseStyle

df['HouseStyle'].head()
df['HouseStyle'].value_counts()

df['HouseStyle'].isnull().sum()

df.groupby('HouseStyle')['SalePrice'].mean().plot(kind='bar')
plt.show()

le = preprocessing.LabelEncoder()
df['HouseStyle_enc'] = le.fit_transform(df['HouseStyle'])
HouseStyle= dict(zip(le.classes_, le.transform(le.classes_)))


# Neighborhood

le = preprocessing.LabelEncoder()
df['Neighborhood_enc'] = le.fit_transform(df['Neighborhood'])
Neighborhood= dict(zip(le.classes_, le.transform(le.classes_)))

le = preprocessing.LabelEncoder()
df['Condition1_enc'] = le.fit_transform(df['Condition1'])
Condition1= dict(zip(le.classes_, le.transform(le.classes_)))

le = preprocessing.LabelEncoder()
df['Condition2_enc'] = le.fit_transform(df['Condition2'])
Condition2= dict(zip(le.classes_, le.transform(le.classes_)))

# Street

le = preprocessing.LabelEncoder()
df['Street_enc'] = le.fit_transform(df['Street'])
Street= dict(zip(le.classes_, le.transform(le.classes_)))

le = preprocessing.LabelEncoder()
df['PavedDrive_enc'] = le.fit_transform(df['PavedDrive'])
PavedDrive= dict(zip(le.classes_, le.transform(le.classes_)))

# MSSubClass

df['MSSubClass'].head()
df['MSSubClass'].value_counts()

df['MSSubClass'].isnull().sum()


df.columns

df.dtypes.value_counts()

df.head()
df.drop('Id', axis=1, inplace=True)

cat = [x for x in df.columns if df.dtypes[x]=='object']
num = [x for x in df.columns if df.dtypes[x]!='object']


df['GarageYrBlt'].dtypes
df['GarageYrBlt'].astype(str).astype(float)


num
cat

len(num)
len(cat)

num_df = df.loc[:, num]
num_df.shape

corr = num_df.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(num_df.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(num_df.columns)
ax.set_yticklabels(num_df.columns)
plt.show()

type(corr)

corr.head()

corr['SalePrice']>0.5

corr.loc[corr['SalePrice']>0.5,'SalePrice']

corr.loc['FullBath','SalePrice']

corr.loc[abs(corr['SalePrice'])>0.5, 'SalePrice']

df['GarageFinish']

num_df['GarageFinish_enc'].head()
num_df.columns

df[['GarageYrBlt', 'SalePrice']].corr()

df[['FireplaceQu', 'SalePrice']].corr()

df['GarageYrBlt'].unique()


df['GarageYrBlt'][1]


lst=[]

df['GarageYrBlt'].apply(np.isreal).unique()

df.drop('GarageYrBlt', axis=1, inplace=True)

from sklearn.model_selection import train_test_split

X=num_df.drop('SalePrice', axis=1)
y=df['SalePrice']

X=

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.ensemble import RandomForestClassifier
feature_names = [f"feature {i}" for i in range(X.shape[1])]
forest = RandomForestClassifier(random_state=0)
forest.fit(X_train, y_train)

# Feature importances are provided by the fitted attribute feature_importances_ 
# and they are computed as the mean and standard deviation of accumulation of the impurity decrease within each tree.

forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

forest_importances = pd.Series(forest.feature_importances_, index=X_train.columns)
forest_importances.sort_values(ascending=False)


fig, ax = plt.subplots()
forest_importances.sort_values(ascending=False).plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()

df['TotBathrooms'] = df['FullBath'] + df['HalfBath']*0.5 + df['BsmtFullBath'] + df['BsmtHalfBath']*0.5

df['Remod'] = df[['YearBuilt','YearRemodAdd']].apply(lambda row: 0 if row[1]==row[2] else 1)

for i in range(len(df)):
    if df['YearBuilt'][i]==df['YearRemodAdd'][i]:
        lst.append(0)
    else:
        lst.append(1)

df['Remod']=lst
df['Age']= df['YrSold'] - df['YearRemodAdd']

sns.scatterplot(x='Age',y='SalePrice', data=df)
plt.show()

lst=[]
for i in range(len(df)):
    if df['YearBuilt'][i]==df['YrSold'][i]:
        lst.append(1)
    else:
        lst.append(0)

df['brandNew']=lst

df['brandNew'].value_counts()

df.groupby('Neighborhood')['SalePrice'].mean().sort_values().plot(kind='bar')
plt.show()

df.groupby('Neighborhood')['SalePrice'].median().sort_values().plot(kind='bar')
plt.show()

lst=[]
for i in range(len(df   )):
    if (df['Neighborhood'][i] =='MeadowV') | (df['Neighborhood'][i] =='IDOTRR') | (df['Neighborhood'][i] =='BrDale'):
        lst.append(0)
    elif (df['Neighborhood'][i] =='StoneBr') | (df['Neighborhood'][i] =='NoRidge') | (df['Neighborhood'][i] =='NridgHt'):
        lst.append(1)
    else:
        lst.append(2)


df['NbrBin'] =lst

df['TotSqFt'] = df['GrLivArea'] + df['TotalBsmtSF']

df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']

dropVars= ['YearRemodAdd', 'GarageArea', 'GarageCond', 'TotalBsmtSF', 'BsmtFinSF1']


df.drop(dropVars, axis=1, inplace=True)

df.dtypes.unique()

num= [x for x in df.columns if df.dtypes[x]!='object'] 

num_df = df.loc[:, num]

num_df.dtypes.unique()

d = preprocessing.normalize(num_df)
scaled_df = pd.DataFrame(d, columns=num_df.columns)
scaled_df.head()

df['logSP'] = np.log(df['SalePrice'])
num_df['logSP'] = np.log(num_df['SalePrice'])

sns.histplot(num_df['logSP'], kde=True)
plt.show()

# final breakdown to training and test data

# Now we just need to run the Lasso and XGBoost model on the dataset

X= num_df.drop(['SalePrice', 'logSP'],axis=1)
y=num_df['logSP']



from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, RepeatedKFold, GridSearchCV

lasso_model = Lasso(alpha=1.0)

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(lasso_model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

scores = np.abs(scores)
print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)

grid = dict()
grid['alpha'] = np.arange(0, 1, 0.01)

search = GridSearchCV(lasso_model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X, y)
# summarize
print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)

lasso_model = Lasso(alpha=0.01)

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(lasso_model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
scores = np.abs(scores)
print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

