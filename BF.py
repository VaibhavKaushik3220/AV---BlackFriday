# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 13:20:16 2017

@author: Vaibhav
"""
##Changing working Directory
import os
os.getcwd()
os.chdir('D:\\Analytics Vidhya\\BlackFriday')

##Reading the excel files
import pandas as pd
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
##Check the Summary
train.head(2)
train.describe()
train.dtypes
##Checking and converting types
train['Product_Category_1'].unique()
train['Product_Category_2'].unique()
train['Product_Category_3'].unique()
train['Occupation'].unique()
train['Age'].unique()
train['City_Category'].unique()
train["Stay_In_Current_City_Years"].unique()
##Lets check if there are any missing values
train.apply(lambda x: sum(x.isnull()))
##Imputing missing category_2,3 with 0
#Lets Convert the age category to numbers
#Concatinating Train and Test to have feature enginerring on both
df=pd.concat([train,test],ignore_index=True)
df.shape,train.shape,test.shape
df.dtypes,train.dtypes,test.dtypes
##Converting age data into numeric numbers
df['Age']=df['Age'].map({'0-17':0, '55+':6, '26-35':2, '46-50':4, '51-55':5, '36-45':3, '18-25':1})
df['Age'].dtypes
df['Age'].unique()
df['Product_Category_2'].dtypes
#Converting Gender,City Category and Stay in Current city  in the same way
df['Gender']=df['Gender'].map({'M':0,'F':1})
df['Gender'].unique()
df['City_Category']=df['City_Category'].map({'A':0,'B':1,'C':2})
df['City_Category'].unique()
df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].map({'2':2, '4+':4, '3':3, '1':1, '0':0})
df['Stay_In_Current_City_Years'].unique()
df['Product_Category_2'].unique()
sum(df['Product_Category_2'].isnull())
df['Product_Category_2'].fillna(0,inplace=True)
df['Product_Category_3'].fillna(0,inplace=True)   


##Lets try running the model with no further modification
df.dtypes
from sklearn.preprocessing import LabelEncoder
le1=LabelEncoder()
df['Product_ID']=le1.fit_transform(df['Product_ID'])

#Divide into Test and Train again
train=df.loc[df['Purchase'].isnull()==False]
test=df.loc[df['Purchase'].isnull()==True]
train.shape,test.shape,df.shape
##train.drop(['Product_Category_2','Product_Category_3'],inplace=True,axis=1)
##test.drop(['Product_Category_2','Product_Category_3'],inplace=True,axis=1)
test.drop(['Purchase'],inplace=True,axis=1)

##1)Lets work on Linear Regression first
target=['Purchase']
predictor=[x for x in train.columns if x not in target]
predictor

from sklearn.linear_model import LinearRegression
clf=LinearRegression(normalize=True)
clf.fit(train[predictor],train[target])
from sklearn import cross_validation
kf_total = cross_validation.KFold(len(train), n_folds=10,shuffle=True, random_state=4)
kf_total
score=cross_validation.cross_val_score(clf, train[predictor], train[target], cv=kf_total, n_jobs=1)
score.mean()##Very bad Linear Regression :(

##2)DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
clf=DecisionTreeRegressor(random_state=0,max_depth=6)
clf.fit(train[predictor],train[target])
kf_total = cross_validation.KFold(len(train), n_folds=10,shuffle=True, random_state=4)
kf_total
score=cross_validation.cross_val_score(clf, train[predictor], train[target], cv=kf_total, n_jobs=1)
score.mean() ##66% accuracy

##3)Random Forrest
from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators = 100, oob_score = True, n_jobs = -1,random_state =50,
                               max_features = 10, min_samples_leaf = 75)
clf.fit(train[predictor],train[target])
kf_total = cross_validation.KFold(len(train), n_folds=10,shuffle=True, random_state=4)
kf_total
score=cross_validation.cross_val_score(clf, train[predictor], train[target], cv=kf_total, n_jobs=1)
score.mean()  ##71.2% accuracy trying with parameter tuning,71.45 after addition of dropped columns     

##4) Running XGboost now   
import os

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-6.2.0-posix-seh-rt_v5-rev1\\mingw64\\bin'

os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']             
import xgboost as xgb
clf=xgboost.XGBClassifier(learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 seed=27)
kf_total = cross_validation.KFold(len(train), n_folds=10,shuffle=True, random_state=4)
score=cross_validation.cross_val_score(clf, train[predictor], train[target], cv=kf_total, n_jobs=1)
score.mean()
