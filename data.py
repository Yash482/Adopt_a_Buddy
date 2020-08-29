import numpy as np
import pandas as pd
import csv

df = pd.read_csv('Dataset/train.csv')
#df = df[df['length(m)'] != 0] 
df['condition'] = df['condition'].fillna(0.883303)

df['pet_id'] = df['pet_id'].apply(lambda x: float(str(x)[5:]))

col = 'color_type'
if( df[col].dtype == np.dtype('object')):
    dummies = pd.get_dummies(df[col],prefix=col)
    dataset = pd.concat([df,dummies],axis=1)

    #drop the encoded column
    dataset.drop([col],axis = 1 , inplace=True)
color = dataset.iloc[:, 10:]

col2 = 'condition'
dummies2 = pd.get_dummies(df[col2],prefix=col2)
dataset2 = pd.concat([df,dummies2],axis=1)

#drop the encoded column
dataset2.drop([col2],axis = 1 , inplace=True)
conditions = dataset2.iloc[:, -4:]

col3 = 'X1'
dummies3 = pd.get_dummies(df[col3],prefix=col3)
dataset3 = pd.concat([df,dummies3],axis=1)
df_X1 = dataset3.iloc[:, -20:]

col4 = 'X2'
dummies4 = pd.get_dummies(df[col4],prefix=col4)
dataset4 = pd.concat([df,dummies4],axis=1)
df_X2 = dataset4.iloc[:, -10:]

def get_date(df):
   dates = df.iloc[:, 1:3]
   dates['issue_year'] = dates['issue_date'].apply(lambda x: float(str(x)[:4]))
   dates['issue_month'] = dates['issue_date'].apply(lambda x: float(str(x)[5:7]))
   dates['issue_day'] = dates['issue_date'].apply(lambda x: float(str(x)[9:11]))
   dates['listing_year'] = dates['listing_date'].apply(lambda x: float(str(x)[:4]))
   dates['listing_month'] = dates['listing_date'].apply(lambda x: float(str(x)[5:7]))
   dates['listing_day'] = dates['listing_date'].apply(lambda x: float(str(x)[9:11]))
   dates.drop(['issue_date'],axis = 1 , inplace=True)
   dates.drop(['listing_date'],axis = 1 , inplace=True)
   
   return dates

dates = get_date(df)

X5 = df.iloc[:, [5,6]].values
X0 = df.iloc[:, [0]].values
X2 = np.concatenate((X0, dates),1)
X = np.concatenate((X2, conditions),1)
X = np.concatenate((X, color),1)
X = np.concatenate((X, X5),1)
X = np.concatenate((X, df_X1),1)
X = np.concatenate((X, df_X2),1)

df_to_test = pd.read_csv('Dataset/test.csv')
#df = df[df['length(m)'] != 0] 
#df['condition'] = df['condition'].fillna(0.883303)
dates_test = get_date(df_to_test)

train_color =df['color_type'].unique()
test_color =df_to_test['color_type'].unique()

extra_color = []
for color in train_color:
    if color not in test_color:
        extra_color.append(color)



    
