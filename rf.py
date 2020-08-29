import numpy as np
import pandas as pd
import csv

df = pd.read_csv('Dataset/train.csv')
df = df[df['length(m)'] >= 0.1] 
df = df[df['X1'] <= 18] 
df = df[df['color_type'] != 'Brown Tiger'] 
df = df[df['color_type'] != 'Black Tiger'] 

#df['condition'] = df['condition'].fillna(0.883303)

y = df.iloc[:, -1].values
y2 = df.iloc[:, -2].values

df.drop(['breed_category' , 'pet_category'],axis = 1 , inplace=True)

df['pet_id'] = df['pet_id'].apply(lambda x: float(str(x)[5:]))

col = 'color_type'
if( df[col].dtype == np.dtype('object')):
    dummies = pd.get_dummies(df[col],prefix=col)
    df = pd.concat([df,dummies],axis=1)

    #drop the encoded column
    df.drop([col],axis = 1 , inplace=True)

#color = dataset.iloc[:, 10:]

col2 = 'condition'
dummies2 = pd.get_dummies(df[col2],prefix=col2)
df = pd.concat([df,dummies2],axis=1)
df.drop([col2],axis = 1 , inplace=True)
#conditions = dataset2.iloc[:, -3:]
df.drop(['color_type_Apricot'],axis = 1 , inplace=True)


col3 = 'X1'
dummies3 = pd.get_dummies(df[col3],prefix=col3)
df = pd.concat([df,dummies3],axis=1)
df.drop([col3], axis=1, inplace = True)
df.drop(['X1_0'], axis=1, inplace = True)

col4 = 'X2'
dummies4 = pd.get_dummies(df[col4],prefix=col4)
df = pd.concat([df,dummies4],axis=1)
df.drop([col4], axis=1, inplace = True)
df.drop(['X2_0'], axis=1, inplace = True)


def get_date(df):
   df['issue_year'] = df['issue_date'].apply(lambda x: float(str(x)[:4]))
   df['issue_month'] = df['issue_date'].apply(lambda x: float(str(x)[5:7]))
   df['issue_day'] = df['issue_date'].apply(lambda x: float(str(x)[9:11]))
   df['listing_year'] = df['listing_date'].apply(lambda x: float(str(x)[:4]))
   df['listing_month'] = df['listing_date'].apply(lambda x: float(str(x)[5:7]))
   df['listing_day'] = df['listing_date'].apply(lambda x: float(str(x)[9:11]))
   df.drop(['issue_date'],axis = 1 , inplace=True)
   df.drop(['listing_date'],axis = 1 , inplace=True)
   
   return df

df = get_date(df)

dummies6 = pd.get_dummies(df['listing_year'], prefix='ly')
df = pd.concat([df,dummies6],axis=1)
dummies6 = pd.get_dummies(df['listing_month'], prefix='lm')
df = pd.concat([df,dummies6],axis=1)
dummies6 = pd.get_dummies(df['listing_day'], prefix='ld')
df = pd.concat([df,dummies6],axis=1)
dummies6 = pd.get_dummies(df['issue_year'], prefix='iy')
df = pd.concat([df,dummies6],axis=1)
dummies6 = pd.get_dummies(df['issue_month'], prefix='im')
df = pd.concat([df,dummies6],axis=1)
dummies6 = pd.get_dummies(df['issue_day'], prefix='id')
df = pd.concat([df,dummies6],axis=1)
df.drop(['listing_year', 'listing_month', 'listing_day'], axis=1, inplace = True)
df.drop(['issue_year', 'issue_month', 'issue_day'], axis=1, inplace = True)
df.drop(['iy_1994.0', 'iy_1997.0', 'iy_1995.0', 'iy_1998.0', 'id_0.0'], axis=1, inplace = True)
df.drop(['ly_2015.0', 'lm_1.0', 'ld_0.0', 'ld_1.0', 'im_1.0', 'id_1.0'], axis=1, inplace = True)


X = df.iloc[:, 1:].values

X1 = np.concatenate((X, y2.reshape(len(y2),1)),1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size = 0.01, random_state = 0)

#training
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators =12, criterion = 'entropy' , random_state =0)
classifier.fit(X_train, y_train)

from sklearn.metrics import  accuracy_score
y_pred = classifier.predict(X_test)
print(accuracy_score(y_test, y_pred))


#=============2nd model===========================================
X2_train, X2_test, y2_train, y2_test = train_test_split(X, y2, test_size = 0.01, random_state = 0)

#training
classifier2 = RandomForestClassifier(n_estimators =15, criterion = 'entropy' , random_state =0)
classifier2.fit(X2_train, y2_train)

from sklearn.metrics import  accuracy_score
y2_pred = classifier2.predict(X2_test)
print(accuracy_score(y2_test, y2_pred))

#===========================================================

df_to_test = pd.read_csv('Dataset/test.csv')
final = df_to_test.iloc[:, :1].values
#df_to_test['condition'] = df_to_test['condition'].fillna(0.883303)
df_to_test['pet_id'] = df_to_test['pet_id'].apply(lambda x: float(str(x)[5:]))

colors = df_to_test.iloc[:, [4]].values
d = { 1 : 0}
dummies = dummies.replace(d)
encoded_color = dummies.iloc[:8072, :].values
for i in range(len(colors)):
    color_name = 'color_type_' +str(colors[i][0])
    encoded_color[i][dummies.columns.get_loc(color_name)] =1
#df_to_test['length(m)'] = df_to_test['length(m)'].apply(lambda x: float(float(x)*100))
encoded_color = np.delete(encoded_color, 1, 1)
df_to_test.drop(['color_type'], axis =1, inplace = True)

condn_test = df_to_test.iloc[:, [3]].values
dummies2 = dummies2.replace(d)
encoded_condn_test = dummies2.iloc[:8072, :].values
for i in range(len(condn_test)):
    if condn_test[i][0] >=0:
        condn_test_name = 'condition_' +str(condn_test[i][0])
        encoded_condn_test[i][dummies2.columns.get_loc(condn_test_name)] =1
df_to_test.drop(['condition'], axis =1, inplace = True)


dfx1_test = df_to_test.iloc[:, [-2]].values
dummies3 = dummies3.replace(d)
encoded_X1_test = dummies3.iloc[:8072, :].values
for i in range(len(dfx1_test)):
    X1_test_name = 'X1_' +str(dfx1_test[i][0])
    encoded_X1_test[i][dummies3.columns.get_loc(X1_test_name)] =1
df_to_test.drop(['X1'], axis =1, inplace = True)
encoded_X1_test = np.delete(encoded_X1_test, 0, 1)

dfx2_test = df_to_test.iloc[:, [-1]].values
dummies4 = dummies4.replace(d)
encoded_X2_test = dummies4.iloc[:8072, :].values
for i in range(len(dfx2_test)):
    X2_test_name = 'X2_' +str(dfx2_test[i][0])
    encoded_X2_test[i][dummies4.columns.get_loc(X2_test_name)] =1
df_to_test.drop(['X2'], axis =1, inplace = True)
encoded_X2_test = np.delete(encoded_X2_test, 0, 1)


df_to_test = get_date(df_to_test)

dummies6 = pd.get_dummies(df_to_test['listing_year'], prefix='ly')
df_to_test = pd.concat([df_to_test,dummies6],axis=1)
dummies6 = pd.get_dummies(df_to_test['listing_month'], prefix='lm')
df_to_test = pd.concat([df_to_test,dummies6],axis=1)
dummies6 = pd.get_dummies(df_to_test['listing_day'], prefix='ld')
df_to_test = pd.concat([df_to_test,dummies6],axis=1)
dummies6 = pd.get_dummies(df_to_test['issue_year'], prefix='iy')
df_to_test = pd.concat([df_to_test,dummies6],axis=1)
dummies6 = pd.get_dummies(df_to_test['issue_month'], prefix='im')
df_to_test = pd.concat([df_to_test,dummies6],axis=1)
dummies6 = pd.get_dummies(df_to_test['issue_day'], prefix='id')
df_to_test = pd.concat([df_to_test,dummies6],axis=1)
df_to_test.drop(['listing_year', 'listing_month', 'listing_day'], axis=1, inplace = True)
df_to_test.drop(['issue_year', 'issue_month', 'issue_day'], axis=1, inplace = True)
df_to_test.drop(['iy_1993.0', 'iy_1996.0', 'iy_1998.0', 'id_0.0'], axis=1, inplace = True)
df_to_test.drop(['ly_2015.0', 'lm_1.0', 'ld_0.0', 'ld_1.0', 'im_1.0', 'id_1.0'], axis=1, inplace = True)



#print(df_to_test['issue_year'].value_counts())
#print(df_to_test['listing_year'].value_counts())


dates_test = df_to_test.iloc[:, 3:].values
Xtest = df_to_test.iloc[:, [1,2]].values
Xtest = np.concatenate((Xtest, encoded_color),1)
Xtest = np.concatenate((Xtest, encoded_condn_test),1)
Xtest = np.concatenate((Xtest, encoded_X1_test),1)
Xtest = np.concatenate((Xtest, encoded_X2_test),1)
Xtest = np.concatenate((Xtest, dates_test),1)

#X2test = sc2.transform(Xtest)
y2_pred_test = classifier2.predict(Xtest)
Xtest1 = np.concatenate((Xtest, y2_pred_test.reshape(len(y2_pred_test),1)),1)
#Xtest1 = sc.transform(Xtest1)

y_pred_test = classifier.predict(Xtest1)

final = np.concatenate((final.reshape(len(final),1), y2_pred_test.reshape(len(y2_pred_test),1)),1)
final = np.concatenate((final, y_pred_test.reshape(len(y_pred_test),1)),1)

fields = ['pet_id', 'breed_category' ,'pet_category']

filename = 'result.csv'
with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(final)


