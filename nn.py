import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.utils import np_utils
import csv
#import numpy.core.defchararray as np_f

df = pd.read_csv('Dataset/train.csv')
df = df[df['length(m)'] >= 0.05] 
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

"""
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
"""
dummies5 = pd.get_dummies(y2,prefix='breed')
dummies5.drop(['breed_2.0'], axis=1, inplace = True)
#df = pd.concat([df,dummies5],axis=1)



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

X = df.iloc[:, :].values

X1 = np.concatenate((X, dummies5),1)
#X1 = X1.values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size = 0.1, random_state = 0)

# one-hot encode the labels
y_train = np_utils.to_categorical(y_train, 5)
y_test = np_utils.to_categorical(y_test, 5)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(512, kernel_initializer='normal',activation='relu'))
NN_model.add(Dropout(0.2))
#NN_model.add(Dense(128, kernel_initializer='normal',activation='relu'))
#NN_model.add(Dropout(0.2))
NN_model.add(Dense(512, kernel_initializer='normal',activation='relu'))
NN_model.add(Dropout(0.2))

# The Output Layer :
NN_model.add(Dense(5, kernel_initializer='normal',activation='softmax'))


NN_model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

checkpoint_path = "weight.txt"
checkpointer = ModelCheckpoint(filepath=checkpoint_path, 
                               verbose=1, save_best_only=True)
NN_model.fit(X_train, y_train, batch_size = 32, epochs= 11,callbacks = [checkpointer], validation_split = 0.1)

score = NN_model.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]

#=========================2nd model===================================
X2_train, X2_test, y2_train, y2_test = train_test_split(X, y2, test_size = 0.1, random_state = 0)
y2_train = np_utils.to_categorical(y2_train, 3)
y2_test = np_utils.to_categorical(y2_test, 3)
sc2 = StandardScaler()
X2_train = sc2.fit_transform(X2_train)
X2_test = sc2.transform(X2_test)
NN_model2 = Sequential()

# The Input Layer :
NN_model2.add(Dense(128, kernel_initializer='normal',input_dim = X2_train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model2.add(Dense(512, kernel_initializer='normal',activation='relu'))
NN_model2.add(Dropout(0.2))
#NN_model2.add(Dense(128, kernel_initializer='normal',activation='relu'))#NN_model2.add(Dropout(0.1))
#NN_model2.add(Dropout(0.2))
NN_model2.add(Dense(512, kernel_initializer='normal',activation='relu'))
NN_model2.add(Dropout(0.2))

# The Output Layer :
NN_model2.add(Dense(3, kernel_initializer='normal',activation='softmax'))


NN_model2.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

checkpoint_path = "weight.txt"
checkpointer = ModelCheckpoint(filepath=checkpoint_path, 
                               verbose=1, save_best_only=True)
NN_model2.fit(X2_train, y2_train, batch_size = 32, callbacks = [checkpointer], epochs= 11, validation_split = 0.1)

score2 = NN_model2.evaluate(X2_test, y2_test, verbose=0)
accuracy2 = 100*score2[1]

#======================================================================

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

"""
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

"""
df_to_test = get_date(df_to_test)


dates_test = df_to_test.iloc[:, -6:].values
Xtest = df_to_test.iloc[:, [0,1,2,3,4]].values
Xtest = np.concatenate((Xtest, encoded_color),1)
Xtest = np.concatenate((Xtest, encoded_condn_test),1)
#Xtest = np.concatenate((Xtest, encoded_X1_test),1)
#Xtest = np.concatenate((Xtest, encoded_X2_test),1)
Xtest = np.concatenate((Xtest, dates_test),1)

X2test = sc2.transform(Xtest)
y2_pred = NN_model2.predict_classes(X2test)
y2_pred =y2_pred.reshape(len(y2_pred))
dummies5 = dummies5.replace(d)
dummies5 = dummies5.iloc[:8072, :].values

for i in range(8072):
    if(y2_pred[i]==0):
        dummies5[i][0] =1
    elif(y2_pred[i]==1):
        dummies5[i][1] =1

Xtest1 = np.concatenate((Xtest, dummies5) ,1)
Xtest1 = sc.transform(Xtest1)

y_pred = NN_model.predict_classes(Xtest1)

final = np.concatenate((final.reshape(len(final),1), y2_pred.reshape(len(y2_pred),1)),1)
final = np.concatenate((final, y_pred.reshape(len(y_pred),1)),1)

fields = ['pet_id', 'breed_category' ,'pet_category']

filename = 'result.csv'
with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(final)



