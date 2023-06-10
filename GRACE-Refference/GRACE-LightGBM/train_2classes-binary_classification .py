print("GRACE")
import pandas as pd
from log_service import LoggingService

logger = LoggingService.get_logger(__name__)

logger.info('Starting Classification Trainig Serice')


df_rename_processed_data = pd.read_csv('rename_processed_data.csv')  
df_processed_columns = df_rename_processed_data[["currentPayor", "primaryDenialCode" ,"patient_state" , "primaryDx" ,"touchedCount" , "covid" , "providerProfile" ,"fuStatus" ]]
df_processed_columns



import category_encoders as ce
#Create object for binary encoding
encoder= ce.BinaryEncoder(cols=['currentPayor', 'primaryDenialCode', 'patient_state', 'primaryDx',
       'touchedCount', 'covid', 'providerProfile'],return_df=True)
df_processed_data_encoded=encoder.fit_transform(df_processed_columns) 
df_processed_data_encoded

logger.info('Dataframe relevant columns Selected')


#Split X and Y Values 
df_processed_data_encoded.columns
X = df_processed_data_encoded.drop(["fuStatus"], axis=1)
X
y = df_processed_data_encoded['fuStatus']

logger.info('Splittig of Data is being done')


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)
le.classes_


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_test_binary = y_test
y_test_binary

y_train_binary = y_train
y_train_binary


for i in range(len(y_train_binary)):
    if y_train_binary[i] == 14:
        print("write off")
        y_train_binary[i] = 0
    else:
        y_train_binary[i] = 1

for i in range(len(y_test_binary)):
    if y_test_binary[i] == 14:
        print("write off")
        y_test_binary[i] = 0
    else:
        y_test_binary[i] = 1




logger.info('Training Started')

import lightgbm as lgb
fit_params={'verbose': 100,
            'eval_names': ['valid']}


from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
param_test ={
             'n_estimators': [1200 , 1500],
             'max_depth' : [15 , 20 , 25 ],
	     'learning_rate ' : [0.01,0.05 ,0.1],
	     'reg_alpha':[0,0.01,0.03],
	     'num_leaves':[20,40,60]
             }


import os 
#This parameter defines the number of HP points to be tested
n_HP_points_to_test = 8

import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

cpuCount = os.cpu_count() - 1

#n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum
clf = lgb.LGBMClassifier(n_jobs=cpuCount ,boosting_type='gbdt' , objective = 'binary' , metric = 'logloss' ,
                        num_class = 2
            )
gs = GridSearchCV(
    estimator=clf, param_grid=param_test, 
    cv=3,
    refit=True,
    verbose=True)

logger.info('Grid Search Training Started')

gs.fit(X_train, y_train, **fit_params)
print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))
logger.info('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))

logger.info('Grid Search Training Ended')



opt_parameters = gs.best_params_
opt_parameters

#Configure locally from hardcoded values
clf_final = lgb.LGBMClassifier(**clf.get_params())
#set optimal parameters
clf_final.set_params(**opt_parameters)

#Train the final model with learning rate decay
clf_final.fit(X_train, y_train, **fit_params,)

logger.info('Prediction Test using the Model')

y_pred_1=clf_final.predict(X_test)

logger.info('Dumping the best Model')

import numpy as np
for i in range(len(y_pred_1)):
    if y_pred_1[i] > 0.5:
       y_pred_1[i] = 1
    else:
       y_pred_1[i] = 0

y_pred_1

#using precision score for error metrics
from sklearn.metrics import mean_squared_error,roc_auc_score,precision_score

print(precision_score(y_pred_1,y_test_binary,average=None).mean())
logger.info('Precision Socreds are')
logger.info(precision_score(y_pred_1,y_test_binary,average=None).mean())


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_binary, y_pred_1)
cm
logger.info('Confusion Matrix is ')
logger.info(cm)



# Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
cm_df = pd.DataFrame(cm,
                     index = ['Write Off' , 'Other'], 
                     columns = ['Write Off' , 'Other'])

from sklearn.metrics import classification_report

print(classification_report(y_test_binary, y_pred_1, target_names=['Write Off' , 'Other']))

logger.info('Classification Report is ')
logger.info(classification_report(y_test_binary, y_pred_1, target_names=['Write Off' , 'Other']))





