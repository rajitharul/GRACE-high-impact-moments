import pandas as pd
import os 
import pickle

from log_service import LoggingService

def dump_file(dump_data , name):

    logger.info(f'Starting to Dump file {name}')
    # open a file, where you ant to store the data
    file = open(name, 'wb')
    # dump information to that file
    pickle.dump(dump_data, file)
    # close the file
    logger.info(f'Successfully Dumped file {name}')
    file.close()


logger = LoggingService.get_logger(__name__)

logger.info('Starting Multi Class Classification Using Without Any Encoding for the Independant Features Training Service Started')


df_rename_processed_data = pd.read_csv('rename_processed_data.csv')  



df_processed_columns = df_rename_processed_data[["currentPayor", "primaryDenialCode" ,"patient_state" , "primaryDx" ,"touchedCount" , "covid" , "providerProfile" ,"fuStatus" ]]
df_processed_columns


df_processed_data_encoded=df_processed_columns
df_processed_data_encoded

#Split X and Y Values 
df_processed_data_encoded.columns
X = df_processed_data_encoded.drop(["fuStatus"], axis=1)
X
y = df_processed_data_encoded['fuStatus']


# Before lable encoding , Code the output variables to only 4 classes 
#APPEAL, REBILLED_E , WriteOff , Other

for i in range(len(y)):
    if y[i] == 'APPEALED': 
        print(y[i])
    elif y[i] == 'REBILLED_E':
        print(y[i])
    elif y[i] == 'WRITE_OFF':
        print(y[i])
    else:
        print('y changes : ' , y[i])
        y[i] = "OTHER"



from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)
le.classes_

cpuCount = os.cpu_count() - 1


categorical_feats = ['currentPayor', 'primaryDenialCode', 'patient_state', 'primaryDx',
       'touchedCount', 'covid', 'providerProfile']
for c in categorical_feats:
    X[c] = X[c].astype('category')

logger.info('All the features converted into categorical features')
logger.info(X.dtypes)


logger.info('Test Train Data Split Started')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logger.info('Test Train Data Split Ended')

import numpy as np
np.unique(y_test)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

X_test, X_test_validation, y_test, y_test_validation = train_test_split(X_test, y_test, test_size=0.3, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(X_test_validation.shape)
print(y_test_validation.shape)

import lightgbm as lgb
fit_params={'verbose': 100,
            "eval_metric" : 'multi_logloss',
            "eval_set" : [(X_test_validation,y_test_validation)],
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
clf = lgb.LGBMClassifier(n_jobs=cpuCount ,boosting_type='gbdt' , objective = 'multiclass' , metric = 'multi_logloss' ,
                        num_class = 4,num_leaves= 2**15-1
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

# open a file, where you ant to store the data
file = open('best_model.pk', 'wb')
# dump information to that file
pickle.dump(clf_final.fit, file)
# close the file
file.close()


#using precision score for error metrics
from sklearn.metrics import mean_squared_error,roc_auc_score,precision_score

print(precision_score(y_pred_1,y_test,average=None).mean())
logger.info('Precision Score Is , ')
logger.info(precision_score(y_pred_1,y_test,average=None).mean())

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_1)
cm

logger.info('Printing the Confusion matrix ')
logger.info(cm)



from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_1, target_names=['APPEALED', 'OTHER', 'REBILLED_E', 'WRITE_OFF']))

logger.info('Priting the Classification Report')
logger.info(classification_report(y_test, y_pred_1, target_names=['APPEALED', 'OTHER', 'REBILLED_E', 'WRITE_OFF']))


#Dump the Important Files 
dump_file(X_train , 'X_train.pk')
dump_file(y_train , 'y_train.pk')
dump_file(X_test , 'X_test.pk')
dump_file(y_test , 'y_test.pk')
dump_file(le , 'label_encoder.pk')
