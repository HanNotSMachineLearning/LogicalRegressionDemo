import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

print(' ')
print('******************Data import****************************')
data = pd.read_csv('./banking.csv', header=0)
data = data.dropna()

print(' ')
print('data.shape: ' + str(data.shape))

print(' ')
print('data.column: ')
print(list(data.columns))

print(' ')
print('data education unique before grouping:')
print(data['education'].unique())

data['education']=np.where(data['education'] =='basic.9y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.6y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.4y', 'Basic', data['education'])

print(' ')
print('data education unique after grouping:')
print(data['education'].unique())

print(' ')
print('data y value_counts :')
print(data['y'].value_counts())

print(' ')
print('****************************Creating dummy data*******************')
print(' ')
print("origin banking data before crating dummy")
print(data)

cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1

print(' ')
print("data1 origin data converted with dummy data")
print(data1)

cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
data_final=data[to_keep]

print(' ')
print("after removing the categorical variable and replaced with dummy data")
print(' ')
print("final data column to keep: ")
print(data_final.columns.values)

data_final_vars=data_final.columns.values.tolist()
y=['y']
X=[i for i in data_final_vars if i not in y]

print(' ')
print('****************************Feature selection*******************')

from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

rfe = RFE(logreg, 18)
rfe = rfe.fit(data_final[X], data_final[y].values.ravel())
print(' ')
print("support")
print(rfe.support_)
print(' ')
print("ranking")
print(rfe.ranking_)

cols=["previous", "euribor3m", "job_blue-collar", "job_retired", "job_services", "job_student", "default_no", 
      "month_aug", "month_dec", "month_jul", "month_nov", "month_oct", "month_sep", "day_of_week_fri", "day_of_week_wed", 
      "poutcome_failure", "poutcome_nonexistent", "poutcome_success"] 
X=data_final[cols]
print('')
print('feature selected columns to use to create a model')
print(cols)
y=data_final['y']

print(' ')
print('***************************Implementing the model*******************')
import statsmodels.api as sm
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary())


print(' ')
print('**********************Logistic regression model fitting****************')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
print('')
print('Model fitting with splitted data')
print(logreg.fit(X_train, y_train))

y_pred = logreg.predict(X_test)
print('')
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print(' ')
print('Cross validation on train data')
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

print(' ')
print('*********************Confusion Matrix************************')
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(' ')
print('result of correct prediction and wrong prediction based on data')
print(confusion_matrix)

print(' ')
print('Accuracy')
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

from sklearn.metrics import classification_report
print(' ')
print('Classification report')
print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()