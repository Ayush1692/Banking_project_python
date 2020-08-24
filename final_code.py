# IPython log file


get_ipython().run_line_magic('cls', '')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#pip install scikit-plot
import scikitplot as skplt

#Read data file
df = pd.read_csv("bank_data.csv")
#view first five records of data file
df.head()

#Heat Map
import seaborn as sns
corrmat = df.corr(method = 'spearman')
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


#Combine previous and pdays columns into single column pdays1
#Convert them into a single Contacted/Non Contacted Column
df.loc[(df.pdays == 999), 'pdays1']= 0
df.loc[(df.pdays < 999), 'pdays1']= 1
#view first five records of data file
df.head()
#Gives the information on the columns
df.info()
#Convert Float into int
df['pdays1'] = df['pdays1'].astype(int)
#Display the contents of pdays1
df.pdays1
#view first five records of data file
df.head()
#Drop pdays and Previous Column
df.drop('pdays', axis = 1, inplace = True)
#view first five records of data file
df.head()
df.drop('previous', axis = 1, inplace = True)
#view first five records of data file
df.head()
#Copy contents of df into df1
df1 = df.copy()
#Check for records which have default = yes
df1.loc[df1.default == 'yes']
#Three records
#Drop three records which have default = yes
df1 = df1.drop(df1[df1.default == 'yes'].index)
#Check whether the records have been dropped
df1.loc[df1.default == 'yes']
#Create dummy variables for Default, housing, Loan, contact, outcome, y
df1 = pd.get_dummies(df1, columns = ["default"])
df1.head()
#Drop default_unknown
df1.drop('default_unknown', axis = 1 , inplace = True)
df1 = pd.get_dummies(df1, columns = ["housing"])
df1.head()
#drop housing_unknown
df1.drop('housing_unknown', axis = 1 , inplace = True)
df1.head()
df1 = pd.get_dummies(df1, columns = ["loan"])
df1.head()
#Drop loan_unknown
df1.drop('loan_unknown', axis = 1 , inplace = True)
df1 = pd.get_dummies(df1, columns = ["contact"])
df1.head()
df1.drop('contact_telephone', axis = 1 , inplace = True)
df1.head()
df1 = pd.get_dummies(df1, columns = ["poutcome"])
df1.head()
#Drop poutcome_nonexistent
df1.drop('poutcome_nonexistent', axis = 1 , inplace = True)
df1.head()
df1 = pd.get_dummies(df1, columns = ["y"])
df1.head()
#Drop y_no
df1.drop('y_no', axis = 1 , inplace = True)
df1.head()
#Copy df1 into df2
df2 = df1.copy()
corr_matrix = df.corr().abs()
#View correlation matrix
corr_matrix
#Select the upper triangle of the correlation marix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
#Find index of feature columns with correlation greater than 0.90
to_drop = [column for column in upper.columns if any(upper[column] > 0.70)]
#Drop features and assign to df4
df3 = df2.drop(df[to_drop], axis=1)
df3
#View information of the coumns of df4
df3.info()
df3['job'] = df3['job'].astype('category')
df3['marital'] = df3['marital'].astype('category')
df3['education'] = df3['education'].astype('category')
df3['month'] = df3['month'].astype('category')
df3['day_of_week'] = df3['day_of_week'].astype('category')
#Recoding the categories of job column
#creating a dictionary
replace_map = {'job': {'admin.': 1, 'blue-collar': 2, 'entrepreneur': 3, 'housemaid': 4,
'management': 5, 'retired': 6, 'self-employed': 7 , 'services': 8 , 'student': 9,'technician': 10,'unemployed': 11 , 'unknown' : 12}}
df4 = df3.copy()
#Replacing the categories with corresponding values in the dictionary
df4.replace(replace_map , inplace = True)
replace_map = {'job': {'admin.': 1, 'blue-collar': 2, 'entrepreneur': 3, 'housemaid': 4,

                                  'management': 5, 'retired': 6, 'self-employed': 7 , 'services': 8 , 'student': 9,'technician': 10,'unemployed': 11 , 'unknown' : 12}}



df4 = df3.copy()

#Replacing the categories with corresponding values in the dictionary

df4.replace(replace_map , inplace = True)
replace_map = {'marital': {'divorced': 1, 'married': 2, 'single': 3, 'unknown': 4}}



df4.replace(replace_map, inplace=True)





#Recoding the categories of education column





#Combining 3 categories into one basic

df4['education']=np.where(df4['education'] =='basic.9y', 'basic', df4['education'])

df4['education']=np.where(df4['education'] =='basic.6y', 'basic', df4['education'])

df4['education']=np.where(df4['education'] =='basic.4y', 'basic', df4['education'])



#creating a dictionary



replace_map = {'education': {'basic': 1, 'high.school': 2,

                                  'illiterate': 3, 'professional.course': 4, 'university.degree': 5 , 'unknown': 6}}



#Replacing the categories with corresponding values in the dictionary

df4.replace(replace_map, inplace=True)







#Recoding the categories of month column

#creating a dictionary

replace_map = {'month': {'mar': 1, 'apr': 2, 'may': 3, 'jun': 4,

                                  'jul': 5, 'aug': 6, 'sep': 7 , 'oct': 8, 'nov':9, 'dec':10}}





#Replacing the categories with corresponding values in the dictionary

df4.replace(replace_map, inplace = True)







#Recoding the categories of day_of_week column

#creating a dictionary



replace_map = {'day_of_week': {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4,

                                  'fri': 5}}



#Replacing the categories with corresponding values in the dictionary

df4.replace(replace_map, inplace = True)



df4
df4.info()




X = df4.iloc[:,0:19] #Independent Variables



y = df4.y_yes #Target Variable


#Calculating the column contribution of different variables

from sklearn.ensemble import RandomForestClassifier
feat_labels = df4.columns[0:19]
forest = RandomForestClassifier(n_estimators = 10000, random_state = 0, n_jobs = -1)
forest.fit(X, y)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]
for f in range(X.shape[1]):

    print( "%2d) %-*s %f" % (f + 1, 30,

                           feat_labels[indices[f]],

                           importances[indices[f]]))




from sklearn.preprocessing import StandardScaler

#train_test_split to split data

from sklearn.model_selection import train_test_split





from sklearn import metrics

from sklearn.metrics import accuracy_score

from statsmodels.stats.outliers_influence import variance_inflation_factor

import statsmodels.api as sm




df5 = df4.copy()


#Calculating the VIFs of columns
#Eliminating columns with VIF >6
#Running the model each time with reduced variables

#Iteration1

X = df5.drop(['y_yes'],axis = 1)


print(X)

y = df5['y_yes']

results = sm.OLS(y,X).fit()

summary = results.summary()

print(summary)

def calc_vif(X):

    vif = pd.DataFrame()

    vif['variables'] = X.columns

    vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]

    return(vif)

print(calc_vif(X))

#Drop housing_no , VIF = Inf



#Iteration 2
X = df5.drop(['y_yes','housing_no'],axis = 1)

print(X)

y = df5['y_yes']

results = sm.OLS(y,X).fit()

summary = results.summary()

print(summary)

def calc_vif(X):

    vif = pd.DataFrame()

    vif['variables'] = X.columns

    vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]

    return(vif)

print(calc_vif(X))
#Drop cons.conf.idx , VIF = 46.6



#Iteration3
X = df5.drop(['y_yes', 'cons.conf.idx','housing_no'],axis = 1)

print(X)

y = df5['y_yes']

results = sm.OLS(y,X).fit()

summary = results.summary()

print(summary)

def calc_vif(X):

    vif = pd.DataFrame()

    vif['variables'] = X.columns

    vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]

    return(vif)

print(calc_vif(X))
#Drop loan_no , VIF = 22.85



#Iteration4

X = df5.drop(['y_yes', 'cons.conf.idx','housing_no','loan_no'],axis = 1)

print(X)

y = df5['y_yes']

results = sm.OLS(y,X).fit()

summary = results.summary()

print(summary)

def calc_vif(X):

    vif = pd.DataFrame()

    vif['variables'] = X.columns

    vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]

    return(vif)

print(calc_vif(X))
#Drop poutcome_success , VIF = 11.11





#Iteration5
X = df5.drop(['y_yes', 'cons.conf.idx','housing_no','loan_no','poutcome_success'],axis = 1)

print(X)

y = df5['y_yes']

results = sm.OLS(y,X).fit()

summary = results.summary()

print(summary)

def calc_vif(X):

    vif = pd.DataFrame()

    vif['variables'] = X.columns

    vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]

    return(vif)

print(calc_vif(X))
#Drop marital , VIF = 9.28



#Iteration6
X = df5.drop(['y_yes', 'cons.conf.idx','housing_no','loan_no','poutcome_success','marital'],axis = 1)

print(X)

y = df5['y_yes']

results = sm.OLS(y,X).fit()

summary = results.summary()

print(summary)

def calc_vif(X):

    vif = pd.DataFrame()

    vif['variables'] = X.columns

    vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]

    return(vif)

print(calc_vif(X))
#Cannot drop age with VIF = 8.78 (>6)
#Age has a column contribution of 11% and is important variable



import seaborn as sns
corrmat = X.corr(method = 'spearman')
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);



#Split the data into training and test, 70:30
X_train, X_test, y_train, y_test = \
train_test_split(X,y, test_size = 0.3, random_state= 0)



#Standardizing data

sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.fit_transform(X_train)

X_test_std = sc.transform(X_test)

##################################################################################################################

#Model1 : logistic Regression

#LogisticRegression to run logistic regression model

print('\n\n\n logistic regression Output')

print('---------------------------------------------')

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train_std,y_train)

prediction = model.predict(X_test_std)



#Accuracy

accuracy = accuracy_score(y_test,prediction)

print('Accuracy %f' %accuracy)

#AUC

AUC = metrics.roc_auc_score(y_test,prediction)

print('AUC %f' %AUC)

#Confusion matrix

confusion_matrix = metrics.confusion_matrix(y_test,prediction)

print(confusion_matrix)

#Missclassified Samples

print('Misclassified samples: %d' %(y_test != prediction).sum())

#Classification_report

classification_report = metrics.classification_report(y_test,prediction)

print(classification_report)



#####################################################################################################################
#Model 2: Perceptron

print('\n\n\n Perceptron Output')

print('---------------------------------------------')

from sklearn.linear_model import Perceptron

ppn = Perceptron(max_iter = 40, eta0=0.1, random_state=0)

ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)



#Accuracy

print('Accuracy %f' %accuracy_score(y_test, y_pred))

#AUC

AUC = metrics.roc_auc_score(y_test,y_pred)
print('AUC %f' %AUC)



#confusion matrix

confusion_matrix = metrics.confusion_matrix(y_test,y_pred)

print(confusion_matrix)

#Missclassified Samples

print('Misclassified samples: %d' %(y_test != y_pred).sum())

#Classification Report

classification_report = metrics.classification_report(y_test,y_pred)

print(classification_report)


####################################################################################################################

#Model 3: SVC

print('\n\n\n SVC Output')

print('---------------------------------------------')

from sklearn.svm import SVC

svm = SVC(probability=True)

svm.fit(X_train_std,y_train)

y_pred = svm.predict(X_test_std)



predicted_probas = svm.predict_proba(X_test_std)


score = accuracy_score(y_test,y_pred)

#Accuracy

print('Accuracy %f' %score)

#89.38%

#AUC

AUC = metrics.roc_auc_score(y_test,y_pred)

print('AUC %f' %AUC)  #0.59



#confusion matrix

confusion_matrix = metrics.confusion_matrix(y_test,y_pred)

print(confusion_matrix)

#Missclassified Samples

print('Misclassified samples: %d' %(y_test != y_pred).sum())

#Classification Report

classification_report = metrics.classification_report(y_test,y_pred)

print(classification_report)




#Cumulative gain , best model

skplt.metrics.plot_cumulative_gain(y_test, predicted_probas)
plt.show()


#Lift Curve

skplt.metrics.plot_lift_curve(y_test,predicted_probas)
plt.show()

#ROC curve


#preds = predicted_probas[:,1]



#fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
#roc_auc = metrics.auc(fpr, tpr)

# method : plt
#import matplotlib.pyplot as plt
#plt.title('Receiver Operating Characteristic')
#plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
#plt.legend(loc = 'lower right')
#plt.plot([0, 1], [0, 1],'r--')
#plt.xlim([0, 1])
#plt.ylim([0, 1])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()







count = 0

for i in y_test:
    if i==1:
        count += 1
    else:
        continue
    
count


d1 = X_test['duration']
len(d1)
y_pred =pd.DataFrame(y_pred)
d2 = y_pred

d1.to_csv("d1.csv",index=False)
d2.to_csv("d2.csv",index=False)

d3 = X_test['age']
d3.to_csv("d3.csv",index=False)

d4 = X_test['job']
d4.to_csv("d4.csv",index=False)






len(d2)

submission = pd.concat([d1, d2], axis=1)

submission

submission.columns.values[1] = "y_yes"

submission.to_csv("submission.csv",index=False)


cols = [0,5]
X1 = X_test[X_test.columns[cols]]
X1
#X1 = sm.add_constant(X1)
#X1

#y_pred

y1 =  pd.DataFrame(y_pred, columns=['y_yes'])
y1
#y1 = sm.add_constant(y1)
#y1


d = pd.concat([X1, y1], axis=1)
d


###################################################################################################################

#Model 4: Naiive Bayes

print('\n\n\n Naiive Bayes Output')

print('---------------------------------------------')

from sklearn.naive_bayes import GaussianNB

naive_bayes = GaussianNB()

naive_bayes.fit(X_train_std,y_train)

y_pred = naive_bayes.predict(X_test_std)



score = accuracy_score(y_test,y_pred)

#Accuracy

print('Accuracy %f' %score)



#AUC

AUC = metrics.roc_auc_score(y_test,y_pred)

print('AUC %f' %AUC)  #0.59



#confusion matrix

confusion_matrix = metrics.confusion_matrix(y_test,y_pred)

print(confusion_matrix)

#Missclassified Samples

print('Misclassified samples: %d' %(y_test != y_pred).sum())

#Classification Report

classification_report = metrics.classification_report(y_test,y_pred)

print(classification_report)



################################################################################################################
#Model 5: Random Forest

print('\n\n\n Random Forest Output')

print('---------------------------------------------')

from sklearn.ensemble import RandomForestClassifier


forest = RandomForestClassifier(n_estimators = 10000, random_state = 0,
                                    n_jobs = -1)

forest.fit(X_train_std, y_train)

y_pred = forest.predict(X_test_std)



score = accuracy_score(y_test,y_pred)

#Accuracy

print('Accuracy %f' %score)



#AUC

AUC = metrics.roc_auc_score(y_test,y_pred)

print('AUC %f' %AUC)  #0.59



#confusion matrix

confusion_matrix = metrics.confusion_matrix(y_test,y_pred)

print(confusion_matrix)

#Missclassified Samples

print('Misclassified samples: %d' %(y_test != y_pred).sum())

#Classification Report

classification_report = metrics.classification_report(y_test,y_pred)

print(classification_report)



feat_labels = X_test.columns[0:15]

forest = RandomForestClassifier(n_estimators = 10000, random_state = 0, n_jobs = -1)
forest.fit(X_test_std, y_test)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]
for f in range(X.shape[1]):

    print( "%2d) %-*s %f" % (f + 1, 30,

                           feat_labels[indices[f]],

                           importances[indices[f]]))

##################################################################################################################



#Model  Gradient boosting

#from sklearn.ensemble import  GradientBoostingClassifier

#gb = GradientBoostingClassifier(n_estimators=5, random_state= 0)

#gb.fit(X_train_std, y_train)

#y_pred = gb.predict(X_test_std)

#y_pred
#score = accuracy_score(y_test,y_pred)

#Accuracy

#print('Accuracy %f' %score)



#AUC

#AUC = metrics.roc_auc_score(y_test,y_pred)

#print('AUC %f' %AUC)  #0.59



#confusion matrix

#confusion_matrix = metrics.confusion_matrix(y_test,y_pred)

#print(confusion_matrix)

#Missclassified Samples

#print('Misclassified samples: %d' %(y_test != y_pred).sum())

#Classification Report

#classification_report = metrics.classification_report(y_test,y_pred)

#print(classification_report)


##################################################################################################################

# Model 6: Neural network

print('\n\n\n Neural Network Output')

print('---------------------------------------------')

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(20,20,20),max_iter=5000,
                    random_state=0)

mlp.fit(X_train_std,y_train)


y_pred = (mlp.predict_proba(X_test_std)[:,1] >=.50).astype(bool)



score = accuracy_score(y_test,y_pred)

#Accuracy

print('Accuracy %f' %score)



#AUC

AUC = metrics.roc_auc_score(y_test,y_pred)

print('AUC %f' %AUC)  #0.59



#confusion matrix

confusion_matrix = metrics.confusion_matrix(y_test,y_pred)

print(confusion_matrix)

#Missclassified Samples

print('Misclassified samples: %d' %(y_test != y_pred).sum())

#Classification Report

classification_report = metrics.classification_report(y_test,y_pred)

print(classification_report)

######################################################################################################################





