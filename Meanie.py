import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline

train_data = pd.read_csv('OG/train.csv', delimiter=',')
test_data = pd.read_csv('OG/test.csv', delimiter = ',')
binary_columns = [c for c in train_data.columns if train_data[c].name.find('bin') != -1]
category_columns = [c for c in train_data.columns if train_data[c].name.find('cat') != -1]
numerical_columns = [c for c in train_data.columns if len(train_data[c].name) <= 10]
numerical_columns.remove('id')
del train_data['id']
cat_dum_columns = ['ps_ind_04_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_07_cat', 'ps_car_10_cat']
cat_norm_columns = ['ps_ind_02_cat', 'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_08_cat', 'ps_car_09_cat', 'ps_car_11_cat']
data_numerical = train_data[numerical_columns]
data_numerical = (data_numerical - data_numerical.mean()) / data_numerical.std()
data_cat_dum = pd.get_dummies(train_data[cat_dum_columns])

finaldata = pd.concat((data_numerical, train_data[binary_columns], train_data[cat_dum_columns], train_data[cat_dum_columns]), axis=1)

finaldata = pd.DataFrame(train_data, dtype=float)
y = finaldata['target']
X = finaldata.drop(('target'), axis=1)
feature_names = X.columns

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 11)

N_train, _ = X_train.shape
N_test,  _ = X_test.shape

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_train_predict = knn.predict(X_train)
y_test_predict = knn.predict(X_test)

err_train = np.mean(y_train != y_train_predict)
err_test  = np.mean(y_test  != y_test_predict)
print (err_train, err_test)

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
clf = QuadraticDiscriminantAnalysis()
clf.fit(X_train, y_train)

err_train = np.mean(y_train != clf.predict(X_train))
err_test  = np.mean(y_test  != clf.predict(X_test))
print (err_train, err_test)

from sklearn import ensemble
rf = ensemble.RandomForestClassifier(n_estimators=100, random_state=11)
rf.fit(X_train, y_train)

err_train = np.mean(y_train != rf.predict(X_train))
err_test  = np.mean(y_test  != rf.predict(X_test))
print (err_train, err_test)

d_first = 50
plt.figure(figsize=(8, 8))
plt.title("Feature importances")
plt.bar(range(d_first), importances[indices[:d_first]], align='center')
plt.xticks(range(d_first), np.array(feature_names)[indices[:d_first]], rotation=90)
plt.xlim([-1, d_first]);

best_features = indices[:40]
best_features_names = feature_names[best_features]

bt = ensemble.GradientBoostingClassifier(n_estimators=100, random_state=11)
gbt.fit(X_train[best_features_names], y_train)

err_train = np.mean(y_train != gbt.predict(X_train[best_features_names]))
err_test = np.mean(y_test != gbt.predict(X_test[best_features_names]))
print (err_train, err_test)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l2', solver='newton-cg', random_state=None)
clf.fit(X_train, y_train)

err_train = np.mean(y_train != clf.predict(X_train))
err_test  = np.mean(y_test  != clf.predict(X_test))
print (err_train, err_test)
