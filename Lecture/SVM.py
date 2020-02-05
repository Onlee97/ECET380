# SVM
import time
import sklearn.datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = sklearn.datasets.make_moons(n_samples = 5000, noise = 0.15)
start_time = time.time()
polynomial_svm_clf = Pipeline([
		("poly_feature", PolynomialFeatures(degree=15)),
		("scaler", StandardScaler()),
		("svm_clf", SVC(kernel="linear", C= 10, max_iter = 1000000))

	])

polynomial_svm_clf.fit(X, y)
print("---%s seconds ---" % (time.time() - start_time))
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Ensemble Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

voting_clf = VotingClassifier(
	estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
	voting='hard')
voting_clf.fit(xTrain, yTrain)

from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
	clf.fit(xTrain, yTrain)
	y_pred = clf.predict(xTest)
	print(clf.__class__.__name__, accuracy_score(yTest, y_pred))

# Bagging
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
	DecisionTreeClassifier(), n_estimators=500, max_samples=100, 
	bootstrap=True, n_jobs=-1)

bag_clf.fit(xTrain, yTrain)
y_pred = bag_clf.predict(xTest)
print(bag_clf.__class__.__name__, accuracy_score(yTest, y_pred))

# RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(xTrain, yTrain)
y_pred = rnd_clf.predict(xTest)
print(rnd_clf.__class__.__name__, accuracy_score(yTest, y_pred))

# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(
	DecisionTreeClassifier(max_depth=1), n_estimators=200,
	algorithm="SAMME.R", learning_rate=0.5)

ada_clf.fit(xTrain, yTrain)
y_pred = ada_clf.predict(xTest)
print(ada_clf.__class__.__name__, accuracy_score(yTest, y_pred))


