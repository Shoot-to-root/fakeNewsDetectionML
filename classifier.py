# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 22:07:10 2018

@author: sone_e
"""

from sklearn.model_selection import cross_val_score 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.calibration import CalibratedClassifierCV
import plot, clustering
from sklearn.neural_network import MLPClassifier


# Multinomial Naives Bayes(MNB)
def multinomialNB(X_train, X_test, y_train, y_test):
    mnb = MultinomialNB()
    #clustering.gridSearch(X_train, y_train, mnb)
    cross_valid(mnb, X_train, y_train)
    #mnb.fit(X_train, y_train) #train training set
    y_pred = mnb.predict(X_test) #test model with X_test, store the result in y_pred
    
    print("The accuracy of Multinomial Naives Bayes classifier is: ", mnb.score(X_test, y_test))
    print(classification_report(y_test, y_pred))
    '''
    cm = metrics.confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
    plot.plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
    '''
    save_model(mnb)

# Decision Tree
def decisionTree(X_train, X_test, y_train, y_test):
    tree = DecisionTreeClassifier()
    #clustering.gridSearch(X_train, y_train, tree)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    
    print("The accuracy of Decision Tree classifier is: ", tree.score(X_test, y_test))
    print(classification_report(y_test, y_pred))

    save_model(tree)
    
# Linear support vector classifier
def svc(X_train, X_test, y_train, y_test):
    svc = LinearSVC()
    #svc.fit(X_train, y_train)
    clf = CalibratedClassifierCV(svc, cv=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("The accuracy of SVC classifier is: ", clf.score(X_test, y_test))
    print(classification_report(y_test, y_pred))
    
    save_model(clf)
    
# random forests
def randomForest(X_train, X_test, y_train, y_test):
    forest = RandomForestClassifier()
    #clustering.gridSearch(X_train, y_train, forest)
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test)
    
    print("The accuracy of Random Forest classifier is: ", forest.score(X_test, y_test))
    print(classification_report(y_test, y_pred))

    save_model(forest)
    
# logistic regression
def logisticReg(X_train, X_test, y_train, y_test):
    lr = LogisticRegression() 
    #clustering.gridSearch(X_train, y_train, lr)
    lr.fit(X_train, y_train) #train model with fit
    lr_y_predict = lr.predict(X_test) #use trained lr model to test X_test   
    
    print("The accuracy of LR classifier is: ", lr.score(X_test, y_test))
    print(classification_report(y_test, lr_y_predict))

    save_model(lr)

# SGDClassifer  
def sgdClassifier(X_train, X_test, y_train, y_test):
    sgdc = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5)  
    #clustering.gridSearch(X_train, y_train, sgdc)
    sgdc.fit(X_train, y_train)
    sgdc_y_predict = sgdc.predict(X_test)
    
    print("The accuracy of SGD classifier is: ", sgdc.score(X_test, y_test))
    print(classification_report(y_test, sgdc_y_predict))

    save_model(sgdc)
    
# Gradient Tree Boosting
def gradientBoost(X_train, X_test, y_train, y_test):
    gbc = GradientBoostingClassifier()
    #clustering.gridSearch(X_train, y_train, gbc)
    gbc.fit(X_train, y_train)
    y_pred = gbc.predict(X_test)
    
    print("The accuracy of Gradient Tree Boosting classifier is: ", gbc.score(X_test, y_test))
    print(classification_report(y_test, y_pred))

    save_model(gbc)

# MLP Classifier (Neural Network)
def mlpClassifier(X_train, X_test, y_train, y_test):
    mlp = MLPClassifier(hidden_layer_sizes=(200,200,200),max_iter=500, alpha=1e-5)
    #clustering.gridSearch(X_train, y_train, mlp)
    #cross_valid(mlp, X_train, y_train)
    mlp.fit(X_train,y_train)
    y_pred = mlp.predict(X_test)
    
    print("The accuracy of MLP classifier is: ", mlp.score(X_test, y_test))
    print(classification_report(y_test, y_pred))
    
    #save_model(mlp)
    
def save_model(classifier):
    joblib.dump(classifier, 'classifier.pkl')

        