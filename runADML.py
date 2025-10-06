# -*- coding: utf-8 -*-
"""
Author: Kiret Dhindsa
Contact: kiretd@gmail.com or dhindsaj@charite.de
Last Content Modification: September 15, 2020
Last Update (clean-up for publication to Ebrains): October 2, 2025

Cleaned code to run nested cross-validation experiment from the published paper
(Triebkorn et al. 2022). 

If you use this code, please cite the paper and the Github Repository at
[LINK]

Triebkorn, Paul, et al. "Brain simulation augments machine‐learning–based classification 
of dementia." Alzheimer's & Dementia: Translational Research & Clinical Interventions 
8.1 (2022): e12303.

GITHUB REPO

"""


# %% Import Libraries
# basic dataset libraries
import numpy as np
import pandas as pd
import scipy
import scipy.io as sio

# machine learning libraries
from sklearn.feature_selection import SelectKBest, SelectFromModel, f_classif
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import metrics, svm, base

# Statistics
from scipy.stats import ttest_rel, shapiro, mannwhitneyu, wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

# plotting libraries
import matplotlib.pyplot as plt

# util libraries
import time

np.random.seed(123)
USE_GI = False # Flag for using graph index features
USE_REG_MEANS = False # flag for using abeta and tau region-wise means vs. all regions

# %% Load Data
def get_data():
    '''
    Extract and organize data from file.
    '''
    filedata = sio.loadmat('Feature_matrices.mat')

    # Correct volumes for subject 4
    corrected_volumes = sio.loadmat('corrected_volumes.mat')
    filedata['Volumes'] = corrected_volumes['Volumes']
    
    sim_varnames = ['Bifurcations','Capacity33','Mean_real_freq_global',
                    'Mean_unreal_freq_global','High_d33','Low_d33','Freq_reg']
    
    # all but Freq_reg should just include the last column
    for varname in sim_varnames[:-1]:
        filedata[varname] = filedata[varname][:,2]

    emp_varnames = ['Tau_reg','tPET','PET','Abeta_reg','Volumes']
    
    FRA = filedata.pop('Freq_reg_allgc')
    MMSE = filedata.pop('MMSE')
    Volume_names = filedata.pop('Volume_names')
    
    Y = filedata.pop('Gnew').flatten()
    
    filedata.pop('__globals__')
    filedata.pop('__header__')
    filedata.pop('__version__')


    return (filedata, Y, sim_varnames, emp_varnames, FRA, MMSE, Volume_names)

(Xdict, Y, sim_varnames, emp_varnames, FRA, MMSE, Volume_names) = get_data()

Nclass = np.unique(Y).shape[0]
Nsubj = Y.shape[0]

# correct from some vectors having dim==1, when sklearn needs dim==2
for key in Xdict.keys():
    if len(Xdict[key].shape) == 1:
        Xdict[key] = np.expand_dims(Xdict[key],axis=1)
        
Ystr = [['HC','MCI','AD'][c-1] for c in Y]

# lists of simulated features to exclude
if USE_GI:
    bad_sim = ['Bifurcations','Mean_real_freq_global','Mean_unreal_freq_global']
else:
    bad_sim = ['Bifurcations','Mean_real_freq_global','Mean_unreal_freq_global',
               'High_d33','Low_d33','Capacity33']

good_sim = list(set(sim_varnames[:-1]).symmetric_difference(bad_sim))

# Feature information / names
flabels = sio.loadmat('Regions+degree.mat')
degrees = np.array(flabels['degree'])
reg = sio.loadmat('regionnames.mat')
reg = reg['regions']
regions = [n[0][0] for n in reg]
volnames = [n[0][0] for n in Volume_names]


# %% Classification Functions
def clean_features(X, featnames=None):
    '''
    Removes deficient columns from feature matrices
    '''
    _, mc = scipy.stats.mode(X, axis=0)
    rmidx = np.where(np.std(X,axis=0)==0)[0]
    rmidx = np.append(rmidx, np.where(mc[0]>10)[0], axis=0)
    rmidx = np.unique(rmidx)
    
    X = np.delete(X, rmidx, axis=1)    
    if featnames is not None:
        trimmednames = [f for i,f in enumerate(featnames) if i not in rmidx]
        return X, trimmednames
    else:
        return X
    
def getFeatureMatrix(Xdict, varnames):
    '''
    Constructs a feature matrix from Xdict using the variable names in varnames
    '''
    X = np.concatenate([Xdict[features] for features in varnames], axis=1) 
    Xclean = clean_features(X)
    
    if Xclean.shape[1] > 0:
        X = Xclean
    if X.shape[1] == 0:
        scores = [np.nan, np.nan]
        est = None
        return scores, est
    return Xclean

def nestedCV(Xdict, Y, varnames):
    '''
    Performs nested cross-validation for simultaneous model selection and 
    feature selection using SVM with Random Forest for feature selection.
    '''
    # Get the faeture matrix according to inputted variable names
    if USE_REG_MEANS:
        X = getFeatureMatrix(Xdict, varnames, opt='abeta_tau_reg_means')
    else:
        X = getFeatureMatrix(Xdict, varnames)

    idx = np.arange(0, X.shape[0], 1)
    
    # Define outputs
    SC = [] # Outer classifier scores
    FI = [] # Feature Importances
    FX = [] # Index of top features
    CM = [] # Test set confusion matrices
    YH = [] # Predicted Class Labels
    YT = [] # True Class Labels
    TI = [] # Test index
    
    # Set up components of the ML pipeline
    scoring = 'f1_weighted' #'accuracy'
    
    ssc = RobustScaler()
    clf_inner = RFC(random_state=999)
    clf_outer = svm.SVC(random_state=999)
    
    cv_inner = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=777)
    cv_outer = StratifiedShuffleSplit(n_splits=100, test_size=0.25, random_state=888)
    
    for train_outer, test_outer in cv_outer.split(X, Y):
        Xtrain, Xtest = X[train_outer], X[test_outer]
        Ytrain, Ytest = Y[train_outer], Y[test_outer]
        
        # Set up Inner CV loop using a grid search CV
        pipeline_inner = Pipeline(memory=None,
                                  steps=[('Scaler', ssc),
                                         ('Classifier', clf_inner)])
        
        inner_params = [{'Classifier__class_weight': ['balanced'],
                         'Classifier__criterion': ['entropy'], # gini or entropy
                         'Classifier__n_estimators': [10, 50],
                         'Classifier__max_depth': [None],
                         'Classifier__min_samples_split': [3, 4],
                         'Classifier__min_samples_leaf': [2, 3],
                         'Classifier__max_features': ['sqrt'],#,'log2',None,
                         }]        
    
        # Run Inner CV loop using a grid search CV
        grid_inner = GridSearchCV(pipeline_inner, inner_params, cv=cv_inner, 
                                  scoring=scoring, verbose=False, n_jobs=-1)
        grid_inner.fit(Xtrain, Ytrain)
    
        # Get best model and its feature importances (GINI index) - try entropy instead
        for obj in grid_inner.best_estimator_.steps:
#            print(obj[1])
            if obj[0] is 'Scaler':
                ssc_inner_best = base.clone(obj[1], safe=True)
            elif obj[0] is 'Feature_sel':
                sel_inner_best = base.clone(obj[1], safe=True)
            elif obj[0] is 'Classifier':
                clf_inner_best = base.clone(obj[1], safe=True)
                fi = obj[1].feature_importances_

        # Get top features
        nfeat = np.sum(fi>0)
        fidx = np.argsort(fi)[:-nfeat-1:-1]
        FI.append(fi)
        FX.append(fidx)
        
        # Set up outer CV lop
        pipeline_outer = Pipeline(memory=None,
                                  steps=[('Scaler', ssc_inner_best),
                                         ('Classifier', clf_outer)])
        
        outer_params = [{'Classifier__kernel': ['rbf'],
                         'Classifier__degree': [2],
                         'Classifier__gamma': [1e-2, 1e-1, 1],
                         'Classifier__C': [0.01, 0.1, 1, 10, 100]},
                        {'Classifier__kernel':['poly'],
                         'Classifier__degree': [2,3],
                         'Classifier__gamma': ['scale'],
                         'Classifier__C': [0.01, 0.1, 1, 10, 100]}]
        
        # set test set manually
        outer_split = -np.ones((len(idx),))
        outer_split[test_outer] = 0
        
        # Run outer CV loop
        grid_outer = GridSearchCV(pipeline_outer, outer_params, cv=PredefinedSplit(outer_split), 
                                  scoring=scoring, verbose=False, n_jobs=-1)#, refit=False)
        
        grid_outer.fit(X[:,fidx], Y)
        SC.append(grid_outer.best_score_)
        
        # return predicted
        for obj in grid_outer.best_estimator_.steps:
            if obj[0] is 'Scaler':
                ssc_outer_best = base.clone(obj[1], safe=True)
            elif obj[0] is 'Feature_sel':
                sel_outer_best = base.clone(obj[1], safe=True)
            elif obj[0] is 'Classifier':
                clf_outer_best = base.clone(obj[1], safe=True)
                
        
        pipeline_opt = Pipeline(memory=None,
                                steps=[('Scaler', ssc_outer_best),
                                       ('Classifier', clf_outer_best)])
            
        pipeline_opt.fit(Xtrain[:,fidx], Ytrain)
        yhat = pipeline_opt.predict(Xtest[:,fidx])

        YH.append(yhat)        
        CM.append(metrics.confusion_matrix(Ytest-1, yhat-1))#, ['HC','MCI','AD']))  
        YT.append(Ytest)
        TI.append(test_outer)
        
        print(grid_outer.best_score_)
    return SC, FI, FX, CM, YH, YT, TI 

def NestedCV_SVM__(X, Y, varnames):
    '''
    Performs nested cross-validation for simultaneous model selection and 
    feature selection using SVM with univariate feature selection.
        - sel_method: feature selection method ('f_classif','lda')
    '''
        # Get the faeture matrix according to inputted variable names
    if USE_REG_MEANS:
        X = getFeatureMatrix(Xdict, varnames, opt='abeta_tau_reg_means')
    else:
        X = getFeatureMatrix(Xdict, varnames)
        
    idx = np.arange(0, X.shape[0], 1)
    
    # Define outputs
    SC = [] # Outer classifier scores
    FI = [] # Feature Importances
    FX = [] # Index of top features
    YH = [] # Predicted Class Labels
    
    # Set up components of the ML pipeline
    scoring = 'f1_weighted' #'accuracy'
    
    ssc = RobustScaler()
    sel = SelectKBest(f_classif)
    clf_inner = svm.SVC(random_state=999)
    clf_outer = svm.SVC(random_state=999)
    
    cv_inner = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=777)
    cv_outer = StratifiedShuffleSplit(n_splits=100, test_size=0.25, random_state=111)
    
    for train_outer, test_outer in cv_outer.split(X, Y):
        Xtrain, Xtest = X[train_outer], X[test_outer]
        Ytrain, Ytest = Y[train_outer], Y[test_outer]
        
        # Set up Inner CV loop using a grid search CV
        pipeline_inner = Pipeline(memory=None,
                                  steps=[('Scaler', ssc),
                                         ('Feature_sel', sel),
                                         ('Classifier', clf_inner)])
        
        inner_params = [{'Feature_sel__k': [5, 10, 15, 20, 30, 40]}]#,
#                        {'Feature_sel__k': [5, 10, 15, 20, 30, 40]}]   
    
        # Run Inner CV loop using a grid search CV
        grid_inner = GridSearchCV(pipeline_inner, inner_params, cv=cv_inner, 
                                  scoring=scoring, verbose=True, n_jobs=8)
        grid_inner.fit(Xtrain, Ytrain)
    
        # Get best model and its feature importances (GINI index) - try entropy instead
        for obj in grid_inner.best_estimator_.steps:
            if obj[0] is 'Scaler':
                ssc_inner_best = base.clone(obj[1], safe=True)
            elif obj[0] is 'Feature_sel':
                sel_inner_best = base.clone(obj[1], safe=True)
                fi = obj[1].scores_
            elif obj[0] is 'Classifier':
                clf_inner_best = base.clone(obj[1], safe=True)

        # Get top features
        nfeat = sel_inner_best.k
        fidx = np.argsort(fi)[:-nfeat-1:-1]
        FI.append(fi)
        FX.append(fidx)
        
        # Set up outer CV lop
        pipeline_outer = Pipeline(memory=None,
                                  steps=[('Scaler', ssc_inner_best),
                                         ('Classifier', clf_outer)])
        
        outer_params = [{'Classifier__kernel': ['rbf'],
                         'Classifier__degree': [2],
                         'Classifier__gamma': [1e-3, 1e-2, 1e-1, 1],
                         'Classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
                        {'Classifier__kernel':['poly'],
                         'Classifier__degree': [2,3,4],
                         'Classifier__gamma': ['scale'],
                         'Classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
        
        # set test set manually
        outer_split = -np.ones((len(idx),))
        outer_split[test_outer] = 0
        
        # Run outer CV loop
        grid_outer = GridSearchCV(pipeline_outer, outer_params, cv=PredefinedSplit(outer_split), 
                                  scoring=scoring, verbose=True, n_jobs=8)#, refit=False)
        
        grid_outer.fit(X[:,fidx], Y)
        SC.append(grid_outer.best_score_)
        
        # return predicted
        for obj in grid_outer.best_estimator_.steps:
            if obj[0] is 'Scaler':
                ssc_outer_best = base.clone(obj[1], safe=True)
            elif obj[0] is 'Feature_sel':
                sel_outer_best = base.clone(obj[1], safe=True)
            elif obj[0] is 'Classifier':
                clf_outer_best = base.clone(obj[1], safe=True)
                
        
        pipeline_opt = Pipeline(memory=None,
                                steps=[('Scaler', ssc_outer_best),
                                       ('Classifier', clf_outer_best)])
            
        pipeline_opt.fit(Xtrain[:,fidx], Ytrain)
        
        yhat = pipeline_opt.predict(Xtest[:,fidx])        
        YH.append(metrics.confusion_matrix(Ytest-1, yhat-1))#, ['HC','MCI','AD']))
                
        print(grid_outer.best_score_)
    return SC, FI, FX, YH


def NestedCV_RFC___(X, Y, varnames):
    '''
    Performs nested cross-validation for simultaneous model selection and 
    feature selection using RFC with univariate feature selection.
    '''
        # Get the faeture matrix according to inputted variable names
    if USE_REG_MEANS:
        X = getFeatureMatrix(Xdict, varnames, opt='abeta_tau_reg_means')
    else:
        X = getFeatureMatrix(Xdict, varnames)
        
    idx = np.arange(0, X.shape[0], 1)
    
    # Define outputs
    SC = [] # Outer classifier scores
    FI = [] # Feature Importances
    FX = [] # Index of top features
    YH = [] # Predicted Class Labels
    
    # Set up components of the ML pipeline
    scoring = 'f1_weighted' #'accuracy'
    
    ssc = RobustScaler()
    clf_inner = RFC(random_state=999)
    clf_outer = RFC(random_state=999)
    
    cv_inner = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=777)
    cv_outer = StratifiedShuffleSplit(n_splits=100, test_size=0.25, random_state=111)
    
    for train_outer, test_outer in cv_outer.split(X, Y):
        Xtrain, Xtest = X[train_outer], X[test_outer]
        Ytrain, Ytest = Y[train_outer], Y[test_outer]
        
        # Set up Inner CV loop using a grid search CV
        pipeline_inner = Pipeline(memory=None,
                                  steps=[('Scaler', ssc),
                                         ('Classifier', clf_inner)])
        
        inner_params = [{'Classifier__class_weight': ['balanced'],
                         'Classifier__criterion': ['gini', 'entropy'],
                         'Classifier__n_estimators': [10, 50, 100, 200],
                         'Classifier__max_depth': [None],
                         'Classifier__min_samples_split': [2, 3, 4, 5],
                         'Classifier__min_samples_leaf': [1, 2, 3],
                         'Classifier__max_features': ['sqrt','log2',None],
                         'Classifier__oob_score': [True],
                         }]
    
        # Run Inner CV loop using a grid search CV
        grid_inner = GridSearchCV(pipeline_inner, inner_params, cv=cv_inner, 
                                  scoring=scoring, verbose=True, n_jobs=8)
        grid_inner.fit(Xtrain, Ytrain)
    
        # Get best model and its feature importances (GINI index) - try entropy instead
        for obj in grid_inner.best_estimator_.steps:
#            print(obj[1])
            if obj[0] is 'Scaler':
                ssc_inner_best = base.clone(obj[1], safe=True)
            elif obj[0] is 'Feature_sel':
                sel_inner_best = base.clone(obj[1], safe=True)
            elif obj[0] is 'Classifier':
                clf_inner_best = base.clone(obj[1], safe=True)
                fi = obj[1].feature_importances_

        # Get top features
        nfeat = np.sum(fi>0)
        fidx = np.argsort(fi)[:-nfeat-1:-1]
        FI.append(fi)
        FX.append(fidx)
        
        # Set up outer CV lop
        pipeline_outer = Pipeline(memory=None,
                                  steps=[('Scaler', ssc_inner_best),
#                                         ('Feature_sel', sel_inner_best)
                                         ('Classifier', clf_inner_best)])
        
        outer_params = [{'Classifier__class_weight': ['balanced'],
                        'Classifier__criterion': ['gini', 'entropy'],
                        'Classifier__n_estimators': [10, 50, 100, 200],
                        'Classifier__max_depth': [None],
                        'Classifier__min_samples_split': [2, 3, 4, 5],
                        'Classifier__min_samples_leaf': [1, 2, 3],
                        'Classifier__max_features': [None],
                        'Classifier__oob_score': [True, False],
                         }]
        
        # set test set manually
        outer_split = -np.ones((len(idx),))
        outer_split[test_outer] = 0
        
        # Run outer CV loop
        grid_outer = GridSearchCV(pipeline_outer, outer_params, cv=PredefinedSplit(outer_split), 
                                  scoring=scoring, verbose=True, n_jobs=8)#, refit=False)
        
        grid_outer.fit(X[:,fidx], Y)
        SC.append(grid_outer.best_score_)
        
        # return predicted
        for obj in grid_outer.best_estimator_.steps:
            if obj[0] is 'Scaler':
                ssc_outer_best = base.clone(obj[1], safe=True)
            elif obj[0] is 'Feature_sel':
                sel_outer_best = base.clone(obj[1], safe=True)
            elif obj[0] is 'Classifier':
                clf_outer_best = base.clone(obj[1], safe=True)
                
        
        pipeline_opt = Pipeline(memory=None,
                                steps=[('Scaler', ssc_outer_best),
#                                       ('Feature_sel', sel_outer_best),
                                       ('Classifier', clf_outer_best)])
            
        pipeline_opt.fit(Xtrain[:,fidx], Ytrain)
        yhat = pipeline_opt.predict(Xtest[:,fidx])
        YH.append(metrics.confusion_matrix(Ytest-1, yhat-1))#, ['HC','MCI','AD']))
                
        print(grid_outer.best_score_)
        print(clf_inner_best)
        print(clf_outer_best)
    return SC, FI, FX, YH

# %% Run Classification Experiment
CLASSIFIER = 'NEST' # 'SVM', 'RFC'
DIM_REDUCTION = 'f_classif' # 'f_classif','pca', 'lda', 'l1', 'rfc'
if CLASSIFIER == 'SVM':
    sf1, FI1, FX1, YH1 = NestedCV_SVM__(Xdict, Y, emp_varnames)
    sf2, FI2, FX2, YH2 = NestedCV_SVM__(Xdict, Y, good_sim + [sim_varnames[-1]])
    sf3, FI3, FX3, YH3 = NestedCV_SVM__(Xdict, Y, emp_varnames + good_sim + [sim_varnames[-1]])
elif CLASSIFIER == 'RFC':
    sf1, FI1, FX1, YH1 = NestedCV_RFC___(Xdict, Y, emp_varnames)
    sf2, FI2, FX2, YH2 = NestedCV_RFC___(Xdict, Y, good_sim + [sim_varnames[-1]])
    sf3, FI3, FX3, YH3 = NestedCV_RFC___(Xdict, Y, emp_varnames + good_sim + [sim_varnames[-1]])
elif CLASSIFIER == 'NEST':
    sf1, FI1, FX1, CM1, YH1, YT1, TI1 = nestedCV(Xdict, Y, emp_varnames)
    sf2, FI2, FX2, CM2, YH2, YT2, TI2 = nestedCV(Xdict, Y, good_sim + [sim_varnames[-1]])
    sf3, FI3, FX3, CM3, YH3, YT3, TI3 = nestedCV(Xdict, Y, emp_varnames + good_sim + [sim_varnames[-1]])

# Save and load results from experiment
def saveResults():
    '''
    Saves classification results to .npz
    '''
    timestr = time.strftime('%Y-%m-%d-%H-%M')
    savefolder = 'Results/'
    gi = 'withGI' if USE_GI else 'noGI'
    savename = 'Results_' + CLASSIFIER + '_' + DIM_REDUCTION + '_' + gi + '_' + timestr

    np.savez(savefolder + savename + '.npz',
             F1_emp=sf1, FI_emp=FI1, FX_emp=FX1, YH_emp=YH1, YT_emp=YT1, TI_emp=TI1,
             F1_sim=sf2, FI_sim=FI2, FX_sim=FX2, YH_sim=YH2, YT_sim=YT2, TI_sim=TI2,
             F1_com=sf3, FI_com=FI3, FX_com=FX3, YH_com=YH3, YT_com=YT3, TI_com=TI3)
	
    r = np.load(savefolder + savename + '.npz', allow_pickle=True)
    sio.savemat(savefolder + savename + '.mat', mdict={key:r[key] for key in r.keys()})
    return None

def loadResults(fname):
    '''
    Loads a previously saved classification result from a .npz file.
    '''
    r = np.load('Results/'+fname)
    return r
    
# saveResults()


# %% Plot Results
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False):#, title=None):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    '''
    cmap=plt.cm.Blues
    
    if y_pred is None:
        if y_true.ndim == 2:
            cm = y_true
    elif y_pred.ndim == 1 and y_true.ndim == 1:
        # Compute confusion matrix
        cm = metrics.confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred).astype(int)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return None

def plotScores(scores, stype='f1'):
    '''
    Plot classification score
    '''
    # barplot with errorbars
    means = [np.mean(scores[i]) for i in range(len(scores))]
    sterr = [np.std(scores[i])/np.sqrt(len(scores[i])) for i in range(len(scores))]
    
    plt.figure()
    plt.bar(['Empirical','Simulated','Combined'], means, yerr=sterr)
    plt.ylim([0,1])
#    for tick in plt.gca().xaxis.get_major_ticks():
#        tick.label.set_fontsize(14)
#    
    # add text for each score
    for i, v in enumerate(scores):
        plt.text(i - 0.1, 0.4, '{:.{}f}'.format(np.mean(v),2),
                 size=12)
    
    # add title
    if stype == 'f1':
        plt.title('Weighted F1 score', fontsize=14)
    elif stype == 'acc':
        plt.title('Percent Accuracy', fontsize=14)
    
    savename = 'NestedCV_{}_Performance_{}_{}_2.png'.format(CLASSIFIER, 'noGI', stype)
#    plt.savefig(savename, bbox_inches='tight', pad_inches=0.1, dpi=400)
    return None

plot_confusion_matrix(np.sum(CM1,axis=0), None, normalize=True, classes=np.array(['HC','MCI','AD']))
plot_confusion_matrix(np.sum(CM2,axis=0), None, normalize=True, classes=np.array(['HC','MCI','AD']))
plot_confusion_matrix(np.sum(CM3,axis=0), None, normalize=True, classes=np.array(['HC','MCI','AD']))

acc1 = [np.trace(cm)/np.sum(cm) for cm in CM1]
acc2 = [np.trace(cm)/np.sum(cm) for cm in CM2]
acc3 = [np.trace(cm)/np.sum(cm) for cm in CM3]

plotScores([sf1,sf2,sf3], stype='f1')
plotScores([acc1,acc2,acc3], stype='acc')


# %% Feature Importance
def extendFeatureNames(varnames, regions, volnames):
    '''
    Extends feature names so that each column has its own name.
    '''
    names = []
    for var in varnames:
        n = Xdict[var].shape[1]
        if n == 1:
            names.append(var)
        elif var == 'Volumes':
            names.extend([var+'_'+volnames[i] for i in range(n)])
        else:         
            names.extend([var+'_'+regions[i] for i in range(n)])
    return names

def plot_FeatureFrequency_Top(Xdict, FI, FX, varnames, title=None):
    # Get feature names    
    X = np.concatenate([Xdict[features] for features in varnames], axis=1) 
    Xclean, featurenames = clean_features(X, extendFeatureNames(varnames, regions, volnames))    

    # Selected Feature Historgram
    fidx = [idx for cvrun in FX for idx in cvrun]
    tmp = [x[0][x[1]] for x in zip(FI,FX)]
    fimp = [imp for x in tmp for imp in x]
    fcount = np.bincount(fidx, minlength=len(FI[0]))[0:]
    
    n50 = np.sum(fcount>50)
    sortInd = fcount.argsort()[-n50:][::-1]
    fcount_sort = fcount[sortInd]
    
    fname_sort = [featurenames[i] if featurenames[i].startswith('Volume') else featurenames[i][:-4] for i in sortInd]
#    region_sort = [regions[i][:-4] for i in sortInd]

    fig, ax = plt.subplots()
    fig.set_size_inches(20,14)

    plt.bar(np.arange(0, len(fcount_sort)), fcount_sort)
    plt.xticks(range(len(fcount_sort)), fname_sort, rotation=90, fontsize=16)
#            plt.xlim([-1, n+1])
    plt.tick_params(axis='x',direction='in',pad=-350)
    plt.gcf().subplots_adjust(bottom=0.2)
    
    plt.title('Feature Selection Frequency ({})'.format(title), fontsize=14)
    plt.tight_layout()
    plt.savefig('Feature_Selection_Frequency_{}.png'.format(title), 
                bbox_inches='tight', pad_inches=0.1, dpi=400)

 
plot_FeatureFrequency_Top(Xdict, FI1, FX1, emp_varnames, title='Empirical') 
plot_FeatureFrequency_Top(Xdict, FI2, FX2, good_sim + [sim_varnames[-1]], title='Simulated')
plot_FeatureFrequency_Top(Xdict, FI3, FX3, emp_varnames + good_sim + [sim_varnames[-1]], title='Combined')

# --- if using results loaded from previously saved file ---
# r = loadResults(fname)
# plot_FeatureFrequency_Top(Xdict, r['FI_emp'], r['FX_emp'], emp_varnames, title='Empirical') 
# plot_FeatureFrequency_Top(Xdict, r['FI_sim'], r['FX_sim'], good_sim + [sim_varnames[-1]], title='Simulated')
# plot_FeatureFrequency_Top(Xdict, r['FI_com'], r['FX_com'], emp_varnames + good_sim + [sim_varnames[-1]], title='Combined')


# %% Save feature imortance stats to excel
def save_FeatureStats(Xdict, FI, FX, FSET):
    from openpyxl import load_workbook
    
    # Get feature names
    if FSET == 'emp':
        varnames = emp_varnames
        sheet_name = 'Empirical Feature Set'
    elif FSET == 'sim':
        varnames = good_sim + [sim_varnames[-1]]
        sheet_name = 'Simulated Feature Set'
    elif FSET == 'com':
        varnames = emp_varnames + good_sim + [sim_varnames[-1]]
        sheet_name = 'Combined Feature Set'
    
    X = np.concatenate([Xdict[features] for features in varnames], axis=1)
    Xclean, featurenames = clean_features(X, extendFeatureNames(varnames, regions, volnames))
    
    # Feature importance metrics
    fidx = [idx for cvrun in FX for idx in cvrun]
    fcount = np.bincount(fidx, minlength=len(FI[0]))[0:] # selection frequency
    fimp = np.mean(FI, axis=0) # entropy criterion
    
    # Save to excel    
    df = pd.DataFrame(np.stack([fcount, fimp], axis=1), index=featurenames,
                      columns=['Selection Frequency','Entropy Criterion'])
    
    book = load_workbook('Results/Feature_Importance_Full.xlsx')
    writer = pd.ExcelWriter('Results/Feature_Importance_Full.xlsx')
    writer.book = book
    
    df.to_excel(writer, sheet_name=sheet_name)
    writer.save()
    return None

# --- if using results loaded from previously saved file ---
# r = loadResults(fname)
# save_FeatureStats(Xdict, r['FI_emp'], r['FX_emp'], 'emp')
# save_FeatureStats(Xdict, r['FI_sim'], r['FX_sim'], 'sim')
# save_FeatureStats(Xdict, r['FI_com'], r['FX_com'], 'com')


# %% Some helpful statistical tests if needed
def mctest(y1, y2, ytrue):
    '''
    Performs McNemar's test to compare two classifiers.
    '''
    table = [[((y1==ytrue)&(y2==ytrue)).sum(), ((y1==ytrue)&(y2!=ytrue)).sum()],
              [((y1!=ytrue)&(y2==ytrue)).sum(), ((y1!=ytrue)&(y2!=ytrue)).sum()]]
    result = mcnemar(table, exact=True)
    print('McNemar Test Result: Chi-squared(1) = {}, p = {}'.format(result.statistic, result.pvalue))
    return None

def shapiro_wilk_test(f1_emp, f1_sim, f1_com):
    '''
    Peforms a shapiro wilk test of normality on performance results.
    '''
    s, p = shapiro(f1_emp)
    print('Shapiro Wilk test of F1 (Empirical): s = {:.2f}, p = {:.2f}'.format(s,p))
    
    s, p = shapiro(f1_sim)
    print('Shapiro Wilk test of F1 (Simulated): s = {:.2f}, p = {:.2f}'.format(s,p))
    
    s, p = shapiro(f1_com)
    print('Shapiro Wilk test of F1 (Combined): s = {:.2f}, p = {:.2f}'.format(s,p))
    return None
    
def ttest_compare(f1_emp, f1_sim, f1_com):
    '''
    Performs a t-test comparison of feature set performance.
    '''
    tstat, p = ttest_rel(f1_emp, f1_sim)
    print('Related-samples t-test of F1 scores [Empirical vs. Simulated]: t(98) = {}, p = {}'.format(tstat,p))
    
    tstat, p = ttest_rel(f1_emp, f1_com)
    print('Related-samples t-test of F1 scores [Empirical vs. Combined]: t(98) = {}, p = {}'.format(tstat,p))
    
    tstat, p = ttest_rel(f1_sim, f1_com)
    print('Related-samples t-test of F1 scores [Simulated vs. Combined]: t(98) = {}, p = {}'.format(tstat,p))
    return None

def utest(f1_emp, f1_sim, f1_com):
    '''
    Performs a Mann-Whitney U-test comparison of feature set performance.
    '''
    u, p = mannwhitneyu(f1_emp, f1_sim)
    print('Mann-Whitney U-test of F1 scores [Empirical vs. Simulated]: U(98) = {}, p = {}'.format(u,p))
    
    u, p = mannwhitneyu(f1_emp, f1_com)
    print('Mann-Whitney U-test of F1 scores [Empirical vs. Combined]: U(98) = {}, p = {}'.format(u,p))
    
    u, p = mannwhitneyu(f1_sim, f1_com)
    print('Mann-Whitney U-test of F1 scores [Simulated vs. Combined]: U(98) = {}, p = {}'.format(u,p))
    return None

def wsrtest(f1_emp, f1_sim, f1_com):
    '''
    Performs a Wilcoxon signed-rank test comparison of feature set performance.
    '''
    u, p = wilcoxon(f1_emp, f1_sim)
    print('Wilcoxon signed-rank test of F1 scores [Empirical vs. Simulated]: U(98) = {}, p = {}'.format(u,p))
    
    u, p = wilcoxon(f1_emp, f1_com)
    print('Wilcoxon signed-rank test of F1 scores [Empirical vs. Combined]: U(98) = {}, p = {}'.format(u,p))
    
    u, p = wilcoxon(f1_sim, f1_com)
    print('Wilcoxon signed-rank testt of F1 scores [Simulated vs. Combined]: U(98) = {}, p = {}'.format(u,p))
    return None

