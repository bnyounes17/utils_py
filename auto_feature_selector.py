#############################################################################
# The purpose of the function is to return the best features according 
# to the selected feature selector function
#
# The selectors suggested here are:
# - Pearson Correalation selector
# - Chi squared selector
# - Recursive Feature Elimination (RFE) selector
# - Logistic Regression selector
# - Random Forest Classifier selector
# - Light Gradient-Boosting Machine (LGBM) classifier selector
#
# Inputs: - X: Features
#         - y: Target
#         - methods: List of features' selectors
#         - max_number_of_features: Maximum number of features to display
#         - num_feats: The number of feature to select 
# Output: List of features
# Other selectors will be added in the future to the list
############################################################################

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


def cor_selector(X : pd.DataFrame, y : pd.Series, num_feats : int):
    
    cor_list = []
    feature_name = X.columns.tolist()
    for i in feature_name:
        print(X[i], y)
        cor_list.append(np.corrcoef(X[i], y)[0, 1])
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    cor_feature = X.iloc[:, np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    cor_support = [True if i in cor_feature else False for i in feature_name]
    
    return cor_support, cor_feature

def chi_squared_selector(X : pd.DataFrame, y : pd.Series, num_feats : int):
    
    X_norm = MinMaxScaler().fit_transform(X)
    chi_support = SelectKBest(chi2, k=num_feats).fit(X_norm, y).get_support()
    chi_feature = X.loc[:, chi_support].columns.tolist()
    
    return chi_support, chi_feature

def rfe_selector(X : pd.DataFrame, y : pd.Series, num_feats : int):
    
    rfe_selector = RFE(estimator=LogisticRegression(),
                       n_features_to_select=num_feats,
                       step=10,
                       verbose=5)
    rfe_selector.fit(X, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:, rfe_support].columns.tolist()
    
    return rfe_support, rfe_feature

def embedded_log_reg_selector(X : pd.DataFrame, y : pd.Series, num_feats : int):
    
    embedded_lr_selector = SelectFromModel(LogisticRegression(penalty='l2'), max_features=num_feats)
    embedded_lr_selector.fit(X, y)
    embedded_lr_support = embedded_lr_selector.get_support()
    embedded_lr_feature = X.loc[:, embedded_lr_support].columns.tolist()
    
    return embedded_lr_support, embedded_lr_feature

def embedded_rf_selector(X : pd.DataFrame, y : pd.Series, num_feats : int):
    
    embedded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
    embedded_rf_selector.fit(X, y)
    embedded_rf_support = embedded_rf_selector.get_support()
    embedded_rf_feature = X.loc[:, embedded_rf_support].columns.tolist()
    
    return embedded_rf_support, embedded_rf_feature

def embedded_lgbm_selector(X : pd.DataFrame, y : pd.Series, num_feats : int):
    
    lgbc = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
                          reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)
    embedded_lgb_selector = SelectFromModel(lgbc, max_features=num_feats)
    embedded_lgb_selector.fit(X, y)
    embedded_lgbm_support = embedded_lgb_selector.get_support()
    embedded_lgbm_feature = X.loc[:, embedded_lgbm_support].columns.tolist()
    
    return embedded_lgbm_support, embedded_lgbm_feature

def autoFeatureSelector(X: pd.DataFrame, y: pd.Series, methods: list, max_number_of_features: int, num_feats: int):

    feature_list = []
    if 'pearson' in methods:
        cor_support, cor_feature = cor_selector(X, y,num_feats)
        feature_list += cor_feature
    if 'chi-square' in methods:
        chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
        feature_list += chi_feature
    if 'rfe' in methods:
        rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
        feature_list += rfe_feature
    if 'log-reg' in methods:
        embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
        feature_list += embedded_lr_feature
    if 'rf' in methods:
        embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
        feature_list += embedded_rf_feature
    if 'lgbm' in methods:
        embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
        feature_list += embedded_lgbm_feature
    
    best_features = pd.DataFrame({'features': feature_list}).value_counts().head(max_number_of_features)
    
    return best_features