"""
This is the main function
"""

import pandas as pd
from typing import Callable


def one_out(train: pd.DataFrame,
            target: pd.DataFrame,
            model_train: Callable[[pd.DataFrame, pd.DataFrame], float]
            ):
    """
    This function will itienary remove one features from train features and return dictionary that include it's performance
    :param train: pandas data frame, include all features
    :param target: padas data frame, include the target we want to predict
    :param model_train: function, will take the train and target and return a metric based on those features
    :return: base_performance: float, the basic performance that include all features in the machine learning model
             one_out_feature_performance:  dictionary, will include the performance after the feature has been removed
    """
    x_train = train.copy()

    one_out_feature_performance = {}
    base_performance = model_train(x_train, target)

    for feature in x_train.columns:
        temp_train = x_train.drop(columns=feature)
        result = model_train(temp_train, target)

        one_out_feature_performance[feature] = result

    return base_performance, one_out_feature_performance



def one_in( train: pd.DataFrame,
            target: pd.DataFrame,
            dropped_features: pd.DataFrame,
            model_train: Callable[[pd.DataFrame, pd.DataFrame], float]
            ):
    """
    This function will itienary add one features from train features and return dictionary that include it's performance
    :param train: pandas data frame, include remaining features
    :param target: padas data frame, include the target we want to predict
    :param droped_features: pandas data frame, include dropped columns from one out function
    :param model_train: function, will take the train and target and return a metric based on those features
    :return: dictionary, will include the performance after the feature has been removed
    """
    x_train = train.copy()
    one_in_feature_performance = {}
    base_performance = model_train(x_train, target)

    for feature in dropped_features.columns:
        temp_train = x_train.copy()
        temp_train[feature] = dropped_features[feature]
        result = model_train(temp_train, target)

        one_in_feature_performance[feature] = result

    return base_performance, one_in_feature_performance



def feature_filter(base_performance: float,
                   feature_performance: dict):
    """
    this function will filter out metrics that are lower than base metrics base on previous performance
    :param base_performance: dict,
    :param feature_performance: dict,
    :return: one_features: list, include all features which metric that are lower than base performance
    """
    one_features = []
    for feature, metrics in feature_performance.items():
        if metrics < base_performance:
            one_features += [feature]
    return one_features


def out_and_in(train: pd.DataFrame,
               target: pd.DataFrame,
               model_train: Callable[[pd.DataFrame, pd.DataFrame], float]
               ):
    """
    this function have two steps,
    the first step will remove one feature in the features and dropped those will increase performance,
    the second step will add them back one by one and add them back for those features will increase performance
    :param train: pandas data frame, include remaining features
    :param target: padas data frame, include the target we want to predict
    :param model_train: function, will take the train and target and return a metric based on those features
    :return: list,
    """

    base_out_performance, one_out_feature_performance = one_out(train, "content")
    out_features = feature_filter(base_out_performance, one_out_feature_performance)
    train_droped = train.drop(columns=out_features)
    out_columns = train[out_features]

    base_in_performance, one_in_feature_performance = one_in(train_droped, "content", out_columns)
    in_features = feature_filter(base_in_performance, one_in_feature_performance)

    
