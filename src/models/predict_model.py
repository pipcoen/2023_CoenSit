from sklearn.pipeline import Pipeline
from sklearn import svm 
from sklearn import dummy
import sklearn as skl
import sklearn.model_selection as sklselection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RepeatedKFold
import sklearn.linear_model as sklinear
from sklearn.utils import resample
import sklearn.metrics as sklmetrics
from tqdm import tqdm
import xarray as xr
import numpy as np
import pandas as pd
from collections import defaultdict
from joblib import Parallel, delayed
import random

# some sklearn models
from sklearn.svm import SVC
from src.models.kernel_regression import groupKFoldRandom

import scipy.stats as sstats

# general util
import itertools
import bunch
import pickle as pkl
import os
import glob
import pdb   # debugging


def make_windowed_feature_matrix(activity_dataset, window_start_loc=0, window_width=20,
                                 window_center_loc=None,
                                 activity_name='firing_rate', window_width_units='bins',
                                 window_type='forward'):
    """
    Generates a feature matrix based on the average activity of each cell across a time window
    that is aligned to some stimulus or behaviour (eg. stimulus onset)
    Arguments 
    activity_dataset : (xarray dataset)
        xarray object with dimensions: Trial, Cell, and Time
    window_start_loc : (int)
        when the window should start (time bin) (for taking the mean)
    window_width     : (int)
        the length (in time bin units) of the window
    activity_name    : (str)
        which variable to obtain from the activity_dataset
        (eg. 'firing_rate' or 'spike_count')
    window_type : (str)
        type of window to make around each window location.
        option 1: 'forward', each window starts at window_loc and ends at window_loc + window_width
        option 2: 'center', each window is centered around window_loc with two sides adding up to window_width
    Output
    --------
    feature_matrix (numpy ndarray)
        feature matrix with shape (num_trial, num_neurons)
    """

    if window_type == 'forward':
        window_end_loc = window_start_loc + window_width
    elif window_type == 'center':
        # window_start_loc = window_start_loc - (window_width / 2)
        # window_end_loc = window_start_loc + window_width
        if window_width_units == 'bins':
            assert window_width % 2 == 0, print('Window width needs to be even for centered windows with bin units')
        window_start_loc = window_center_loc - (window_width / 2)
        window_end_loc = window_center_loc + (window_width / 2)
    else:
        print('Warning: no valid window type specified.')

    if window_width_units == 'bins':
        mean_rate_feature_matrix = activity_dataset.sel(Time=slice(window_start_loc,
                                                 window_end_loc)).mean(dim='Time')
    else:
        print('Implement provision of window width in seconds.')

    if mean_rate_feature_matrix['Cell'].size > 1:
        mean_rate_feature_matrix = mean_rate_feature_matrix.transpose('Trial', 'Cell')
        feature_matrix = mean_rate_feature_matrix[activity_name].values
    else:
        feature_matrix = mean_rate_feature_matrix[activity_name].values
        feature_matrix = feature_matrix.reshape(-1, 1)

    return feature_matrix


def make_labels(behaviour_df, event='left_right'):
    """
    Obtain labels for doing classification.
    Arugments 
    behaviour_df   : dataframe containing trial by trial behaviour and stimulus information 
    event          : which event to classifiy 
    TODO: have support for modality and no-goes as well 
    """

    if event == 'left_right':
        assert len(behaviour_df) == np.sum(behaviour_df['goLeft'].astype(int)) + \
                            np.sum(behaviour_df['goRight'].astype(int)), print('Left and Right and not the'
                                                                               'only events')
        label = behaviour_df['goRight'].astype(int)
        # 0 : left
        # 1 : right
    elif event == 'responseMade':
        # directly uses the responseMade column
        # 0: no-go
        # 1: left
        # 2: right
        label = behaviour_df['responseMade'].astype(int)

    else:
        print('Warning: no valid event name specified.')


    return label.values


def stratify_from_target_y(y_target, y, X=None, verbose=False):
    """
    Subsample labels y such that it has the same class distributions as target_y.
    If the feature matrix X is provided, will also subsample X. (assumes X and y are ordered)
    :param target_y:
    :param y:
    :param X:
    :return:
    """

    if type(y_target) is list:
        y_target = np.array(y_target)
    if type(y) is list:
        y = np.array(y)

    subsample_index_list = list()

    for label in np.unique(y_target):

        if verbose:
            print('Label ' + str(label))

        num_label_in_y_t = len(y_target[y_target == label])
        num_not_label_in_y = len(y[y != label])
        label_eq_y_idx = np.where(y == label)[0]

        prop_label_in_y_t = num_label_in_y_t / len(y_target)
        prop_label_in_y = 1 - (num_not_label_in_y / len(y))

        if verbose:
            print('Prop in target: ' + str(prop_label_in_y_t))
            print('Prop in original: ' + str(prop_label_in_y))

        if prop_label_in_y_t > prop_label_in_y:

            subsample_index = label_eq_y_idx

        else:

            num_subsample = num_not_label_in_y / (
                    (len(y_target) / num_label_in_y_t) - 1
            )

            num_subsample = int(np.ceil(num_subsample))

            subsample_index = np.random.choice(label_eq_y_idx,
                                               size=num_subsample, replace=False)

        subsample_index_list.append(subsample_index)

    all_subsample_index = np.concatenate(subsample_index_list)

    y_subsample = y[all_subsample_index]

    if X is not None:
        X_subsample = X[all_subsample_index, :]
    else:
        X_subsample = None

    return y_subsample, X_subsample


def predict_decision(X, y, clf, control_clf='default', get_importance=True, loading_bar=False,
                     n_cv_splits=5, n_repeats=5, chunk_vector=None, cv_random_seed=None,
                     feature_importance_type='list', feature_names=None, extra_feature_fields=None,
                     include_feature_importance=True, extra_X=None, extra_y=None, match_y_extra_dist=False,
                     cv_split_method='stratifiedKFold', tune_hyperparam=False,
                     n_inner_loop_cv_splits=5, n_inner_loop_repeats=2,
                     hyperparam_tune_method='nested_cv_w_loop', param_grid=[{'C':  np.logspace(-5, 2, 11)}],
                     param_search_scoring_method=None, include_confidence_score=False,
                     aud_cond=None, vis_cond=None, classifier_object_name='svc',
                     trial_cond=None):
    """
    # TODO: add option to subsample y_extra based on the class distribution in y_test
    Evaluates classifier performance using repeated stratified k-fold cross-validation.
    NOTE: This function is going to be supercedeed by predict_decision_multi_clf
    Parameters
    -----------
    X           : (numpy ndarray)
        feature matrix of shape (num_samples, num_features)
    y           : (numpy ndarray)
        label vector of shape (num_samples, )
    clf         : (sklearn classifier object)
        classifier to use
    control_clf : (str, sklearn classifier object, NoneType)
        if (1) 'default' - runs a dummy cer based on class distribution in training set
           (2) None      - does not run a control classfier
           (3) sklearn classifier object, any classifier you want to compare with
    get_importance : (bool)
        if True, attempts to get the feature importance (eg. coefficients) of the model
            in linear sklearn models, this will work, and should return coefs with shape
                (num_class, num_features)
    chunk_vector : (None or numpy ndarray)
        a vector that controls the chunking in cross validation splits, such that samples within the same
        chunk are not split: they will either all be in the training set, or all in the validation set.

    extra_X : (numpy ndarray)
        feature matrix of shape (num_samples, num_features) that you want use your trained model to make predictions about on top
        of the testing set (eg. if you want to see if a model trained in a certain dataset will generalise
        to another dataset without training on that dataset)
    extra_y : (numpy ndarray)
        label vector of shape (num_samples, )
    classifier_object_name : (str)
        name of the classifier object when the clf object provided is a Pipeline object.
        this is needed to get the coefficients of the classifier object.
    extra_feature_fields : (dictionary)
        fields that you want to add to your features, eg. location of each neuron
        these will be accsesed via the dictinoary: key will be the field name, and
        the items will be the data associated with each feature.
    trial_cond : (dictionary)
        fields associated with each trial / sample (row of your feature matrix)
        this is a generalisiation of aud_cond and vis_cond, and allows for arbitrary data
        associated with each trial (eg. reaction time)
    """

    if control_clf is not None:
        if control_clf == 'default':
            dummy_clf = dummy.DummyClassifier(strategy='stratified')
        elif control_clf == 'most_frequent':
            dummy_clf = dummy.DummyClassifier(strategy='most_frequent')
        else:
            dummy_clf = control_clf

    # sklearn object to train-test splits
    if chunk_vector is None:
        if cv_split_method == 'stratifiedKFold':
            cv_splitter = RepeatedStratifiedKFold(n_splits=n_cv_splits, n_repeats=n_repeats)
        elif cv_split_method == 'repeatedKFold':
            cv_splitter = RepeatedKFold(n_splits=n_cv_splits, n_repeats=n_repeats)
    else:
        # TODO: add the random seed option.
        cv_splitter = groupKFoldRandom(groups=chunk_vector, n=n_cv_splits)

    accuracy_score_list = list()
    hyperparam_search_score_list = list()
    extra_accuracy_score_list = list()
    dummy_accuracy_score_list = list()
    feature_importance_list = list()

    # Include confidence metric and the audio-visual condition to be used later when looking at the metric
    # for different stimulus conditions.
    if include_confidence_score:
        confidence_score_df_list = list()

    if tune_hyperparam:

        if hyperparam_tune_method == 'nested_cv':
            print('WARNING: implementation incomplete')
            # nested cv using sklearn built in methods
            outer_cv = cv_splitter
            inner_cv = RepeatedKFold(n_splits=n_inner_loop_cv_splits, n_repeats=n_inner_loop_repeats)

            clf = sklselection.GridSearchCV(estimator=clf, param_grid=param_grid,
                                            cv=inner_cv)
            nested_cv_score = cross_val_score(clf, X=X, y=y, cv=outer_cv)
            accuracy_score_list = nested_cv_score

        elif hyperparam_tune_method == 'nested_cv_w_loop':
            # Note here that we use a nested cross validation strategy
            # See for example: https://chrisalbon.com/machine_learning/model_evaluation/nested_cross_validation/
            # But there will be high variability of results with small datasets
            # tune hyperparameter using cross validation, then do a final test on the test set

            # Outer-loop
            for n_split, (dev_index, test_index) in enumerate(cv_splitter.split(X, y)):
                X_dev, X_test = X[dev_index], X[test_index]
                y_dev, y_test = y[dev_index], y[test_index]

                # Inner loop: do cross-validation on development set to find best hyperparameter
                if chunk_vector is None:
                    # if no cv method provided, then this does 5-fold cross validation.
                    inner_loop_cv_splitter = sklselection.RepeatedStratifiedKFold(
                        n_splits=n_inner_loop_cv_splits, n_repeats=n_inner_loop_repeats)

                    grid_search = sklselection.GridSearchCV(clf, param_grid, n_jobs=-1,
                                                            cv=inner_loop_cv_splitter,
                                                            scoring=param_search_scoring_method,
                                                            refit=True, iid=True)

                    grid_search_results = grid_search.fit(X=X_dev, y=y_dev)

                    # evaluate best hyperparameter on the test set
                    best_model = grid_search_results.best_estimator_
                    accuracy_score = best_model.score(X_test, y_test)
                    accuracy_score_list.append(accuracy_score)

                    grid_search_result_df = pd.DataFrame(grid_search_results.cv_results_)
                    grid_search_result_df['n_split'] = n_split

                    hyperparam_search_score_list.append(grid_search_result_df)

                    if include_confidence_score:
                        test_confidence_score = best_model.decision_function(X_test)
                        dev_confidence_score = best_model.decision_function(X_dev)

                        if aud_cond is not None and vis_cond is not None:
                            print('Aud cond and vis cond arugment soon to be generalised'
                                  'to trial_cond, use that instead.')
                            dev_confidence_df = pd.DataFrame({
                                'confidence_score': dev_confidence_score,
                                'aud_cond': aud_cond[dev_index],
                                'vis_cond': vis_cond[dev_index],
                            })
                        else:
                            dev_confidence_df = pd.DataFrame({
                                'confidence_score': dev_confidence_score,
                            })

                        dev_confidence_df['dataset'] = 'dev'

                        if aud_cond is not None and vis_cond is not None:
                            test_confidence_df = pd.DataFrame({
                                'confidence_score': test_confidence_score,
                                'aud_cond': aud_cond[test_index],
                                'vis_cond': vis_cond[test_index]
                            })
                        else:
                            test_confidence_df = pd.DataFrame({
                                'confidence_score': test_confidence_score,
                            })

                        test_confidence_df['dataset'] = 'test'

                        if trial_cond is not None:
                            for field_name, field_data in trial_cond.items():
                                dev_confidence_df[field_name] = field_data[dev_index]
                                test_confidence_df[field_name] = field_data[test_index]

                        confidence_score_df = pd.concat([dev_confidence_df, test_confidence_df])

                        confidence_score_df['n_split'] = n_split

                        confidence_score_df_list.append(confidence_score_df)

                # Run a control classifier
                if control_clf is not None:
                    dummy_accuracy_score = dummy_clf.fit(X_dev, y_dev).score(X_test, y_test)
                    dummy_accuracy_score_list.append(dummy_accuracy_score)

                # Get the feature importance of the best model
                if get_importance is True:
                    if type(best_model) is skl.pipeline.Pipeline:
                        feature_importance = best_model[classifier_object_name].coef_
                    else:
                        feature_importance = best_model.coef_
                    if feature_importance_type == 'list':
                        feature_importance_list.append(feature_importance)
                    elif feature_importance_type == 'df':
                        if feature_names is None:
                            feature_names = np.arange(0, np.shape(feature_importance)[1])

                        feature_importance_df = pd.DataFrame.from_dict({'feature': feature_names,
                                                                        'weight': feature_importance[0, :],
                                                                        'n_split': np.repeat(n_split,
                                                                                             len(feature_names)),
                                                                        }
                                                                       )
                        # add extra information about each feature (eg. cell location of each neuron)
                        if extra_feature_fields is not None:
                            for feature_field_name, feature_field_value in extra_feature_fields.items():
                                feature_importance_df[feature_field_name] = feature_field_value

                        feature_importance_list.append(feature_importance_df)

        elif hyperparam_tune_method == 'hold_out':

            print('Implement hold out hyperparameter tuning.')

        else:
            print('No valid method selected.')

    else:

        for n_split, (train_index, test_index) in enumerate(cv_splitter.split(X, y)):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # TODO: need to do fit_trasnform if using pipeline (not sure why, got the error
            # 'StandardScalar has no attribute mean_'
            fitted_model = clf.fit(X_train, y_train)
            accuracy_score = fitted_model.score(X_test, y_test)
            accuracy_score_list.append(accuracy_score)

            if extra_X is not None and extra_y is not None:
                if match_y_extra_dist:
                    extra_y, extra_X = stratify_from_target_y(y_target=y_test,
                                                              y=extra_y, X=extra_X)
                extra_accuracy_score = fitted_model.score(extra_X, extra_y)
                extra_accuracy_score_list.append(extra_accuracy_score)

            if get_importance is True:
                if type(clf) is skl.pipeline.Pipeline:
                    feature_importance = clf[classifier_object_name].coef_
                elif 'coef_' in clf.__dict__.keys():
                    feature_importance = clf.coef_[0, :]
                else:
                    feature_importance = np.zeros((np.shape(X_train)[1], ))

                if feature_importance_type == 'list':
                    feature_importance_list.append(feature_importance)
                elif feature_importance_type == 'df':
                    if feature_names is None:
                        if 'coef_' in clf.__dict__.keys():
                            feature_names = np.arange(0, np.shape(clf.coef_)[1])
                        else:
                            feature_names = np.arange(0, np.shape(X_train)[1])

                    feature_importance_df = pd.DataFrame.from_dict({'feature': feature_names,
                                                                    #  'weight': clf.coef_[0, :],
                                                                    'weight': feature_importance,
                                                                     'n_split': np.repeat(n_split, len(feature_names)),
                                                                    }
                                                                   )

                    # add extra information about each feature (eg. cell location of each neuron)
                    if extra_feature_fields is not None:
                        for feature_field_name, feature_field_value in extra_feature_fields.items():
                            feature_importance_df[feature_field_name] = feature_field_value

                    feature_importance_list.append(feature_importance_df)

            if control_clf is not None:
                dummy_accuracy_score = dummy_clf.fit(X_train, y_train).score(X_test, y_test)
                dummy_accuracy_score_list.append(dummy_accuracy_score)

    if control_clf is None:
        dummy_accuracy_score_list.append(np.repeat(np.nan, len(accuracy_score_list)))

    if get_importance:
        if feature_importance_type == 'list':
            feature_importance_output = feature_importance_list
        elif feature_importance_type == 'df':
            feature_importance_output = pd.concat(feature_importance_list)
    else:
        feature_importance_output = None

    if len(extra_accuracy_score_list) > 0:
        accuracy_score_output = {'original_condition': accuracy_score_list,
                                 'extra_condition': extra_accuracy_score_list}
    else:
        accuracy_score_output = accuracy_score_list

    if tune_hyperparam:
        hyperparam_tune_output = pd.concat(hyperparam_search_score_list)
    else:
        hyperparam_tune_output = None

    if include_confidence_score:
        confidence_score_output = pd.concat(confidence_score_df_list)
    else:
        confidence_score_output = None

    return accuracy_score_output, dummy_accuracy_score_list, feature_importance_output,\
           hyperparam_tune_output, confidence_score_output


def train_and_evaluate(X, y, clf, n_cv_splits=5, n_repeat=5, get_importance=False, rand_seed=123,
                       run_parallel=False, run_stratified=True, get_confusion_matrix=False):
    """
    Fit and evaluate the performance of a single classifier.
    Parameters
    -------------
    X : (numpy ndarray)
        feature matrix (numpy ndarray) of shape (num_samples, num_features)
    y : (numpy ndarray)
        label vector (numpy ndarray)  of shape (nsamples, )
    clf : (sklearn classifier object)
        classifier to use
    rand_seed : (int)
        random seed for use in cross-validation
    get_importance : (bool)
        whether to get some type of importance metric from the classifier
        for linear classifiers (eg. linear SVM), this generally corresponds to the weights
        of each feature
    run_parallel : (bool)
        whether to run cross validation as parallel processes
        generally provide a speed up, but is more resource intensive
    run_stratified : (bool)
        whether to run stratified cross-validation, which means that the class
        distribution is kept the same in the training and testing set
    get_confusion_matrix : (bool)
        whether to get the confusion matrix of the classifier
        currently only implemented if run_parallel is False
    Returns
    -------------
    accuracy_score_list : (list)
        list of float, each of which is the accuracy - P(predicted class = true class) of
        the classifier for each test set of the n-fold cross validation process
    feature_importance_list : (list)
    confusion_matrix_list : (list)
        list of numpy arrays, each of which is the confusion matrix
        in a specific test set of the n-fold cross validation process
        therefore the list have length equal to n_cv_splits * n_repeats
    """


    # sklearn object to train-test splits
    if run_stratified:
        cv = RepeatedStratifiedKFold(n_splits=n_cv_splits, n_repeats=n_repeat,
                                   random_state=rand_seed)
    else:
        cv = RepeatedKFold(n_splits=n_cv_splits, n_repeats=n_repeat,
                                   random_state=rand_seed)

    num_features = np.shape(X)[1]
    num_class = len(np.unique(y))

    accuracy_score_list = list()
    feature_importance_list = list()
    if get_confusion_matrix:
        confusion_matrix_list = list()
    else:
        confusion_matrix_list = None

    if run_parallel:
        """
        accuracy_score_list, window_control_accuracy_list, feature_importance_list = zip(
            *Parallel(n_jobs=-1, backend='threading')(delayed(predict_decision)(X=X, y=y, clf=clf,
                    control_clf='default', n_cv_splits=n_cv_splits, n_repeats=n_repeats,
                    feature_importance_type=feature_importance_type,
                    extra_feature_fields=extra_field_dict) for X in tqdm(feature_matrix_list)))
        """
        # TODO: look into sklearn.base.ClassifierMixin to make K fold stratification being used.
        accuracy_score_list = cross_val_score(estimator=clf, X=X, y=y, cv=cv, n_jobs=-1)
        feature_importance_list = None

    else:
        for train_index, test_index in cv.split(X, y):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            fitted_model = clf.fit(X_train, y_train)
            accuracy_score = fitted_model.score(X_test, y_test)
            accuracy_score_list.append(accuracy_score)

            if get_confusion_matrix:
                y_test_pred = fitted_model.predict(X_test)
                confusion_matrix = sklmetrics.confusion_matrix(y_true=y_test,
                                                               y_pred=y_test_pred)
                confusion_matrix_list.append(confusion_matrix)

            if get_importance:
                if hasattr(clf, 'coef_'):
                    feature_importance = clf.coef_
                else:
                    feature_importance = np.full((num_class, num_features), np.nan)
                feature_importance_list.append(feature_importance)

    return accuracy_score_list, feature_importance_list, confusion_matrix_list
    

def predict_decision_multi(X_list, y, clf_list, clf_name_list, X_names, feature_names_list, include_control='default',
                           get_importance=True, n_cv_splits=5, n_repeats=5, rand_seed=123):
    """
    Generalisation of predict_decision, allowing for 
    (1) running a list of classifiers.
    (2) running a list of feature matrices.
    Parameters
    ------------
    X_list : (list of numpy ndarray)
        list of feature matrices
    y : (numpy ndarrays)
        vector of class labels 
    clf_list : (list of sklearn objects)
        list of classifier
    clf_name_list : (list of str)
        list of the names of the classifiers in clf_list
    X_names : (list of str)
        list of names of the feature matrices (NOT the names of the features)
    include_control : (bool)
    rand_seed : (int)
        random seed for use in cross-validation
        this is useful here to fairly compare between the different classifiers to make sure
        they are trained on the same training sets and tested on the same test sets
    Returns
    -------------
    classifier_result_df    : 
    feature_importance_dict : dictionary of dataframes

    TODO: handle feature names and preserve that in featurer_importance_list
    """

    assert len(X_list) == len(X_names)
    assert len(clf_list) == len(clf_name_list)
    
    classifier_result_dict = defaultdict(list)
    num_splits = n_cv_splits * n_repeats
    
    if include_control == 'default':
        clf_list.append(dummy.DummyClassifier(strategy='stratified'))
        clf_name_list.append('stratified-control')

    feature_importance_dict = dict()

    for X, X_name, feature_names in zip(X_list, X_names, feature_names_list):

        # list to store feature importance for a particular feature set
        feature_importance_list = list()
        
        for clf, clf_name in zip(clf_list, clf_name_list):

            accuracy_score, feature_importance = train_and_evaluate(
                X, y, clf=clf, n_cv_splits=n_cv_splits, n_repeat=n_repeats,
                get_importance=get_importance, rand_seed=rand_seed)

            classifier_result_dict['classifier_name'].append(np.repeat(clf_name, num_splits))
            classifier_result_dict['split_ref_num'].append(np.arange(0, num_splits))
            classifier_result_dict['accuracy_score'].append(accuracy_score)
            # classifier_result_dict['feature_importance'].append(feature_importance)
            classifier_result_dict['X_name'].append(np.repeat(X_name, num_splits))

            # feature_importance_list.append(feature_importance)

            # get feature improtance across all splits and convert to dataframe
            # feature_importance_matrix = np.stack(feature_importance_list, axis=-1)
            if get_importance:
                if len(X_list) > 1:
                    feature_importance_matrix = np.stack(feature_importance, axis=-1)
                else:
                    feature_importance_matrix = feature_importance
                print(np.shape(feature_importance_matrix))
                feature_importance_df = feature_importance_matrix_to_df(feature_importance_matrix,
                                                                    feature_names=feature_names)
                feature_importance_dict[X_name + '_' + clf_name] = feature_importance_df
            else:
                feature_importance_dict = None

    flattened_dict = dict((k, np.array(classifier_result_dict[k]).flatten()) for 
                         k in classifier_result_dict.keys())

    classifier_result_df = pd.DataFrame.from_dict(flattened_dict)
                        
    return classifier_result_df, feature_importance_dict


def feature_importance_matrix_to_df(feature_importance_matrix, feature_names, feature_dim=1, repeat_dim=2, class_dim=0):

    num_class = np.shape(feature_importance_matrix)[class_dim]
    num_features = np.shape(feature_importance_matrix)[feature_dim]
    num_splits = np.shape(feature_importance_matrix)[repeat_dim]

    
    feature_names_list = list()
    coef_values = list()
    split_idx = list()
    class_num_list = list()

    for class_n in np.arange(0, num_class):
        class_feature_importance = np.squeeze(feature_importance_matrix[
        class_n, :, :])

        for feature_num in np.arange(num_features):

            feature_names_list.append(np.repeat(feature_names[feature_num], num_splits))
            coef_values.append(class_feature_importance[feature_num, :])
            split_idx.append(np.arange(0, num_splits))
            class_num_list.append(np.repeat(class_n, num_splits))
    
    class_feature_importance_df = pd.DataFrame({
        'featureName': np.array(feature_names_list).flatten(),
        'splitIdx': np.array(split_idx).flatten(), 
        'coefVal': np.array(coef_values).flatten(),
         'classNum': np.array(class_num_list).flatten()
        })

    return class_feature_importance_df


def run_window_classification(activity_dataset, labels, clf, num_bins=50, window_width=20, num_window=10,
                              window_width_units='bins', custom_window_locs=None,
                              even_steps=True, run_parallel=False, n_cv_splits=5, n_repeats=5,
                              cv_split_method='stratifiedKFold',
                              random_shuffle_labels=False, window_type='center',
                              feature_importance_type='list', include_cell_loc=False,
                              include_peri_event_time=False,
                              print_loading_bar=False, get_importance=True,
                              control_clf='most_frequent', activity_name='firing_rate',
                              extra_condition_activity_dataset=None, extra_labels=None,
                              include_baseline_accuracy=False, tune_hyperparam=False,
                              n_inner_loop_cv_splits=5, n_inner_loop_repeats=2,
                              hyperparam_tune_method='nested_cv_w_loop', param_grid=[{'C': np.logspace(-5, 2, 11)}],
                              param_search_scoring_method=None, scale_features=False, include_confidence_score=False,
                              peri_event_time_name='PeriEventTime', trial_cond=None,
                              ):
    """
    Generates a feature matrix from each time window, and run classifiers to predict the event identity.
    Arguments
    -------------------
    activity_dataset      : (xarray dataset)
        object containing the feature matrix for classification
    labels                : (numpy ndarray)
        (num_trial, ) array with int labels (eg. left vs. right choice) to classify
    num_bins              : (int)
        number of time bins in the entire aligned time window
    window_width          : (int)
        width (in time bins) of the window to average activity over
    num_window            : (int)
        number of time windows to make
    window_width_units    : (str)
        units of the window_width arugment
        option 1: 'bins'
        option 2: 'seconds'
    even_steps           : (bool)
        if True, automatically spreads the windows over the entire time range
    custom_window_locs  : (numpy ndarray)
        if not None, then this be a 1D numpy ndarray with shape (numWindow, )
        these will be the start points each window, such that
        the window span will be start_point + window_width
        TODO: allow specifying the center of the windows instead of the starts
        Window location are specified in bin units.
    n_cv_splits : (int)
        number of cross validation splits
    n_repeats : (int)
        number of times to repeat the cross-validation evaluation
    random_shuffle_labels : (bool)
        if True, randomly shuffle labels provided (to act as a control)
    window_type : (str)
        type of window to make around each window location.
        option 1: 'forward', each window starts at window_loc and ends at window_loc + window_width
        option 2: 'center', each window is centered around window_loc with two sides adding up to window_width
    feature_importance_type : (str)
        type of data format to return the feature importance in
        option 1 : 'list': list of numpy arrays
        option 2 : 'df' pandas dataframe
    include_cell_loc : (bool)
        whether to include cell location in the feature importance dataframe
    scale_features : (bool)
        whether to scale features before running decoding
    include_confidence_score : (bool)
        applies to SVM only for now
        compute confidence score for each sample for being in the left or right
        for SVM, this is just the dot product between the weight vector and the sample
    TODO: num_bins should just be meta-data from activity_dataset / be inferred by the Time dimension shape
    TODO: tqdm for parallel processing jobs still not running properly
    """

    if random_shuffle_labels is True:
        np.random.shuffle(labels)  # note that this is done in-place

    if (even_steps) is True and (custom_window_locs is None):
        window_start_locs = np.linspace(0, num_bins-window_width, num_window)
        window_end_locs = window_start_locs + window_width - 1  # due to the zero indexing
        # example: my window start at 0, and my width is 3, and so my window end loc is really 2
        # since 0, 1, 2 makes up three windows

        assert window_end_locs[-1] == num_bins - 1

    elif custom_window_locs is not None:
        if type(custom_window_locs) is int:
            custom_window_locs = np.array([custom_window_locs])
        elif type(custom_window_locs) is list:
            custom_window_locs = np.array(custom_window_locs)
        window_start_locs = custom_window_locs
        window_end_locs = window_start_locs + window_width - 1

    if clf is None:
        print('Warning: no classifier specified, using out-of-the-box linear SVM')
        clf = svm.SVC(kernel='linear')

    y = labels

    window_start_loc_list = list()
    window_end_loc_list = list()
    window_clf_accuracy_list = list()
    window_control_accuracy_list = list()
    repeated_cv_split_index_list = list()

    if extra_condition_activity_dataset is not None:
        window_clf_extra_condition_accuracy_list = list()

    classification_results_dict = defaultdict(list)

    if include_cell_loc:
        extra_field_dict = {'cell_loc': activity_dataset['CellLoc'].values}
    else:
        extra_field_dict = None

    if include_confidence_score:
        aud_cond = activity_dataset.isel(Cell=0)['audDiff'].values
        vis_cond = activity_dataset.isel(Cell=0)['visDiff'].values
    else:
        aud_cond = None
        vis_cond = None

    if run_parallel is True:
        # first create a list of feature matrix we are going to use

        feature_matrix_list = Parallel(n_jobs=-1, backend='threading')(
            delayed(make_windowed_feature_matrix)(activity_dataset=activity_dataset, window_start_loc=win_start,
                                                  window_width=window_width, activity_name=activity_name,
                                                  window_width_units=window_width_units,
                                                  window_type=window_type,
                                                  ) for win_start in window_start_locs)

        if extra_condition_activity_dataset is not None:
            extra_feature_matrix_list = Parallel(n_jobs=-1, backend='threading')(
            delayed(make_windowed_feature_matrix)(activity_dataset=extra_condition_activity_dataset, window_start_loc=win_start,
                                                  window_width=window_width, activity_name=activity_name,
                                                  window_width_units=window_width_units,
                                                  window_type=window_type) for win_start in window_start_locs)

        if extra_condition_activity_dataset is None:
            accuracy_score_list, window_control_accuracy_list, feature_importance_list, hyperparam_tune_list,\
                confidence_score_list = zip(
                *Parallel(n_jobs=-1, backend='threading')(delayed(predict_decision)(X=X, y=y, clf=clf,
                        control_clf=control_clf, n_cv_splits=n_cv_splits, n_repeats=n_repeats,
                        feature_importance_type=feature_importance_type,
                        extra_feature_fields=extra_field_dict, get_importance=get_importance,
                        cv_split_method=cv_split_method, tune_hyperparam=tune_hyperparam,
                        n_inner_loop_cv_splits=n_inner_loop_cv_splits, n_inner_loop_repeats=n_inner_loop_repeats,
                        hyperparam_tune_method=hyperparam_tune_method,
                        param_grid=param_grid, param_search_scoring_method=param_search_scoring_method,
                        include_confidence_score=include_confidence_score, aud_cond=aud_cond, vis_cond=vis_cond,
                        trial_cond=trial_cond) for X in tqdm(feature_matrix_list, disable=(not print_loading_bar))))
            classification_results_dict['classifier_score'] = np.array(accuracy_score_list).flatten()
        else:
            accuracy_score_list, window_control_accuracy_list, feature_importance_list, hyperparam_tune_list, \
            confidence_score_list = zip(
                *Parallel(n_jobs=-1, backend='threading')(delayed(predict_decision)(X=X, y=y, clf=clf,
                        control_clf=control_clf, n_cv_splits=n_cv_splits, n_repeats=n_repeats,
                        feature_importance_type=feature_importance_type,
                        extra_feature_fields=extra_field_dict, get_importance=get_importance,
                        cv_split_method=cv_split_method,
                        extra_X=X_extra, extra_y=extra_labels, tune_hyperparam=tune_hyperparam,
                        n_inner_loop_cv_splits=n_inner_loop_cv_splits, n_inner_loop_repeats=n_inner_loop_repeats,
                        hyperparam_tune_method=hyperparam_tune_method, param_grid=param_grid,
                        param_search_scoring_method=param_search_scoring_method,
                        include_confidence_score=include_confidence_score, aud_cond=aud_cond, vis_cond=vis_cond,
                        trial_cond=trial_cond) for X, X_extra in zip(feature_matrix_list, extra_feature_matrix_list)))

            # accuracy_score_list is now list of dicts
            original_data_accuracy_score = [x['original_condition'] for x in accuracy_score_list]
            extra_data_accuracy_score = [x['extra_condition'] for x in accuracy_score_list]
            classification_results_dict['classifier_score'] = np.array(original_data_accuracy_score).flatten()

            classification_results_dict['classifier_score_extra'] = np.array(extra_data_accuracy_score).flatten()

        # Note: repeat of [1, 2, 3] gives [1, 1, 2, 2, 3, 3], whereas tile give [1, 2, 3, 1, 2, 3]
        classification_results_dict['window_start_locs'] = np.repeat(window_start_locs, n_cv_splits * n_repeats)
        classification_results_dict['window_end_locs'] = np.repeat(window_end_locs, n_cv_splits * n_repeats)

        classification_results_dict['control_score'] = np.array(window_control_accuracy_list).flatten()
        classification_results_dict['repeated_cv_index'] = np.tile(np.arange(n_cv_splits * n_repeats),
                                                                               len(window_start_locs))

        # pdb.set_trace()
        classification_results_df = pd.DataFrame.from_dict(classification_results_dict)

        # Do the same for hyperparam tuning results: get the tuning result for each time window
        if tune_hyperparam:
            hyperparam_tune_results_list = list()
            # reminder: hyperparam_tune_list is a list of dataframes
            # hyperparam_tune_list has the same length as the number of windows
            for n_tune_df, tune_df in enumerate(hyperparam_tune_list):
                tune_df['window_start_locs'] = window_start_locs[n_tune_df]
                tune_df['window_end_locs'] = window_end_locs[n_tune_df]
                hyperparam_tune_results_list.append(tune_df)



        # TODO: add information about the window in the feature importance dataframe.
        
    else:

        hyperparam_tune_list = list()
        confidence_score_list = list()
        for window_start_loc, window_end_loc in zip(window_start_locs, window_end_locs):
            feature_matrix = make_windowed_feature_matrix(activity_dataset, window_start_loc=window_start_loc,
                                                          window_width=window_width, activity_name=activity_name,
                                                          window_width_units=window_width_units,
                                                          window_type=window_type)

            if extra_condition_activity_dataset is None:
                X_extra = None
                extra_labels = None
            else:
                X_extra = make_windowed_feature_matrix(extra_condition_activity_dataset,
                                                       window_start_loc=window_start_loc,
                                                       window_width=window_width, activity_name=activity_name,
                                                       window_width_units=window_width_units,
                                                       window_type=window_type)

                print(np.shape(X_extra))
                print(np.shape(extra_labels))

            accuracy_score_list, dummy_accuracy_score_list, feature_importance_list, hyperparam_tune_output, \
                confidence_score_output = predict_decision(X=feature_matrix, y=y,
                                                            clf=clf, control_clf=control_clf,
                                                            n_cv_splits=n_cv_splits,
                                                            n_repeats=n_repeats,
                                                            feature_importance_type=feature_importance_type,
                                                            extra_feature_fields=extra_field_dict,
                                                            get_importance=get_importance,
                                                            cv_split_method=cv_split_method,
                    extra_X=X_extra, extra_y=extra_labels, tune_hyperparam=tune_hyperparam,
                    n_inner_loop_cv_splits=n_inner_loop_cv_splits, n_inner_loop_repeats=n_inner_loop_repeats,
                    hyperparam_tune_method=hyperparam_tune_method, param_grid=param_grid,
                    param_search_scoring_method=param_search_scoring_method,
                    include_confidence_score=include_confidence_score, aud_cond=aud_cond, vis_cond=vis_cond,
                    trial_cond=trial_cond)

            # pdb.set_trace()

            # append to result lists
            hyperparam_tune_list.append(hyperparam_tune_output)
            confidence_score_list.append(confidence_score_output)

            if extra_condition_activity_dataset is None:
                accuracy_score_list = np.array(accuracy_score_list)

            dummy_accuracy_score_list = np.array(dummy_accuracy_score_list)

            if extra_condition_activity_dataset is not None:
                # accuracy_score_list is now list of dicts
                print(accuracy_score_list)
                original_data_accuracy_score = accuracy_score_list['original_condition']
                extra_condition_accuracy_score = accuracy_score_list['extra_condition']
                accuracy_score_list = original_data_accuracy_score

            window_start_loc_list.append(np.repeat(window_start_loc, len(accuracy_score_list)))

            window_end_loc_list.append(np.repeat(window_end_loc, len(accuracy_score_list)))
            repeated_cv_split_index_list.append(np.arange(len(accuracy_score_list)))
            window_clf_accuracy_list.append(accuracy_score_list)
            window_control_accuracy_list.append(dummy_accuracy_score_list)
            if extra_condition_activity_dataset is not None:
                window_clf_extra_condition_accuracy_list.append(extra_condition_accuracy_score)

        classification_results_dict['window_start_locs'] = np.concatenate(window_start_loc_list)
        classification_results_dict['window_end_locs'] = np.concatenate(window_end_loc_list)
        classification_results_dict['classifier_score'] = np.concatenate(window_clf_accuracy_list)
        classification_results_dict['control_score'] = np.concatenate(window_control_accuracy_list)
        classification_results_dict['repeated_cv_index'] = np.concatenate(repeated_cv_split_index_list)
        if extra_condition_activity_dataset is not None:
            classification_results_dict['extra_condition_classifier_score'] = \
                np.concatenate(window_clf_extra_condition_accuracy_list)

        classification_results_df = pd.DataFrame.from_dict(classification_results_dict)

        if tune_hyperparam:
            hyperparam_tune_results_list = list()
            # reminder: hyperparam_tune_list is a list of dataframes
            # hyperparam_tune_list has the same length as the number of windows
            for n_tune_df, tune_df in enumerate(hyperparam_tune_list):
                tune_df['window_start_locs'] = window_start_locs[n_tune_df]
                tune_df['window_end_locs'] = window_end_locs[n_tune_df]
                hyperparam_tune_results_list.append(tune_df)


    if include_peri_event_time:
        classification_results_df['window_start_sec'] = classification_results_df[
            'window_start_locs'].apply(
            lambda x: activity_dataset[peri_event_time_name].values[int(x)])

        classification_results_df['window_end_sec'] = classification_results_df[
            'window_end_locs'].apply(
            lambda x: activity_dataset[peri_event_time_name].values[int(x)])

        classification_results_df['window_width_sec'] = classification_results_df['window_end_sec'] - \
                                                        classification_results_df['window_start_sec']
    if include_baseline_accuracy:
        # Get baseline hit proportion if only predicting using the majority class
        baseline_accuracy = compute_baseline_performance(y=labels, metric='accuracy')
        classification_results_df['baseline_accuracy'] = baseline_accuracy

    # See whether feature importance output is needed
    if get_importance:
        if feature_importance_type == 'df':
            # Deal with having only a single dataframe (if only single window used)
            if type(feature_importance_list) == pd.core.frame.DataFrame:
                feature_importance_list = [feature_importance_list]
            df_list = list()
            for window_idx, feature_importance_df in enumerate(feature_importance_list):
                feature_importance_df['window'] = window_idx
                feature_importance_df['window_start_locs'] = window_start_locs[window_idx]
                feature_importance_df['window_end_locs'] = window_end_locs[window_idx]
                df_list.append(feature_importance_df)
            feature_importance_output = pd.concat(df_list)
        else:
            feature_importance_output = feature_importance_list
    else:
        feature_importance_output = None

    # See whether hyperparameter tuning output is needed
    if tune_hyperparam:
        # Output is a single dataframe
        hyperparam_tune_output = pd.concat(hyperparam_tune_results_list)
    else:
        hyperparam_tune_output = None

    if include_confidence_score:
        # Include window information (note that hyperparam result already did this early on)
        assert len(confidence_score_list) == len(window_start_locs)
        processed_confidence_df_list = list()
        for confidence_df, window_start_loc, window_end_loc in zip(
                confidence_score_list, window_start_locs, window_end_locs):

            confidence_df['window_start_locs'] = window_start_loc
            confidence_df['window_end_locs'] = window_end_loc

            processed_confidence_df_list.append(confidence_df)

        confidence_score_output = pd.concat(processed_confidence_df_list)
    else:
        confidence_score_output = None

    return classification_results_df, feature_importance_output, hyperparam_tune_output, confidence_score_output


def run_one_window_classification(activity_dataset, labels, activity_name='firing_rate',
                                  clf=None, trial_vector=None, trial_chunk_size=5,
                                  peri_event_time_name='PeriEventTime', decode_up_till_movement=False,
                                  window_start_time=-0.5, window_end_time=0, unfold_time=False):
    """
    Performs classification (and associated evaluation) on a single window.

    Parameters
    ----------
    activity_dataset : (xarray dataset)
        dataset with dimensinos 'Neuron' and 'Time', 'Time' should have coordinates 'PeriStimTime'
    labels : (numpy ndarray)

    clf : (sklearn or sklearn-like classifier object)
        classifier to use : expect all the standard methods of a standard sklearn model.
    trial_chunk_size
    window_start_time
    window_end_time
    unfold_time

    Returns
    -------

    """

    if (trial_vector is not None):
        chunk_vector = make_trial_chunks(trial_vector, trial_chunk_size)

    classification_results_dict = dict()

    if clf is None:
        print('Warning: no classifier specified, using out-of-the-box linear SVM')
        clf = svm.SVC(kernel='linear')

    y = labels

    time_sliced_activity = activity_dataset.where((activity_dataset[peri_event_time_name] >= window_start_time) &
                                        (activity_dataset[peri_event_time_name] <= window_end_time), drop=True)
    mean_across_time = time_sliced_activity.mean('Time')

    X = mean_across_time.transpose('Trial', 'Cell')[activity_name].values

    accuracy_score_list, dummy_accuracy_score_list, feature_importance_list, \
    hyperparam_tune_output, confidence_score_output = predict_decision(
        X, y, clf, control_clf='default', get_importance=True, loading_bar=False,
                     n_cv_splits=5, n_repeats=5, chunk_vector=None, cv_random_seed=None)

    # TODO: extract meta-data (exp and window) and add to dataframe
    classification_results_dict['classifier_score'] = np.array(accuracy_score_list).flatten()
    classification_results_dict['control_score'] = np.array(dummy_accuracy_score_list).flatten()
    classification_results_dict['window_start_time'] = np.repeat(window_start_time, len(accuracy_score_list))
    classification_results_dict['window_end_time'] = np.repeat(window_end_time, len(accuracy_score_list))

    classification_results_df = pd.DataFrame.from_dict(classification_results_dict)

    return classification_results_df, feature_importance_list


def run_single_cell_classification(alignment_folder, neuron_df, brain_region='all',
                                   min_reaction_time=0, decode_target='audLeftRight',
                                   min_trial=5, clf=None, dummy_clf=None,
                                   decoding_window=[0, 0.1], n_cv_repeat=5):
    """
    Parameters
    -----------
    :param alignment_folder:
    :param neuron_df:
    :param brain_region:
    :param min_reaction_time:
    :param decode_target:
    :param min_trial:
    :return:
    """
    import src.data.process_ephys_data as pephys


    # Lists for storing things
    cell_list = list()
    mean_cv_score_list = list()
    mean_dummy_score_list = list()
    subject_list = list()
    exp_list = list()
    baseline_bias_list = list()

    if clf is None:
        clf = sklinear.LogisticRegression(solver='lbfgs')
    if dummy_clf is None:
        dummy_clf = skl.dummy.DummyClassifier(strategy='stratified')

    for subject in np.unique(neuron_df['subjectRef']):
        subject_neuron_df = neuron_df.loc[
            neuron_df['subjectRef'] == subject
            ]
        for exp in np.unique(subject_neuron_df['expRef']):
            exp_neuron_df = subject_neuron_df.loc[
                subject_neuron_df['expRef'] == exp
            ]
            if brain_region == 'all':
                target_brain_region_list = np.unique(exp_neuron_df['cellLoc'])
            elif type(brain_region) is not list:
                target_brain_region_list = [brain_region]
            else:
                target_brain_region_list = brain_region

            for target_brain_region in target_brain_region_list:
                alignment_ds = pephys.load_subject_exp_alignment_ds(
                                alignment_folder=alignment_folder,
                                subject_num=subject, exp_num=exp,
                                target_brain_region=target_brain_region,
                                aligned_event='stimOnTime',
                                alignment_file_ext='.nc')

                for cell in tqdm(alignment_ds['Cell'].values):
                    cell_ds = alignment_ds.sel(Cell=cell)

                    if decode_target == 'audLeftRight':
                        # only get trials where audio is either on left or right
                        # (remove audio center and no audio trials)
                        cell_ds_aud_left_right = cell_ds.where(
                            ((cell_ds['audDiff'] == 60) | (cell_ds['audDiff'] == -60)), drop=True)

                        if min_reaction_time is not None:
                            cell_ds_aud_left_right = cell_ds_aud_left_right.where(
                                cell_ds_aud_left_right['firstTimeToWheelMove'] >= min_reaction_time,
                                drop=True
                            )

                        num_aud_left_trial = len(cell_ds_aud_left_right.where(
                            (cell_ds_aud_left_right['audDiff'] == -60), drop=True
                        )['Trial'].values)

                        num_aud_right_trial = len(cell_ds_aud_left_right.where(
                            (cell_ds_aud_left_right['audDiff'] == 60), drop=True
                        )['Trial'].values)

                        if (num_aud_left_trial < min_trial) or (num_aud_right_trial < min_trial):
                            print('Subject %.f experiment %.f has not enough aud left/right trials, skpping.'
                                  % (subject, exp))
                            continue

                        # get the mean activity relative to stimulus
                        cell_ds_aud_left_right_time_mean = cell_ds_aud_left_right.where(
                            (cell_ds_aud_left_right.PeriEventTime >= decoding_window[0]) &
                            (cell_ds_aud_left_right.PeriEventTime <= decoding_window[1]),
                            drop=True)['firing_rate'].mean('Time')

                        # get labels (audio left or right)
                        aud_left_right = cell_ds_aud_left_right['audDiff'].values

                        # Decoding
                        X = cell_ds_aud_left_right_time_mean.values.reshape(-1, 1)
                        y = np.sign(aud_left_right)

                    elif decode_target == 'responseLeftRight':
                        cell_ds_response_left_right = cell_ds.where(
                            ((cell_ds['responseMade'] == 1) | (cell_ds['responseMade'] == 2)) &
                            (cell_ds['firstTimeToWheelMove'] >= min_reaction_time), drop=True)

                        num_respond_left_trial = len(cell_ds_response_left_right.where(
                            (cell_ds_response_left_right['responseMade'] == 1), drop=True
                        )['Trial'].values)

                        num_respond_right_trial = len(cell_ds_response_left_right.where(
                            (cell_ds_response_left_right['responseMade'] == 2), drop=True
                        )['Trial'].values)

                        if (num_respond_left_trial < min_trial) or (num_respond_right_trial < min_trial):
                            print('Subject %.f experiment %.f has not enough res left/right trials, skipping.'
                                  % (subject, exp))
                            continue

                        cell_ds_respond_left_right_time_mean = cell_ds_response_left_right.where(
                            (cell_ds_response_left_right.PeriEventTime >= decoding_window[0]) &
                            (cell_ds_response_left_right.PeriEventTime <= decoding_window[1]),
                            drop=True)['firing_rate'].mean('Time')

                        response_left_right = cell_ds_response_left_right['responseMade'].values

                        X = cell_ds_respond_left_right_time_mean.values.reshape(-1, 1)
                        y = response_left_right

                    elif decode_target == 'visLeftRight':

                        cell_ds_vis_left_right = cell_ds.where(
                            ((cell_ds['visDiff'] < 0) | (cell_ds['visDiff'] > 0)), drop=True)

                        if min_reaction_time is not None:
                            cell_ds_vis_left_right = cell_ds_vis_left_right.where(
                                cell_ds_vis_left_right['firstTimeToWheelMove'] >= min_reaction_time,
                                drop=True
                            )

                        num_vis_left_trial = len(cell_ds_vis_left_right.where(
                            (cell_ds_vis_left_right['visDiff'] < 0), drop=True
                        )['Trial'].values)

                        num_vis_right_trial = len(cell_ds_vis_left_right.where(
                            (cell_ds_vis_left_right['visDiff'] > 0), drop=True
                        )['Trial'].values)

                        if (num_vis_left_trial < min_trial) or (num_vis_right_trial < min_trial):
                            print('Subject %.f experiment %.f has not enough vis left/right trials, skpping.'
                                  % (subject, exp))
                            continue

                        # get the mean activity relative to stimulus
                        cell_ds_vis_left_right_mean = cell_ds_vis_left_right.where(
                            (cell_ds_vis_left_right.PeriEventTime >= decoding_window[0]) &
                            (cell_ds_vis_left_right.PeriEventTime <= decoding_window[1]),
                            drop=True)['firing_rate'].mean('Time')

                        # get labels (audio left or right)
                        vis_left_right = cell_ds_vis_left_right['visDiff'].values

                        # Decoding
                        X = cell_ds_vis_left_right_mean.values.reshape(-1, 1)
                        y = np.sign(vis_left_right)

                    elif decode_target == 'visLeftRight0p4':

                        cell_ds_vis_left_right = cell_ds.where(
                            ((cell_ds['visDiff'] == -0.4) | (cell_ds['visDiff'] == 0.4)), drop=True)

                        if min_reaction_time is not None:
                            cell_ds_vis_left_right = cell_ds_vis_left_right.where(
                                cell_ds_vis_left_right['firstTimeToWheelMove'] >= min_reaction_time,
                                drop=True
                            )

                        num_vis_left_trial = len(cell_ds_vis_left_right.where(
                            (cell_ds_vis_left_right['visDiff'] == -0.4), drop=True
                        )['Trial'].values)

                        num_vis_right_trial = len(cell_ds_vis_left_right.where(
                            (cell_ds_vis_left_right['visDiff'] == 0.4), drop=True
                        )['Trial'].values)

                        if (num_vis_left_trial < min_trial) or (num_vis_right_trial < min_trial):
                            print('Subject %.f experiment %.f has not enough vis left/right trials, skpping.'
                                  % (subject, exp))
                            continue

                        # get the mean activity relative to stimulus
                        cell_ds_vis_left_right_mean = cell_ds_vis_left_right.where(
                            (cell_ds_vis_left_right.PeriEventTime >= decoding_window[0]) &
                            (cell_ds_vis_left_right.PeriEventTime <= decoding_window[1]),
                            drop=True)['firing_rate'].mean('Time')

                        # get labels (audio left or right)
                        vis_left_right = cell_ds_vis_left_right['visDiff'].values

                        # Decoding
                        X = cell_ds_vis_left_right_mean.values.reshape(-1, 1)
                        y = np.sign(vis_left_right)
                    else:
                        print('Error: no valid decoding target specified.')

                    cv_splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_cv_repeat)
                    cv_score = sklselection.cross_val_score(clf, X, y, cv=cv_splitter,
                                                            n_jobs=-2)
                    mean_cv_score = np.mean(cv_score)

                    # Dummy decoding as a control
                    dummy_cv_score = sklselection.cross_val_score(dummy_clf, X, y, cv=cv_splitter,
                                                                  n_jobs=-2)
                    mean_dummy_cv_score = np.mean(dummy_cv_score)

                    # Adding results to the lists
                    cell_list.append(cell)
                    mean_cv_score_list.append(mean_cv_score)
                    mean_dummy_score_list.append(mean_dummy_cv_score)
                    subject_list.append(subject)
                    exp_list.append(exp)

                    # Get the baseline bias
                    baseline_bias = compute_baseline_performance(y=y, metric='accuracy')
                    baseline_bias_list.append(baseline_bias)

    single_cell_decoding_df = pd.DataFrame.from_dict({
        'subjectRef': subject_list,
        'expRef': exp_list,
        'Cell': cell_list,
        'meanCVscore': mean_cv_score_list,
        'baselineAccuracy': baseline_bias_list,
        'meanDummyCVscore': mean_dummy_score_list
    })

    single_cell_decoding_df['decode_target'] = decode_target

    return single_cell_decoding_df


def get_accuracy_mean_and_std(classification_result_df, across_window=True,
                              across_variables=['window_start_locs'],
                              accuracy_metric_name='classifier_score'):
    """
    Computes mean accuracy
    :param classification_result_df:
    :param across_window:
    :param across_variables:
    :return:
    """

    if across_window:
        classifier_mean = classification_result_df.groupby(['window_start_locs'], as_index=False).agg(
            {accuracy_metric_name: 'mean',
             'control_score': 'mean'})

        classifier_std = classification_result_df.groupby(['window_start_locs'], as_index=False).agg(
            {accuracy_metric_name: 'std',
             'control_score': 'std'})

    else:
        classifier_mean = classification_result_df.groupby(across_variables, as_index=False).agg(
            {accuracy_metric_name: 'mean',
             'control_score': 'mean'})

        classifier_std = classification_result_df.groupby(across_variables, as_index=False).agg(
            {accuracy_metric_name: 'std',
             'control_score': 'std'})

    return classifier_mean, classifier_std


def get_accuracy_percentile(classification_result_df, lower_percentile=5, upper_percentile=95):

    # From: https://stackoverflow.com/questions/17578115/pass-percentiles-to-pandas-agg-function
    def percentile(n):
        def percentile_(x):
            return np.percentile(x, n)

        percentile_.__name__ = 'percentile_%s' % n
        return percentile_

    classifier_percentile = classification_result_df.groupby(['window_start_locs'], as_index=False).agg(
            {'classifier_score':  [percentile(lower_percentile), percentile(upper_percentile)],
             'control_score': [percentile(lower_percentile), percentile(upper_percentile)]})

    return classifier_percentile


def make_some_classifier(type='SVM'):
    """
    Create some simple classifier objects to get started.
    Parameters
    ----------
    type

    Returns
    -------

    """


    return clf


def make_trial_chunks(trial_vector, chunk_size=5):

    # TODO: convert trial vector to chunks


    return trial_chunks


def save_windowed_classification_results(classification_results_df, feature_importance_list,
                                         labels,
                                        aligned_xarray, subject_ref=None, brain_loc=None,
                                         exp_ref=None,
                                         save_path=None,
                                         decode_target=None, alignment_time_name='PeriStimTime',
                                         peri_event_time_name='PeriEventTime'):
    """
    Save all the data associated with a classification result.
    This includes the feature matrix, labels, info about the feature matrix and labels,
    classifier used, etc.

    Parameters
    ----------
    classification_results_df (pandas dataframe)
    feature_importance_list
    aligned_xarray
    decode_target : (str)
        string to specify what is being decoded
        examples: 'left-right', 'left-right-nogo', 'audio-on-off'
    Returns
    -------

    """

    classification_result_bunch = bunch.Bunch()
    classification_result_bunch['subset_subject_ref'] = subject_ref
    classification_result_bunch['subset_brain-region'] = brain_loc
    classification_result_bunch['subset_exp_number'] = exp_ref
    classification_result_bunch['decode_target'] = decode_target
    classification_result_bunch['features'] = aligned_xarray.isel(Exp=0)
    classification_result_bunch['labels'] = labels
    classification_result_bunch['classification_results_df'] = classification_results_df
    classification_result_bunch['feature_importance_list'] = feature_importance_list
    classification_result_bunch['window_start_time'] = aligned_xarray[alignment_time_name].values
    classification_result_bunch['window_width'] = aligned_xarray.attrs['bin_width']


    # Calculate baseline accuracy from knowing the actual class distribution
    # ie. always predict the majority class
    baseline_hit_prop = compute_baseline_performance(y=labels, metric='accuracy')
    classification_results_df['baseline_hit_prop'] = baseline_hit_prop

    # classification_results_df['aligned_event'] = aligned_event

    classification_results_df['window_start_sec'] = classification_results_df[
        'window_start_locs'].apply(
        lambda x: aligned_xarray[peri_event_time_name].values[int(x)])

    classification_results_df['window_end_sec'] = classification_results_df[
        'window_end_locs'].apply(
        lambda x: aligned_xarray[peri_event_time_name].values[int(x)])

    classification_results_df['window_width_sec'] = classification_results_df['window_end_sec'] - \
                                                    classification_results_df['window_start_sec']

    if save_path is not None:
        with open(save_path, 'wb') as handle:
            pkl.dump(classification_result_bunch, handle)
    else:
        return classification_result_bunch


def calculate_optimal_choice_model_accuracy(behaviour_df, include_no_go=False):
    """
    Calculate highest level of accuracy possible if you know all the repsonse
    probabilities given the stimulus. This applies to decision classification only,
    which is either a binary classification of (1) left (2) right
    or a 3-class classification of (1) left (2) right (3) timeout (no-go)

    Arguments
    ----------
    behaviour_df (pandas dataframe)
        dataframe containing trial by trial information
    include_no_go (whether no-go is also classified)
        TODO: the code currently does not work with 3-class classification.

    Output
    ---------
    expected_correct_prop (float)
        the expected accuracy (from 0 to 1) that you can get if you know all the
        response probabilties given the stimulus (estimated using the entire session)
    """

    behaviour_df_subsetted = behaviour_df
    behaviour_df_subsetted['correctChoice'].fillna('Undefined', inplace=True)
    p_right_given_stimulus = behaviour_df_subsetted.groupby(
        ['correctChoice', 'visDiff', 'audDiff']).agg(['mean', 'count'])['goRight']

    total_trials = np.sum(p_right_given_stimulus['count'])
    assert np.sum(p_right_given_stimulus['count']) == len(behaviour_df_subsetted)
    expected_correct_classification = 0
    for prop_right, count in zip(p_right_given_stimulus['mean'], p_right_given_stimulus['count']):

        if prop_right >= 0.5:
            expected_correct_classification += prop_right * count
        else:
            expected_correct_classification += (1 - prop_right) * count

    expected_correct_prop = expected_correct_classification / total_trials

    return expected_correct_prop


# Ephys decoding functions

def make_subsampled_neurons(all_cell_alignment_data, target_brain_region='all',
                            min_neurons=30, random_seed=None,
                            reject_sub_threshold=True, custom_subsample_neurons=None,
                            verbose=False):
    """
    Subsample cells to make a fair comparison of decoding accuracy acaross brain regions.
    Number of subsample is the brain region with the smallest number of neurons.
    By default, brain regions with neurons fewer than min_neurons are removed.

    Parameters
    ----------
    all_cell_alignment_data: (xarray dataset)
        xarray dataset containing aligned neural activity with all brain regions
        should have dimensions: Cell, Exp, Time, Trial
    :param target_brain_region:
    :param min_neurons:
    :param random_seed:
    :param reject_sub_threshold:

    Return
    ---------
    subsampled_cells (xarray dataset)
        dataset with subsampled cells
        Should have dimensions: Cell, Exp, Time, Trials
    cell_count_dict (dict)
        key: cell location (str), eg. 'FRP', 'MOs', 'ORBvl'
        value : number of cells (int)
    """
    unique_cell_loc = np.unique(all_cell_alignment_data['CellLoc'].values)
    cell_count_dict = dict()
    for cell_loc in unique_cell_loc:
        cell_count_dict[cell_loc] = len(all_cell_alignment_data.where(
            all_cell_alignment_data['CellLoc'] == cell_loc, drop=True)['CellLoc'].values)

    if custom_subsample_neurons is not None:
        num_subsample = custom_subsample_neurons
        min_neurons = custom_subsample_neurons
    else:
        cell_counts = np.array(list(cell_count_dict.values()))
        if np.max(cell_counts) < min_neurons:
            print('None of the brain region in this experiment exceeds min_neurons, returning None')
            return None, cell_count_dict
        else:
            num_subsample = np.min(cell_counts[cell_counts >= min_neurons])

    if target_brain_region == 'all':
        target_brain_loc_alignemnt_data = all_cell_alignment_data
    else:
        target_brain_loc_alignemnt_data = all_cell_alignment_data.where(
            all_cell_alignment_data['CellLoc'] == target_brain_region, drop=True
        )

    if reject_sub_threshold:
        if len(target_brain_loc_alignemnt_data['Cell'].values) < min_neurons:
            print("""Warning: brain region has fewer cells than threshold,
            returning None for first argument""")
            return None, cell_count_dict

    if random_seed is not None:
        np.random.seed(seed=random_seed)

    subsample_index = np.random.choice(
        target_brain_loc_alignemnt_data.Cell.values,
        size=num_subsample, replace=False)

    if verbose:
        print('Shape of subsample index')
        print(np.shape(subsample_index))

    subsampled_cells = target_brain_loc_alignemnt_data.sel(Cell=subsample_index)

    return subsampled_cells, cell_count_dict


# Data loading functions

def load_classification_results(data_folder, subject_num, exp_num, brain_region,
                                aligned_to='movement'):
    """
    Loads classification result data.
    This is made for ephys prediction of movement (aligned to movement), but can in principle used
    to load other classification data as long as they have the same file name format.

    Parameters
    -----------
    data_folder: (str)
        path to the folder containing the classification results
    subject_num: (int)
        subject reference number
        TODO: make this accept arbitrary strings as well
    exp_num: (int)
        experiment reference number
    brain_region: (str)
        brain region
    aligned_to: (str)
    :return:
    """

    supported_alignment_types = ['movement', 'stimulus']
    assert aligned_to in supported_alignment_types, print('Alignment type unsupported.')

    subject_info = dict()
    subject_info['subject_num'] = subject_num
    subject_info['exp_num'] = exp_num

    subject_info_str = 'subject_' + str(subject_num) + '_exp_' + str(exp_num) + '_' + brain_region

    target_file_name_list = glob.glob(os.path.join(data_folder, subject_info_str + '*'
                                                   + aligned_to + '*.pkl'))

    if len(target_file_name_list) == 1:
        target_file_name = target_file_name_list[0]
    else:
        print('Warning: file not found or has multiple targets.')
        return None

    with open(target_file_name, 'rb') as handle:
        classification_result_bunch = pkl.load(handle)

    return classification_result_bunch


def read_and_combine_brain_region_classification_df(folder_path, subject_num, exp_num,
                                                    alignment_event='movementTimes', save_path=None,
                                                    print_matching=False):
    """
    Combines classification result df across multiple brain regions.
    Assume each df contains the classification result for that brain region (indepednent of other brain regions)

    Parameters
    -----------
    folder_path: (str)
    subject_num: (int)
    :param exp_num: (int)
    :param alignment_event: (str)
    :param save_path:
    :param print_matching:
    :return:
    """
    file_search_str = 'subject_' + str(subject_num) + '_exp_' + str(exp_num) + '*' + alignment_event + '*.pkl'
    matching_files = glob.glob(os.path.join(folder_path, file_search_str))

    if print_matching:
        print('Matched files:')
        print(matching_files)

    combined_df = combine_brain_region_classification_df(matching_files)

    if save_path is not None:
        with open(save_path, 'wb') as handle:
            pkl.dump(combined_df, handle)

    return combined_df


def combine_brain_region_classification_df(classification_data_file_path_list):
    combined_df_list = list()

    for classification_data_file_path in classification_data_file_path_list:
        classification_data = pd.read_pickle(classification_data_file_path)
        classification_data_df = classification_data['classification_results_df']
        # get brain region
        classification_data_df['brain_region'] = classification_data['subset_brain-region']

        # get peri-stimulus time
        classification_data_df['window_peri_stim_start'] = classification_data[
            'classification_results_df']['window_start_locs'].apply(
            lambda x: classification_data['features'].sel(Time=x).PeriStimTime.values)

        # note the -1: should be a temporary correction, the window should be +4 rather than +5
        classification_data_df['window_peri_stim_end'] = classification_data[
            'classification_results_df']['window_end_locs'].apply(
            lambda x: classification_data['features'].sel(Time=x - 1).PeriStimTime.values)

        # compute bias (only applies to two class classfication)
        if len(np.unique(classification_data['features']['responseMade'].values)) == 2:
            dominant_class_prop = sstats.mode(classification_data['features']['responseMade'].values).count[0] / \
                                  len(classification_data['features']['responseMade'].values)
            classification_data_df['dominant_class_prop'] = dominant_class_prop

        # get number of features (neurons)
        feature_xarray = classification_data['features']
        feature_xarray_brain_loc = feature_xarray.where(
            feature_xarray['CellLoc'] == classification_data['subset_brain-region'],
            drop=True)
        classification_data_df['num_neurons'] = len(feature_xarray_brain_loc['Cell'].values)


        combined_df_list.append(classification_data_df)

    combined_df = pd.concat(combined_df_list)

    return combined_df


def compute_baseline_performance(y, metric='accuracy'):
    """
    Computes baseline performance to be expected based on just knowing the labels (and nothing about the features)

    Parameters
    ----------
    y (numpy ndarray)
        target categorical variable to be decoded
    metric (str)
        how to calculate the performance metric
        options:
            'accuracy' : simple expected proportion of hits

    Returns
    -------

    """

    if y.ndim >= 2:
        print('Warning: label has more than one dimension, will try squeezing it.')
        y = np.squeeze(y)

    if metric == 'accuracy':
        mode_value, mode_count = sstats.mode(y)
        prop_most_common_class = mode_count[0] / len(y)
        baseline_performance = prop_most_common_class


    return baseline_performance


def compute_relative_score(decoding_result_df, classifier_score_name='classifier_score',
                           baseline_score_name='baseline_hit_prop', rel_score_name='relativeScore'):
    """
    Compute decoding accuracy relative to baseline.
    :param decoding_result_df:
    :return:
    """

    decoding_result_df[rel_score_name] = \
        (decoding_result_df[classifier_score_name] - decoding_result_df[baseline_score_name]) / \
        (1 - decoding_result_df[baseline_score_name])

    return decoding_result_df


def compute_window_mean_accuracy(decoding_results_df, window_units='bins',
                                 target_window_start_loc=20, use_is_close=False):

    mean_accuracy_df = decoding_results_df.groupby(['subject_num', 'exp_num',
                                                    'window_start_locs', 'window_end_locs']).agg(
                       np.mean).reset_index()

    if use_is_close:
        specific_window_decoding_df = mean_accuracy_df.loc[
            np.isclose(mean_accuracy_df['window_start_locs'], target_window_start_loc)
            ]
    else:
        specific_window_decoding_df = mean_accuracy_df.loc[
            mean_accuracy_df['window_start_locs'] == target_window_start_loc
            ]

    # get only the columns we need
    specific_window_decoding_df = specific_window_decoding_df[['subject_num', 'exp_num', 'relativeScore']]

    # make the name into standardised format
    specific_window_decoding_df.rename(columns={'subject_num': 'subjectRef',
                                                      'exp_num': 'expRef'}, inplace=True)
    specific_window_decoding_df['subjectRef'] = specific_window_decoding_df['subjectRef'].astype(int)
    specific_window_decoding_df['expRef'] = specific_window_decoding_df['expRef'].astype(int)

    return specific_window_decoding_df


def shuffle_alignment_ds(alignment_ds, variable_name='firing_rate', preserve_var='responseMade',
                         dim='Trial', check_shuffle=True):
    """

    Parameters
    ----------
    alignment_ds
    variable_name
    preserve_var
    dim
    check_shuffle

    Returns
    -------

    """

    num_time_bin = len(alignment_ds.Time.values)

    var_subset_alignment_ds_list = list()

    for p_var in np.unique(alignment_ds[preserve_var]):

        var_subset_alignment_ds = alignment_ds.where(alignment_ds[preserve_var] == p_var, drop=True)
        var_subset_alignment_ds_var = var_subset_alignment_ds[variable_name]

        shuffled_trial_by_time_ds_list = list()

        for cell_dim_val in var_subset_alignment_ds_var['Cell'].values:
            trial_by_time_ds = var_subset_alignment_ds_var.sel(Cell=cell_dim_val).values.T
            shuffled_trial_by_time_ds = np.random.permutation(trial_by_time_ds)

            # make sure that each time bin still have the same spikes across all trials
            og_spike_per_bin = np.sum(trial_by_time_ds, axis=0)
            shuffeld_spike_per_bin = np.sum(shuffled_trial_by_time_ds, axis=0)
            assert len(og_spike_per_bin) == num_time_bin
            assert np.all(np.isclose(og_spike_per_bin, shuffeld_spike_per_bin))
            shuffled_trial_by_time_ds_list.append(shuffled_trial_by_time_ds)

        shuffled_subset_alignment_ds_var = np.stack(shuffled_trial_by_time_ds_list, axis=-1)

        var_subset_alignment_ds = var_subset_alignment_ds.assign({'firing_rate_shuffled':
                                                                (['Trial', 'Time', 'Cell'],
                                                                shuffled_subset_alignment_ds_var)})

        var_subset_alignment_ds_list.append(var_subset_alignment_ds)

    shuffled_alignment_ds = xr.concat(var_subset_alignment_ds_list, dim='Trial')

    # transpose the shuffled firing rate to match the original firing rate dimension ordering
    shuffled_alignment_ds['firing_rate_shuffled'] = shuffled_alignment_ds['firing_rate_shuffled'].transpose('Cell',
                                                                                                            'Time',
                                                                                                            'Trial')

    if check_shuffle:
        ### Double check that the trial-averaged PSTH is the same
        # (just to make sure response made associated with each PSTH is preserve)

        left_choice_mean_psth = alignment_ds.where(alignment_ds['responseMade'] == 1,
                                                   drop=True).mean('Trial')['firing_rate']
        right_choice_mean_psth = alignment_ds.where(alignment_ds['responseMade'] == 2,
                                                    drop=True).mean('Trial')['firing_rate']

        shuffled_left_choice_mean_psth = shuffled_alignment_ds.where(shuffled_alignment_ds['responseMade'] == 1,
                                                                     drop=True).mean('Trial')['firing_rate']
        shuffled_right_choice_mean_psth = shuffled_alignment_ds.where(shuffled_alignment_ds['responseMade'] == 2,
                                                                      drop=True).mean('Trial')['firing_rate']

        assert np.all(np.isclose(left_choice_mean_psth, shuffled_left_choice_mean_psth))
        assert np.all(np.isclose(right_choice_mean_psth, shuffled_right_choice_mean_psth))

    return shuffled_alignment_ds


def balance_left_right_choice(alignment_ds, verbose=False, random_seed=None):
    """
    Subset left/right response (regardless of stimulus condition)
    Parameters
    ----------
    alignment_ds
    verbose

    Returns
    -------

    """

    left_choice_alignment_ds = alignment_ds.where(
        alignment_ds['responseMade'] == 1.0, drop=True)

    right_choice_alignment_ds = alignment_ds.where(
        alignment_ds['responseMade'] == 2.0, drop=True)

    if (len(left_choice_alignment_ds.Trial) == 0) | (len(right_choice_alignment_ds.Trial) == 0):
        print('No choice in at least one condition, returning None')
        return None
    elif len(left_choice_alignment_ds.Trial) > len(right_choice_alignment_ds.Trial):

        # subset left choices
        subset_idx = np.random.choice(np.arange(len(left_choice_alignment_ds.Trial)),
                                      size=len(right_choice_alignment_ds.Trial))

        subset_left_choice_alignment_ds = left_choice_alignment_ds.isel(Trial=subset_idx)
        subset_right_choice_alignment_ds = right_choice_alignment_ds  # no subsetting needed

    elif len(left_choice_alignment_ds.Trial) < len(right_choice_alignment_ds.Trial):

        # subset right choices
        subset_idx = np.random.choice(np.arange(len(left_choice_alignment_ds.Trial)),
                                      size=len(left_choice_alignment_ds.Trial))

        subset_right_choice_alignment_ds = right_choice_alignment_ds.isel(Trial=subset_idx)
        subset_left_choice_alignment_ds = left_choice_alignment_ds
    else:
        # equal number of trials, no need subsetting
        subset_right_choice_alignment_ds = right_choice_alignment_ds
        subset_left_choice_alignment_ds = left_choice_alignment_ds

    num_left_trials = len(subset_left_choice_alignment_ds.Trial)
    num_right_trials = len(subset_right_choice_alignment_ds.Trial)

    if num_left_trials != num_right_trials:
        pdb.set_trace()

    both_choice_alignment_ds = [subset_left_choice_alignment_ds, subset_right_choice_alignment_ds]

    if len(both_choice_alignment_ds) > 0:
        balanced_alignment_ds = xr.concat(both_choice_alignment_ds, dim='Trial')
    else:
        print('Warning: no trials found in alignment ds, returning None')
        balanced_alignment_ds = None

    return balanced_alignment_ds


def balance_left_right_choice_per_stim_cond(alignment_ds, verbose=False, random_seed=None):
    """
    Balance number of left and right choices per stimulus condition

    Parameters
    ----------
    alignment_ds : (xarray dataset)
    verbose : (bool)
        whether to print information about the trial obtained for each stimulus pair
    random_seed : (int)
        random seed to randomly choose trials from each stimulus condition
        this makes a certain set of random choice reproducible
    keep_trial_coord : (bool)
        whether to preserve trial number in the Trial dimension

    Returns
    -------

    """
    left_choice_alignment_ds_list = list()
    right_choice_alignment_ds_list = list()

    if random_seed is not None:
        np.random.seed(random_seed)

    unique_vis_diff = np.unique(alignment_ds['visDiff'])
    unique_aud_diff = np.unique(alignment_ds['audDiff'])

    balance_succesful = True

    for vis_diff, aud_diff in itertools.product(unique_vis_diff, unique_aud_diff):

        stim_cond_alignment_ds = alignment_ds.where(
            (alignment_ds['visDiff'] == vis_diff) &
            (alignment_ds['audDiff'] == aud_diff), drop=True
        )

        left_choice_stim_cond_alignment_ds = stim_cond_alignment_ds.where(
            stim_cond_alignment_ds['responseMade'] == 1.0, drop=True)

        right_choice_stim_cond_alignment_ds = stim_cond_alignment_ds.where(
            stim_cond_alignment_ds['responseMade'] == 2.0, drop=True)

        # pdb.set_trace()

        if verbose:
            print('Vis cond: %.2f, Aud cond: %.2f, Num trials: %.f' % (vis_diff, aud_diff,
                                                                       len(stim_cond_alignment_ds.Trial.values)))

        if (len(left_choice_stim_cond_alignment_ds.Trial) == 0) | (len(right_choice_stim_cond_alignment_ds.Trial) == 0):
            if verbose:
                print('No choice in at least one condition, skipping')
            continue
        elif len(left_choice_stim_cond_alignment_ds.Trial) > len(right_choice_stim_cond_alignment_ds.Trial):
            right_choice_alignment_ds_list.append(right_choice_stim_cond_alignment_ds)
            # subset left choices
            subset_idx = np.random.choice(np.arange(len(left_choice_stim_cond_alignment_ds.Trial)),
                                          size=len(right_choice_stim_cond_alignment_ds.Trial))

            left_choice_alignment_ds_list.append(left_choice_stim_cond_alignment_ds.isel(Trial=subset_idx))

        elif len(left_choice_stim_cond_alignment_ds.Trial) < len(right_choice_stim_cond_alignment_ds.Trial):
            left_choice_alignment_ds_list.append(left_choice_stim_cond_alignment_ds)
            # subset right choices
            subset_idx = np.random.choice(np.arange(len(right_choice_stim_cond_alignment_ds.Trial)),
                                          size=len(left_choice_stim_cond_alignment_ds.Trial))

            subset_right_choice_stim_cond_alignment_ds = right_choice_stim_cond_alignment_ds.isel(Trial=subset_idx)

            right_choice_alignment_ds_list.append(subset_right_choice_stim_cond_alignment_ds)

            num_left_trials = len(left_choice_stim_cond_alignment_ds.Trial)
            num_right_trials = len(subset_right_choice_stim_cond_alignment_ds.Trial)

            if num_left_trials != num_right_trials:
                pdb.set_trace()
        else:
            # same number of left and right choices, no need to subset
            left_choice_alignment_ds_list.append(left_choice_stim_cond_alignment_ds)
            right_choice_alignment_ds_list.append(right_choice_stim_cond_alignment_ds)

    # Re-assemble the datasets to one dataset
    both_choice_alignment_ds = left_choice_alignment_ds_list + right_choice_alignment_ds_list

    if len(both_choice_alignment_ds) > 0:
        subset_alignment_ds = xr.concat(both_choice_alignment_ds, dim='Trial')
    else:
        print('Warning: no trials found in alignment ds')
        subset_alignment_ds = alignment_ds
        balance_succesful = False

    # Doubel check that left and right choices are equal
    num_left_choice_trials = len(np.where(subset_alignment_ds.responseMade == 1)[0])
    num_right_choice_trials = len(np.where(subset_alignment_ds.responseMade == 2)[0])

    if num_left_choice_trials != num_right_choice_trials:
        print('Number of left and right choice trials are not balanced, something is wrong')
        print('Number of left choice trials: %.f' % num_left_choice_trials)
        print('Number of right choice trials: %.f' % num_right_choice_trials)
        balance_succesful = False

    return subset_alignment_ds, balance_succesful
