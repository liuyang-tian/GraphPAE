import numpy as np

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer


def ogbg_regression(train_emb, valid_emb, test_emb, train_y):

    base_classifier = Ridge(fit_intercept=True, copy_X=True, max_iter=10000)
    params_dict = {'alpha': [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]}

    classifier = make_pipeline(Normalizer(),
                GridSearchCV(base_classifier, params_dict, cv=5, scoring='neg_root_mean_squared_error', n_jobs=16, verbose=0))
    classifier.fit(train_emb, np.squeeze(train_y))

    train_pred = classifier.predict(train_emb)
    valid_pred = classifier.predict(valid_emb)
    test_pred  = classifier.predict(test_emb)

    return np.expand_dims(train_pred, axis=1), np.expand_dims(valid_pred, axis=1), np.expand_dims(test_pred, axis=1)


def ogbg_binary_classification(train_emb, valid_emb, test_emb, train_y):

    base_classifier = LogisticRegression(dual=False, fit_intercept=True, max_iter=10000)
    params_dict = {'C': [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]}

    classifier = make_pipeline(StandardScaler(), 
                GridSearchCV(base_classifier, params_dict, cv=5, scoring='roc_auc', n_jobs=16, verbose=0))
    classifier.fit(train_emb, np.squeeze(train_y))

    train_pred = classifier.predict_proba(train_emb)[:, 1]
    valid_pred = classifier.predict_proba(valid_emb)[:, 1]
    test_pred  = classifier.predict_proba(test_emb)[:, 1]

    return np.expand_dims(train_pred, axis=1), np.expand_dims(valid_pred, axis=1), np.expand_dims(test_pred, axis=1)


def ogbg_multi_binary_classification(train_emb, valid_emb, test_emb, train_y):
    
    base_classifier = LogisticRegression(dual=False, fit_intercept=True, max_iter=10000)
    params_dict = {'multioutputclassifier__estimator__C': [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]}

    if np.isnan(train_y).any():
        train_y = np.nan_to_num(train_y)

    #classifier = make_pipeline(StandardScaler(), MultiOutputClassifier(base_classifier, n_jobs=-1))
    pipe = make_pipeline(StandardScaler(), MultiOutputClassifier(base_classifier))
    classifier = GridSearchCV(pipe, params_dict, cv=5, scoring='roc_auc', n_jobs=16, verbose=0)
    classifier.fit(train_emb, train_y)

    train_pred = np.transpose([y_pred[:, 1] for y_pred in classifier.predict_proba(train_emb)])
    valid_pred = np.transpose([y_pred[:, 1] for y_pred in classifier.predict_proba(valid_emb)])
    test_pred  = np.transpose([y_pred[:, 1] for y_pred in classifier.predict_proba(test_emb)])

    return train_pred, valid_pred, test_pred


def ogbg_evaluation(emb, y, split_idx, evaluator, task_type, num_tasks):

    train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    train_emb, valid_emb, test_emb = emb[train_idx], emb[valid_idx], emb[test_idx]
    train_y,   valid_y,   test_y   = y[train_idx],   y[valid_idx],   y[test_idx]

    if 'regression' in task_type:

        metric = 'rmse'
        train_pred, valid_pred, test_pred = ogbg_regression(train_emb, valid_emb, test_emb, train_y)

    elif 'classification' in task_type:

        metric = 'rocauc'
        if num_tasks == 1:
            train_pred, valid_pred, test_pred = ogbg_binary_classification(train_emb, valid_emb, test_emb, train_y)
        else:
            train_pred, valid_pred, test_pred = ogbg_multi_binary_classification(train_emb, valid_emb, test_emb, train_y)

    else:
        raise NotImplementedError

    train_score = evaluator.eval({"y_true": train_y, "y_pred": train_pred})[metric]
    valid_score = evaluator.eval({"y_true": valid_y, "y_pred": valid_pred})[metric]
    test_score  = evaluator.eval({"y_true": test_y,  "y_pred": test_pred})[metric]

    return train_score, valid_score, test_score

