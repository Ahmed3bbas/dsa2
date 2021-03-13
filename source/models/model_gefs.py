# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""
"""
import os
import pandas as pd
import numpy as np
import sklearn

try :
  ### in repo/model_gefs/  
  from repo.model_gefs.gefs import RandomForest
except :
  os.system( " python -m pip install git+https://github.com/arita37/GeFs/GeFs.git@aa32d657013b7cacf62aaad912a9b88110cee5d1  -y ")
  from gefs import RandomForest


####################################################################################################
VERBOSE = True


# MODEL_URI = get_model_uri(__file__)

def log(*s):
    print(*s, flush=True)


####################################################################################################
global model, session


def init(*kw, **kwargs):
    global model, session
    model = Model(*kw, **kwargs)
    session = None


class Model(object):
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None):
        self.model_pars, self.compute_pars, self.data_pars = model_pars, compute_pars, data_pars

        if model_pars is None:
            self.model = None
        else:
            self.n_estimators = model_pars.get('n_estimators', 100)
            self.ncat         = model_pars.get('ncat', None)  # Number of categories of each variable This is an ndarray
            if self.ncat is None:
                self.model = None  # In order to create an instance of the model we need to calculate the ncat mentioned above on our dataset
                log('ncat is not define')
            else:
                self.model = RandomForest(n_estimators=self.n_estimators, ncat=self.ncat)
            if VERBOSE: log(None, self.model)

                
def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """
    global model, session
    session = None  # Session type for compute
    Xtrain, ytrain, Xtest, ytest = get_dataset(data_pars, task_type="train")
    if VERBOSE: log(Xtrain.shape, model.model)

    if model.ncat is None:
        log("#!IMPORTANT This indicates that the preprocessing pipeline was not adapted to GEFS! and we need to calculate ncat")
        cont_cols  = data_pars['cols_input_type'].get("colnum")  #  continous, float column is this correct?
        temp_train = pd.concat([Xtrain, ytrain], axis=1)
        temp_test  = pd.concat([Xtest, ytest],   axis=1)
        df         = pd.concat([temp_train, temp_test], ignore_index=True, sort=False)
        model.ncat = pd_colcat_get_catcount(df.values, classcol=-1,
                                            continuous_ids=[df.columns.get_loc(c) for c in cont_cols])

        model.model = RandomForest(model.n_estimators, ncat=model.ncat)

    model.model.fit(Xtrain, ytrain)
    model.model = model.model.topc()  # Convert to a GeF


def eval(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
       Return metrics of the model when fitted.
    """
    global model, session
    data_pars['train'] = True
    Xval, yval        = get_dataset(data_pars, task_type="eval")
    # ypred      = model.model.predict(Xval)
    ypred, ypred_prob = predict(Xval, data_pars, compute_pars, out_pars)

    # log(data_pars)
    mpars = compute_pars.get("metrics_pars", {'metric_name': 'auc'})

    scorer = {
        "auc": sklearn.metrics.roc_auc_score,
    }[mpars['metric_name']]

    mpars2 = mpars.get("metrics_pars", {})  ##Specific to score
    score_val = scorer(yval, ypred_prob, **mpars2)

    ddict = [{"metric_val": score_val, 'metric_name': mpars['metric_name']}]

    return ddict


def predict(Xpred=None, data_pars={}, compute_pars={}, out_pars={}, **kw):
    global model, session
    post_process_fun = model.model_pars.get('post_process_fun', None)
    if post_process_fun is None:
        def post_process_fun(y):
            return y

    if Xpred is None:
        data_pars['train'] = False
        Xpred              = get_dataset(data_pars, task_type="predict")

    ypred, y_prob = model.model.classify(Xpred, classcol = Xpred.shape[1], return_prob=True)
    ypred         = post_process_fun(ypred)
    y_prob        = np.max(y_prob, axis=1)
    
    ypred_proba = y_prob  if compute_pars.get("probability", False) else None

    return ypred, ypred_proba


def reset():
    global model, session
    model, session = None, None


def save(path=None, info=None):
    global model, session
    import cloudpickle as pickle
    os.makedirs(path, exist_ok=True)

    filename = "model.pkl"
    pickle.dump(model, open(f"{path}/{filename}", mode='wb'))  # , protocol=pickle.HIGHEST_PROTOCOL )

    filename = "info.pkl"
    pickle.dump(info, open(f"{path}/{filename}", mode='wb'))  # ,protocol=pickle.HIGHEST_PROTOCOL )


def load_model(path=""):
    global model, session
    import cloudpickle as pickle
    model0 = pickle.load(open(f"{path}/model.pkl", mode='rb'))

    model = Model()  # Empty model
    model.model = model0.model
    model.model_pars = model0.model_pars
    model.compute_pars = model0.compute_pars
    session = None
    return model, session


def load_info(path=""):
    import cloudpickle as pickle, glob
    dd = {}
    for fp in glob.glob(f"{path}/*.pkl"):
        if not "model.pkl" in fp:
            obj = pickle.load(open(fp, mode='rb'))
            key = fp.split("/")[-1]
            dd[key] = obj
    return dd


def preprocess(prepro_pars):
    if prepro_pars['type'] == 'test':
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split

        X, y = make_classification(n_features=10, n_redundant=0, n_informative=2,
                                   random_state=1, n_clusters_per_class=1)

        # log(X,y)
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
        return Xtrain, ytrain, Xtest, ytest

    if prepro_pars['type'] == 'train':
        from sklearn.model_selection import train_test_split
        df = pd.read_csv(prepro_pars['path'])
        dfX = df[prepro_pars['colX']]
        dfy = df[prepro_pars['coly']]
        Xtrain, Xtest, ytrain, ytest = train_test_split(dfX.values, dfy.values,
                                                        stratify=dfy.values, test_size=0.1)
        return Xtrain, ytrain, Xtest, ytest

    else:
        df = pd.read_csv(prepro_pars['path'])
        dfX = df[prepro_pars['colX']]

        Xtest, ytest = dfX, None
        return None, None, Xtest, ytest


####################################################################################################
############ Do not change #########################################################################
def get_dataset(data_pars=None, task_type="train", **kw):
    """
      "ram"  : 
      "file" :
    """
    # log(data_pars)
    data_type = data_pars.get('type', 'ram')
    if data_type == "ram":
        if task_type == "predict":
            d = data_pars[task_type]
            return d["X"]

        if task_type == "eval":
            d = data_pars[task_type]
            return d["X"], d["y"]

        if task_type == "train":
            d = data_pars[task_type]

            return d["Xtrain"], d["ytrain"], d["Xtest"], d["ytest"]

    elif data_type == "file":
        raise Exception(f' {data_type} data_type Not implemented ')

    raise Exception(f' Requires  Xtrain", "Xtest", "ytrain", "ytest" ')


def get_params_sklearn(deep=False):
    return model.model.get_params(deep=deep)


def get_params(param_pars={}, **kw):
    import json
    # from jsoncomment import JsonComment ; json = JsonComment()
    pp          = param_pars
    choice      = pp['choice']
    config_mode = pp['config_mode']
    data_path   = pp['data_path']

    if choice == "json":
        cf = json.load(open(data_path, mode='r'))
        cf = cf[config_mode]
        return cf['model_pars'], cf['data_pars'], cf['compute_pars'], cf['out_pars']

    else:
        raise Exception(f"Not support choice {choice} yet")

        
        
####################################################################################################        
############ Custom Code ############################################################################
def is_continuous(v_array):
    """
        Returns true if df was sampled from a continuous variables, and false
    """
    observed = v_array[~np.isnan(v_array)]  # not consider missing values for this.
    rules    = [np.min(observed) < 0,
                np.sum((observed) != np.round(observed)) > 0,
                len(np.unique(observed)) > min(30, len(observed) / 3)]
    if any(rules):
        return True
    else:
        return False


def pd_colcat_get_catcount(df, classcol=None, continuous_ids=[]):
    """
        Learns the number of categories in each variable and standardizes the df.
        Parameters
        ----------
        df: numpy n x m  Numpy array comprising n realisations (instances) of m variables.
        classcol: int   The column index of the class variables (if any).
        continuous_ids: list of ints
            List containing the indices of known continuous variables. Useful for
            discrete df like age, which is better modeled as continuous.
        Returns
        -------
        ncat: numpy m The number of categories of each variable. One if the variable is continuous.
    """
    df   = df.copy()
    ncat = np.ones(df.shape[1])
    if not classcol:
        classcol = df.shape[1] - 1
    for i in range(df.shape[1]):
        if i != classcol and (i in continuous_ids or is_continuous(df[:, i])):
            continue
        else:
            df[:, i] = df[:, i].astype(int)
            ncat[i] = max(df[:, i]) + 1
    return ncat

                                     
def test_model():
    # Auxiliary functions

    def get_stats(data, ncat=None):
        """
            Compute univariate statistics for continuous variables. Parameters
            ----------
            data: numpy n x m
                Numpy array comprising n realisations (instances) of m variables.
            Returns
            -------
            data: numpy n x m
                The normalized data.
            maxv, minv: numpy m
                The maximum and minimum values of each variable. One and zero, resp.
                if the variable is categorical.
            mean, std: numpy m
                The mean and standard deviation of the variable. Zero and one, resp.
                if the variable is categorical.
        """
        data = data.copy()
        maxv = np.ones(data.shape[1])
        minv = np.zeros(data.shape[1])
        mean = np.zeros(data.shape[1])
        std  = np.zeros(data.shape[1])
        if ncat is not None:
            for i in range(data.shape[1]):
                if ncat[i] == 1:
                    maxv[i] = np.max(data[:, i])
                    minv[i] = np.min(data[:, i])
                    mean[i] = np.mean(data[:, i])
                    std[i] = np.std(data[:, i])
                    assert maxv[i] != minv[i], 'Cannot have constant continuous variable in the data'
                    data[:, i] = (data[:, i] - minv[i]) / (maxv[i] - minv[i])
        else:
            for i in range(data.shape[1]):
                if is_continuous(data[:, i]):
                    maxv[i] = np.max(data[:, i])
                    minv[i] = np.min(data[:, i])
                    mean[i] = np.mean(data[:, i])
                    std[i]  = np.std(data[:, i])
                    assert maxv[i] != minv[i], 'Cannot have constant continuous variable in the data'
                    data[:, i] = (data[:, i] - minv[i]) / (maxv[i] - minv[i])
        return data, maxv, minv, mean, std

                                     
    def standardize_data(data, mean, std):
        """
            Standardizes the data given the mean and standard deviations values of
            each variable.
        """
        data = data.copy()
        for v in range(data.shape[1]):
            if std[v] > 0:
                data[:, v] = (data[:, v] - mean[v]) / (std[v])
                #  Clip values more than 6 standard deviations from the mean
                data[:, v] = np.clip(data[:, v], -6, 6)
        return data

                                     
    def train_test(data, ncat, train_ratio=0.7, prep='std'):
        assert train_ratio >= 0
        assert train_ratio <= 1
        shuffle    = np.random.choice(range(data.shape[0]), data.shape[0], replace=False)
        data_train = data[shuffle[:int(train_ratio * data.shape[0])], :]
        data_test  = data[shuffle[int(train_ratio * data.shape[0]):], :]
                                                    
        if prep == 'std':
            _, maxv, minv, mean, std = get_stats(data_train, ncat)
            data_train               = standardize_data(data_train, mean, std)
            X_train, y_train         = data_train[:, :-1], data_train[:, -1]
            return X_train, y_train, data_train, data_test, mean, std

                                     
    # Load toy dataset
    df_white   = pd.read_csv('../../data/input/wine-quality/raw/winequality-white.csv', sep=';').values
    ncat_white = pd_colcat_get_catcount(df_white, classcol=-1)
    ncat_white[-1] = 2

    X_train_white, y_train_white, data_train_white, data_test_white, mean_white, std_white = train_test(df_white,
                                                                                                        ncat_white, 0.7)
    y_train_white = np.where(y_train_white <= 6, 0, 1)
    
    model_pars = {
        'n_estimators':100,
        'ncat': ncat_white
    }
    model_white = Model(model_pars=model_pars)

    model_white.model.fit(X_train_white, y_train_white)
    gef_white = model_white.model.topc(learnspn=np.Inf)

    log('gefs model test ok')

                                     
if __name__ == "__main__":
    import fire
    fire.Fire()
                                     
"""
python model_gef.py test_model


"""
                                     
                                     
                                     
                                     
"""

    def learncats(data, classcol=None, continuous_ids=[]):
  
            Learns the number of categories in each variable and standardizes the data.
            Parameters
            ----------
            data: numpy n x m
                Numpy array comprising n realisations (instances) of m variables.
            classcol: int
                The column index of the class variables (if any).
            continuous_ids: list of ints
                List containing the indices of known continuous variables. Useful for
                discrete data like age, which is better modeled as continuous.
            Returns
            -------
            ncat: numpy m
                The number of categories of each variable. One if the variable is
                continuous.
      
        data = data.copy()
        ncat = np.ones(data.shape[1])
        if not classcol:
            classcol = data.shape[1] - 1
        for i in range(data.shape[1]):
            if i != classcol and (i in continuous_ids or is_continuous(data[:, i])):
                continue
            else:
                data[:, i] = data[:, i].astype(int)
                ncat[i] = max(data[:, i]) + 1
        return ncat


"""
