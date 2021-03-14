# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""
Genreate  train_data  ---> New train data by sampling

cd source/
python source/model_sampler.py test


Transformation for ALL Columns :
   Increase samples, Reduce Samples.

Main isssue is the number of rows change  !!!!
  cannot merge with others

  --> store as train data

  train data ---> new train data

  Transformation with less rows !

2 usage :
    Afte preprocessing, over sample, under-sample.


"""
import os, sys
sys.path.append( os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/")
### import util_feature


import os, pandas as pd, numpy as np,  sklearn
import copy
from sklearn.cluster import KMeans, DBSCAN, Birch
from sklearn.decomposition import TruncatedSVD, MiniBatchSparsePCA, FastICA

from umap import UMAP

try:
    #from sdv.demo import load_tabular_demo
    from sdv.tabular import TVAE, CTGAN
    from sdv.timeseries import PAR
    from sdv.evaluation import evaluate
    import ctgan
    
    if ctgan.__version__ != '0.3.1.dev0':
        raise Exception('ctgan outdated, updating...')
except:
    os.system("pip install sdv")
    os.system('pip install ctgan==0.3.1.dev0')
    from sdv.tabular import TVAE, CTGAN
    from sdv.timeseries import PAR
    from sdv.evaluation import evaluate  

### IMBLEARN
import six
import sys
sys.modules['sklearn.externals.six'] = six

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import NearMiss


### MMAE
try:
  from mmae.multimodal_autoencoder import MultimodalAutoencoder

except:
    os.system("pip install mmae[keras]")
####################################################################################################

# CONSTANTS
SDV_MODLES = ['TVAE', 'CTGAN', 'PAR'] # The Synthetic Data Vault Models
IMBLEARN_MODELS = ['SMOTE', 'SMOTEENN', 'SMOTETomek', 'NearMiss']
verbosity = 3

def log(*s):
    print(*s, flush=True)


def log2(*s):
    if verbosity >= 2 :
       print(*s, flush=True)


def log3(*s):
    if verbosity >= 3 :
       print(*s, flush=True)





####################################################################################################
def autoencoder_multimodal():
    """
    pip install mmae[keras]
    :return:
    """
    # Remove 'tensorflow.' from the next line if you use just Keras
    from tensorflow.keras.datasets import mnist
    from mmae.multimodal_autoencoder import MultimodalAutoencoder


    # Load example data
    (x_train, y_train), (x_validation, y_validation) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    y_train = y_train.astype('float32') / 255.0
    x_validation = x_validation.astype('float32') / 255.0
    y_validation = y_validation.astype('float32') / 255.0


    data = [x_train, y_train]
    validation_data = [x_validation, y_validation]

    # Set network parameters
    input_shapes = [x_train.shape[1:], (1,)]
    # Number of units of each layer of encoder network
    hidden_dims = [128, 64, 8]
    # Output activation functions for each modality
    output_activations = ['sigmoid', 'relu']

    optimizer = 'adam'
    # Loss functions corresponding to a noise model for each modality
    loss = ['bernoulli_divergence', 'poisson_divergence']
    # Construct autoencoder network
    autoencoder = MultimodalAutoencoder(input_shapes, hidden_dims,
                                        output_activations)
    autoencoder.compile(optimizer, loss)


    # Train model where input and output are the same
    autoencoder.fit(data, epochs=100, batch_size=256,
                    validation_data=validation_data)

    #To obtain a latent representation of the training data:
    latent_data = autoencoder.encode(data)

    #To decode the latent representation:
    reconstructed_data = autoencoder.decode(latent_data)

    #Encoding and decoding can also be merged into the following single statement:
    reconstructed_data = autoencoder.predict(data)




def autoEncoder(df):
    """"
    (4) Autoencoder
    An autoencoder is a type of artificial neural network used to learn efficient data codings in an unsupervised manner.
    The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for dimensionality reduction,
    by training the network to ignore noise.
    (i) Feed Forward
    The simplest form of an autoencoder is a feedforward, non-recurrent
    neural network similar to single layer perceptrons that participate in multilayer perceptrons
    """
    from sklearn.preprocessing import minmax_scale
    import tensorflow as tf
    import pandas as pd
    import numpy as np



    def encoder_dataset(df, drop=None, dimesions=20):
        # encode categorical columns
        cat_columns = df.select_dtypes(['category']).columns
        df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
        print(cat_columns)

        # encode objects columns
        from sklearn.preprocessing import OrdinalEncoder

        def encode_objects(X_train):
            oe = OrdinalEncoder()
            oe.fit(X_train)
            X_train_enc = oe.transform(X_train)
            return X_train_enc

        selected_cols = df.select_dtypes(['object']).columns
        df[selected_cols] = encode_objects(df[selected_cols])

        # df = df[[c for c in df.columns if c not in df.select_dtypes(['object']).columns]]
        if drop:
            train_scaled = minmax_scale(df.drop(drop,axis=1).values, axis = 0)
        else:
           train_scaled = minmax_scale(df.values, axis = 0)
        return train_scaled
    # define the number of encoding dimensions
    encoding_dim = pars.get('dimesions', 2)
    # define the number of features



    train_scaled = encoder_dataset(df, pars.get('drop',None), encoding_dim)

    print("train scaled: ", train_scaled)
    ncol = train_scaled.shape[1]


    input_dim = tf.keras.Input(shape = (ncol, ))
    # Encoder Layers
    encoded1      = tf.keras.layers.Dense(3000, activation = 'relu')(input_dim)
    encoded2      = tf.keras.layers.Dense(2750, activation = 'relu')(encoded1)
    encoded3      = tf.keras.layers.Dense(2500, activation = 'relu')(encoded2)
    encoded4      = tf.keras.layers.Dense(750, activation = 'relu')(encoded3)
    encoded5      = tf.keras.layers.Dense(500, activation = 'relu')(encoded4)
    encoded6      = tf.keras.layers.Dense(250, activation = 'relu')(encoded5)
    encoded7      = tf.keras.layers.Dense(encoding_dim, activation = 'relu')(encoded6)



    encoder       = tf.keras.Model(inputs = input_dim, outputs = encoded7)
    encoded_input = tf.keras.Input(shape = (encoding_dim, ))




    encoded_train = pd.DataFrame(encoder.predict(train_scaled),index=df.index)
    encoded_train = encoded_train.add_prefix('encoded_')
    if 'drop' in pars :
        drop = pars['drop']
        encoded_train = pd.concat((df[drop],encoded_train),axis=1)

    return encoded_train
    # df_out = mapper.encoder_dataset(df.copy(), ["Close_1"], 15); df_out.head()








####################################################################################################
global model, session

def init(*kw, **kwargs):
    global model, session
    model   = Model(*kw, **kwargs)
    session = None


####################################################################################################
class Model(object):
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None):
        self.model_pars, self.compute_pars, self.data_pars = model_pars, compute_pars, data_pars

        if model_pars is None:
            self.model = None
        else:
            model_class = globals()[model_pars['model_class']]
            self.model  = model_class(**model_pars['model_pars'])
            log2(model_class, self.model)



def fit(data_pars: dict=None, compute_pars: dict=None, out_pars: dict=None, **kw):
    """
    """
    global model, session
    session = None  # Session type for compute
    Xtrain_tuple, ytrain, Xtest_tuple, ytest = get_dataset(data_pars, task_type="train")

    cpars = copy.deepcopy(compute_pars.get("compute_pars", {}))

    if ytrain is not None and model.model_pars['model_class'] not in SDV_MODLES :  ###with label
       model.model.fit(Xtrain_tuple, ytrain, **cpars)
    else :
       model.model.fit(Xtrain_tuple, **cpars)


def eval(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
       Return metrics of the model when fitted.
    """
    global model, session
    from sdv.evaluation import evaluate

    data_pars['train'] = True
    Xval, yval         = get_dataset(data_pars, task_type="eval")

    if model.model_pars['model_class'] in IMBLEARN_MODELS:
        Xnew, ynew     = transform((Xval, yval), data_pars, compute_pars, out_pars)
    else:
        Xnew            = transform(Xval, data_pars, compute_pars, out_pars)
    
    # log(data_pars)
    mpars = compute_pars.get("metrics_pars", {'aggregate': True})

    if model.model_pars['model_class'] in SDV_MODLES:
        evals = evaluate(Xnew, Xval, **mpars )
        return evals
    else:
        return None


def transform(Xpred=None, data_pars={}, compute_pars={}, out_pars={}, **kw):
    """ Geenrate Xtrain  ----> Xtrain_new
    :param Xpred:
        Xpred ==> None            if you want to get generated samples by by SDV models
              ==> tuple of (x, y) if you want to resample dataset with IMBLEARN models
              ==> dataframe       if you want to transorm by sklearn models like TruncatedSVD
    :param data_pars:
    :param compute_pars:
    :param out_pars:
    :param kw:
    :return:
    """

    global model, session
    post_process_fun = model.model_pars.get('post_process_fun', None)
    if post_process_fun is None:
        def post_process_fun(y):
            return y

    #######
    if Xpred is None:
        if model.model_pars['model_class'] in IMBLEARN_MODELS:
            Xpred_tuple, y = get_dataset(data_pars, task_type="eval")

        else:
            Xpred_tuple = get_dataset(data_pars, task_type="predict")

    else :
        cols_type         = data_pars['cols_model_type2']
        cols_ref_formodel = cols_type
        split = kw.get("split", False)

        if model.model_pars['model_class'] in IMBLEARN_MODELS:
            if isinstance(Xpred, tuple) and len(Xpred) == 2:
                x, y = Xpred
                Xpred_tuple = get_dataset_tuple(x, cols_type, cols_ref_formodel, split)

            else:
                raise  Exception(f"IMBLEARN MODELS need to pass x, y to resample,you have to pass them as tuple => Xpred = (x, y)")

        else:
            Xpred_tuple       = get_dataset_tuple(Xpred, cols_type, cols_ref_formodel, split)


    Xnew= None
    if model.model_pars['model_class'] in SDV_MODLES :
       Xnew = model.model.sample(compute_pars.get('n_sample_generation', 100) )

    elif model.model_pars['model_class'] in IMBLEARN_MODELS :   ### Sampler
       # fit_resample(x ,y ) ==> resample dataset, it returns x,y after resampling
       # Xnew ==> tuple(x, y) after resmapling
       Xnew = model.model.fit_resample( Xpred_tuple, y, **compute_pars.get('compute_pars', {}) )


    else :
       Xnew = model.model.transform( Xpred_tuple, **compute_pars.get('compute_pars', {}) )

    log3("generated data", Xnew)
    return Xnew



def predict(Xpred=None, data_pars={}, compute_pars={}, out_pars={}, **kw):
    global model, session
    pass
    ### No need


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
    model.model        = model0.model
    model.model_pars   = model0.model_pars
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




####################################################################################################
############ Do not change #########################################################################
def get_dataset_tuple(Xtrain, cols_type_received, cols_ref, split=False):
    """  Split into Tuples = (df1, df2, df3) to feed model, (ie Keras)
    :param Xtrain:
    :param cols_type_received:
    :param cols_ref:
    :param split: 
        True :  split data to list of dataframe 
        False:  return same input of data
    :return:
    """
    if len(cols_ref) <= 1  or not split :
        return Xtrain
        
    Xtuple_train = []
    for cols_groupname in cols_ref :
        assert cols_groupname in cols_type_received, "Error missing colgroup in config data_pars[cols_model_type] "
        cols_i = cols_type_received[cols_groupname]
        Xtuple_train.append( Xtrain[cols_i] )

    #if len(cols_ref) == 1 :
    #    return Xtuple_train[0]  ### No tuple
    #else :
    return Xtuple_train



def get_dataset(data_pars=None, task_type="train", **kw):
    """
      return tuple of dataframes OR single dataframe
    """
    #### log(data_pars)
    data_type = data_pars.get('type', 'ram')

    ### Sparse columns, Dense Columns
    cols_type_received     = data_pars.get('cols_model_type2', {} )
    cols_ref  = list( cols_type_received.keys())
    split = kw.get('split', False)

    if data_type == "ram":
        # cols_ref_formodel = ['cols_cross_input', 'cols_deep_input', 'cols_deep_input' ]
        ### dict  colgroup ---> list of colname
        cols_type_received     = data_pars.get('cols_model_type2', {} )

        if task_type == "predict":
            d = data_pars[task_type]
            Xtrain       = d["X"]
            Xtuple_train = get_dataset_tuple(Xtrain, cols_type_received, cols_ref, split)
            return Xtuple_train

        if task_type == "eval":
            d = data_pars[task_type]
            Xtrain, ytrain  = d["X"], d["y"]
            Xtuple_train    = get_dataset_tuple(Xtrain, cols_type_received, cols_ref, split)
            return Xtuple_train, ytrain

        if task_type == "train":
            d = data_pars[task_type]
            Xtrain, ytrain, Xtest, ytest  = d["Xtrain"], d["ytrain"], d["Xtest"], d["ytest"]

            ### dict  colgroup ---> list of df
            Xtuple_train = get_dataset_tuple(Xtrain, cols_type_received, cols_ref, split)
            Xtuple_test  = get_dataset_tuple(Xtest, cols_type_received, cols_ref, split)

            log2("Xtuple_train", Xtuple_train)

            return Xtuple_train, ytrain, Xtuple_test, ytest

    elif data_type == "file":
        raise Exception(f' {data_type} data_type Not implemented ')

    raise Exception(f' Requires  Xtrain", "Xtest", "ytrain", "ytest" ')



def test():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_features=10, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
    X = pd.DataFrame( X, columns = [ 'col_' +str(i) for i in range(X.shape[1])] )
    y = pd.DataFrame( y, columns = ['coly'] )


    X['colid'] = np.arange(0, len(X))
    X_train, X_test, y_train, y_test    = train_test_split(X, y)
    X_train, X_valid, y_train, y_valid  = train_test_split(X_train, y_train, random_state=2021, stratify=y_train)

    #####
    colid  = 'colid'
    colnum = [ 'col_0', 'col_3', 'col_4', 'coly']
    colcat = [ 'col_1', 'col_7', 'col_8', 'col_9']
    cols_input_type_1 = {
        'colnum' : colnum,
        'colcat' : colcat
    }

    colg_input = {
      'cols_wide_input':   ['colnum', 'colcat' ],
      'cols_deep_input':   ['colnum', 'colcat' ],
    }

    cols_model_type2= {}
    for colg, colist in colg_input.items() :
        cols_model_type2[colg] = []
        for colg_i in colist :
          cols_model_type2[colg].extend( [i for i in cols_input_type_1[colg_i] if i not in y.columns ]   )
    ###############################################################################
    n_sample = 100
    data_pars = {'n_sample': n_sample,
                  'cols_input_type': cols_input_type_1,

                  'cols_model_group': ['colnum',
                                       'colcat',
                                       # 'colcross_pair'
                                       ],

                  'cols_model_type2' : cols_model_type2


        ### Filter data rows   #######################3############################
        , 'filter_pars': {'ymax': 2, 'ymin': -1}
                  }

    data_pars['train'] ={'Xtrain': X_train,  'ytrain': y_train,
                         'Xtest': X_test,  'ytest': y_test}
    data_pars['eval'] =  {'X': X_valid,
                          'y': y_valid}
    data_pars['predict'] = {'X': X_valid}

    compute_pars = { 'compute_pars' : { # 'epochs': 2,
                   } }

    #####################################################################
    # log("test 1")
    # model_pars = {'model_class': 'TruncatedSVD',
    #               'model_pars': {
    #                   "n_components": 3,
    #                   'n_iter': 2,
    #                  # 'ratio' :'auto',
    #             },
    #             }
    # test_helper(model_pars, data_pars, compute_pars)






    log("test Umap")






    log("test Umap 2")







    log("test Umap 3")





    log("test 5")







    
    # log("test 2 --> CTGAN")
    # model_pars = {'model_class': 'CTGAN',
    #               'model_pars': {
    #                  ## CTGAN
    #                  'primary_key': colid,
    #                  'epochs': 1,
    #                  'batch_size' :100,
    #                  'generator_dim' : (256, 256, 256),
    #                  'discriminator_dim' : (256, 256, 256)
    #             },
    #             }
    # compute_pars = { 
    #     'compute_pars' : {}
    #     }
    # test_helper(model_pars, data_pars, compute_pars)


    # log("test 3 --> TVAE")
    # model_pars = {'model_class': 'TVAE',
    #               'model_pars': { 
    #                   ## TVAE
    #                  'primary_key': colid,
    #                  'epochs': 1,
    #                  'batch_size' :100,
    #             },
    #             }
    # compute_pars = { 
    #     'compute_pars' : {}
    #     }
    # test_helper(model_pars, data_pars, compute_pars)

    # log("test 4 --> PAR")
    # entity_columns = [colid]
    # context_columns = None
    # sequence_index = None
    # model_pars = {'model_class': 'PAR',
    #               'model_pars': {
    #                  ## PAR
    #                  'epochs': 1,
    #                  'entity_columns': entity_columns,
    #                  'context_columns': context_columns,
    #                  'sequence_index': sequence_index
    #             },
    #             }
    # compute_pars = { 
    #     'compute_pars' : {}
    #     }
    # test_helper(model_pars, data_pars, compute_pars)

    log("test 6 --> SMOTE")
    model_pars = {'model_class': 'SMOTE',
                  'model_pars': {
                     ## SMOTE
                },
                }
    compute_pars = { 
        'compute_pars' : {}
        }
    test_helper(model_pars, data_pars, compute_pars)



def test_helper(model_pars, data_pars, compute_pars):
    global model, session
    root  = "ztmp/"
    model = Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)

    log('\n\nTraining the model')
    fit(data_pars=data_pars, compute_pars=compute_pars, out_pars=None)

    log('Predict data..')
    Xnew = transform(Xpred=None, data_pars=data_pars, compute_pars=compute_pars)
    log(f'Xnew', Xnew)

    log('Evaluating the model..')
    log(eval(data_pars=data_pars, compute_pars=compute_pars))

    log('Saving model..')
    save(path= root + '/model_dir/')

    log('Load model..')
    model, session = load_model(path= root + "/model_dir/")
    log(model)




if __name__ == "__main__":
    import fire
    fire.Fire()
    
    
    
















def pd_sample_imblearn(df=None, col=None, pars=None):
    """
        Over-sample
    """
    params_check(pars, ['model_name', 'pars_resample', 'coly']) # , 'dfy'
    prefix = '_sample_imblearn'

    ######################################################################################
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.under_sampling import NearMiss

    # model_resample = { 'SMOTE' : SMOTE, 'SMOTEENN': SMOTEENN }[  pars.get("model_name", 'SMOTEENN') ]
    model_resample = locals()[  pars.get("model_name", 'SMOTEENN')  ]
    pars_resample  = pars.get('pars_resample',
                             {'sampling_strategy' : 'auto', 'random_state':0}) # , 'n_jobs': 2

    if 'path_pipeline' in pars :   #### Inference time
        return df, {'col_new': col }

    else :     ### Training time
        colX    = col # [col_ for col_ in col if col_ not in coly]
        coly    = pars['coly']
        train_y = pars['dfy']  ## df[coly] #
        train_X = df[colX].fillna(method='ffill')
        gp      = model_resample( **pars_resample)
        X_resample, y_resample = gp.fit_resample(train_X, train_y)

        col_new   = [ t + f"_{prefix}" for t in col ]
        df2       = pd.DataFrame(X_resample, columns = col_new) # , index=train_X.index
        df2[coly] = y_resample

    ###################################################################################
    if 'path_features_store' in pars and 'path_pipeline_export' in pars:
       save_features(df2, prefix.replace("col_", "df_"), pars['path_features_store'])
       save(gp,             pars['path_pipeline_export'] + f"/{prefix}_model.pkl" )
       save(col,            pars['path_pipeline_export'] + f"/{prefix}.pkl" )
       save(pars_resample,  pars['path_pipeline_export'] + f"/{prefix}_pars.pkl" )


    col_pars = {'prefix' : prefix , 'path' :   pars.get('path_pipeline_export', pars.get('path_pipeline', None)) }
    col_pars['cols_new'] = {
       prefix :  col_new  ###  for training input data
    }
    return df2, col_pars







def pd_augmentation_sdv(df, col=None, pars={})  :
    '''
    Using SDV Variation Autoencoders, the function augments more data into the dataset
    params:
            df          : (pandas dataframe) original dataframe
            col : column name for data enancement
            pars        : (dict - optional) contains:
                n_samples     : (int - optional) number of samples you would like to add, defaul is 10%
                primary_key   : (String - optional) the primary key of dataframe
                aggregate  : (boolean - optional) if False, prints SVD metrics, else it averages them
                path_model_save: saving location if save_model is set to True
                path_model_load: saved model location to skip training
                path_data_new  : new data where saved
    returns:
            df_new      : (pandas dataframe) df with more augmented data
            col         : (list of strings) same columns
    '''
    n_samples       = pars.get('n_samples', max(1, int(len(df) * 0.10) ) )   ## Add 10% or 1 sample by default value
    primary_key     = pars.get('colid', None)  ### Custom can be created on the fly
    metrics_type    = pars.get('aggregate', False)
    path_model_save = pars.get('path_model_save', 'data/output/ztmp/')
    model_name      = pars.get('model_name', "TVAE")

    # model fitting
    if 'path_model_load' in pars:
            model = load(pars['path_model_load'])
    else:
            log('##### Training Started #####')

            model = {'TVAE' : TVAE, 'CTGAN' : CTGAN, 'PAR' : PAR}[model_name]
            if model_name == 'PAR':
                model = model(entity_columns = pars['entity_columns'],
                              context_columns = pars['context_columns'],
                              sequence_index = pars['sequence_index'])
            else:
                model = model(primary_key=primary_key)
            model.fit(df)
            log('##### Training Finshed #####')
            try:
                 save(model, path_model_save )
                 log('model saved at: ', path_model_save  )
            except:
                 log('saving model failed: ', path_model_save)

    log('##### Generating Samples #############')
    new_data = model.sample(n_samples)
    log_pd( new_data, n=7)


    log('######### Evaluation Results #########')
    if metrics_type == True:
      evals = evaluate(new_data, df, aggregate= True )
      log(evals)
    else:
      evals = evaluate(new_data, df, aggregate= False )
      log_pd(evals, n=7)

    # appending new data
    df_new = df.append(new_data)
    log(str(len(df_new) - len(df)) + ' new data added')

    if 'path_newdata' in pars :
        new_data.to_parquet( pars['path_newdata'] + '/features.parquet' )
        log('###### df augmentation save on disk', pars['path_newdata'] )

    log('###### augmentation complete ######')
    return df_new, col







    

# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""


"""
import warnings
warnings.filterwarnings('ignore')
import sys, gc, os, pandas as pd, json, copy, numpy as np

####################################################################################################
#### Add path for python import
sys.path.append( os.path.dirname(os.path.abspath(__file__)) + "/")

#### Root folder analysis
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)


#### Debuging state (Ture/False)
DEBUG_=True

####################################################################################################
####################################################################################################

def logs(*s):
    if DEBUG_:
        print(*s, flush=True)


def log_pd(df, *s, n=0, m=1):
    sjump = "\n" * m
    ### Implement pseudo Logging
    print(sjump,  df.head(n), flush=True)


from util_feature import  save, load_function_uri, load, save_features, params_check
####################################################################################################
####################################################################################################
def pd_export(df, col, pars):
    """
       Export in train folder for next training
       colsall
    :param df:
    :param col:
    :param pars:
    :return:
    """
    colid, colsX, coly = pars['colid'], pars['colsX'], pars['coly']
    dfX   = df[colsX]
    dfX   = dfX.set_index(colid)
    dfX.to_parquet( pars['path_export'] + "/features.parquet")


    dfy = df[coly]
    dfy = dfy.set_index(colid)
    dfX.to_parquet( pars['path_export'] + "/target.parquet")


###################################################################################################
##### Filtering / cleaning rows :   ###############################################################
def pd_filter_rows(df, col, pars):
    """
       Remove rows based on criteria
    :param df:
    :param col:
    :param pars:
    :return:
    """
    import re
    coly = col
    filter_pars =  pars
    def isfloat(x):
        #x = re.sub("[!@,#$+%*:()'-]", "", str(x))
        try :
            a= float(x)
            return 1
        except:
            return 0

    ymin, ymax = pars.get('ymin', -9999999999.0), filter_pars.get('ymax', 999999999.0)

    df['_isfloat'] = df[ coly ].apply(lambda x : isfloat(x),axis=1 )
    df = df[ df['_isfloat'] > 0 ]
    df = df[df[coly] > ymin]
    df = df[df[coly] < ymax]
    del df['_isfloat']
    return df, col




def pd_autoencoder(df, col, pars):
    """"
    (4) Autoencoder
    An autoencoder is a type of artificial neural network used to learn efficient data codings in an unsupervised manner.
    The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for dimensionality reduction,
    by training the network to ignore noise.
    (i) Feed Forward
    The simplest form of an autoencoder is a feedforward, non-recurrent
    neural network similar to single layer perceptrons that participate in multilayer perceptrons
    """
    from sklearn.preprocessing import minmax_scale
    import tensorflow as tf
    import pandas as pd
    import numpy as np
    def encoder_dataset(df, drop=None, dimesions=20):
        # encode categorical columns
        cat_columns = df.select_dtypes(['category']).columns
        df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
        print(cat_columns)

        # encode objects columns
        from sklearn.preprocessing import OrdinalEncoder

        def encode_objects(X_train):
            oe = OrdinalEncoder()
            oe.fit(X_train)
            X_train_enc = oe.transform(X_train)
            return X_train_enc

        selected_cols = df.select_dtypes(['object']).columns
        df[selected_cols] = encode_objects(df[selected_cols])

        # df = df[[c for c in df.columns if c not in df.select_dtypes(['object']).columns]]
        if drop:
            train_scaled = minmax_scale(df.drop(drop,axis=1).values, axis = 0)
        else:
           train_scaled = minmax_scale(df.values, axis = 0)
        return train_scaled
    # define the number of encoding dimensions
    encoding_dim = pars.get('dimesions', 2)
    # define the number of features
    train_scaled = encoder_dataset(df, pars.get('drop',None), encoding_dim)
    print("train scaled: ", train_scaled)
    ncol = train_scaled.shape[1]
    input_dim = tf.keras.Input(shape = (ncol, ))
    # Encoder Layers
    encoded1      = tf.keras.layers.Dense(3000, activation = 'relu')(input_dim)
    encoded2      = tf.keras.layers.Dense(2750, activation = 'relu')(encoded1)
    encoded3      = tf.keras.layers.Dense(2500, activation = 'relu')(encoded2)
    encoded4      = tf.keras.layers.Dense(750, activation = 'relu')(encoded3)
    encoded5      = tf.keras.layers.Dense(500, activation = 'relu')(encoded4)
    encoded6      = tf.keras.layers.Dense(250, activation = 'relu')(encoded5)
    encoded7      = tf.keras.layers.Dense(encoding_dim, activation = 'relu')(encoded6)
    encoder       = tf.keras.Model(inputs = input_dim, outputs = encoded7)
    encoded_input = tf.keras.Input(shape = (encoding_dim, ))
    encoded_train = pd.DataFrame(encoder.predict(train_scaled),index=df.index)
    encoded_train = encoded_train.add_prefix('encoded_')
    if 'drop' in pars :
        drop = pars['drop']
        encoded_train = pd.concat((df[drop],encoded_train),axis=1)

    return encoded_train
    # df_out = mapper.encoder_dataset(df.copy(), ["Close_1"], 15); df_out.head()




#####################################################################################
#####################################################################################
def test_pd_augmentation_sdv():
    from sklearn.datasets import load_boston
    data = load_boston()
    df   = pd.DataFrame(data.data, columns=data.feature_names)
    log_pd(df)

    dir_tmp = 'ztmp/'
    path = dir_tmp + '/model_par_augmentation.pkl'
    os.makedirs(dir_tmp, exist_ok=True)

    log('##### testing augmentation CTGAN ######################')
    pars = {'path_model_save': path,  'model_name': 'CTGAN'}
    df_new, _ = pd_augmentation_sdv(df, pars=pars)

    log('####### Reload')
    df_new, _ = pd_augmentation_sdv(df, pars={'path_model_load': path})


    log('##### testing augmentation VAE #########################')
    pars = {'path_model_save': path, 'model_name': 'VAE'}
    df_new, _ = pd_augmentation_sdv(df, pars=pars)
    log('####### Reload')
    df_new, _ = pd_augmentation_sdv(df, pars={'path_model_load': path})


    log('##### testing Time Series #############################')
    from sdv.demo import load_timeseries_demo
    df = load_timeseries_demo()
    log_pd(df)

    entity_columns  = ['Symbol']
    context_columns = ['MarketCap', 'Sector', 'Industry']
    sequence_index  = 'Date'

    pars = {'path_model_save': path,
            'model_name': 'PAR',
            'entity_columns' : entity_columns,
            'context_columns': context_columns,
            'sequence_index' : sequence_index,
            'n_samples' : 5}
    df_new, _ = pd_augmentation_sdv(df, pars=pars)

    log('####### Reload')
    df_new, _ = pd_augmentation_sdv(df, pars={'path_model_load': path,  'n_samples' : 5 })
    log_pd(df_new)



def pd_covariate_shift_adjustment():
    """
    https://towardsdatascience.com/understanding-dataset-shift-f2a5a262a766
     Covariate shift has been extensively studied in the literature, and a number of proposals to work under it have been published. Some of the most important ones include:
        Weighting the log-likelihood function (Shimodaira, 2000)
        Importance weighted cross-validation (Sugiyama et al, 2007 JMLR)
        Integrated optimization problem. Discriminative learning. (Bickel et al, 2009 JMRL)
        Kernel mean matching (Gretton et al., 2009)
        Adversarial search (Globerson et al, 2009)
        Frank-Wolfe algorithm (Wen et al., 2015)
    """
    import numpy as np
    from scipy import sparse
    import pylab as plt

    # .. to generate a synthetic dataset ..
    from sklearn import datasets
    n_samples, n_features = 1000, 10000
    A, b = datasets.make_regression(n_samples, n_features)
    def FW(alpha, max_iter=200, tol=1e-8):
        # .. initial estimate, could be any feasible point ..
        x_t = sparse.dok_matrix((n_features, 1))
        trace = []  # to keep track of the gap
        # .. some quantities can be precomputed ..
        Atb = A.T.dot(b)
        for it in range(max_iter):
            # .. compute gradient. Slightly more involved than usual because ..
            # .. of the use of sparse matrices ..
            Ax = x_t.T.dot(A.T).ravel()
            grad = (A.T.dot(Ax) - Atb)
            # .. the LMO results in a vector that is zero everywhere except for ..
            # .. a single index. Of this vector we only store its index and magnitude ..
            idx_oracle = np.argmax(np.abs(grad))
            mag_oracle = alpha * np.sign(-grad[idx_oracle])
            g_t = x_t.T.dot(grad).ravel() - grad[idx_oracle] * mag_oracle
            trace.append(g_t)
            if g_t <= tol:
                break
            q_t = A[:, idx_oracle] * mag_oracle - Ax
            step_size = min(q_t.dot(b - Ax) / q_t.dot(q_t), 1.)
            x_t = (1. - step_size) * x_t
            x_t[idx_oracle] = x_t[idx_oracle] + step_size * mag_oracle
        return x_t, np.array(trace)

    # .. plot evolution of FW gap ..
    sol, trace = FW(.5 * n_features)
    plt.plot(trace)
    plt.yscale('log')
    plt.xlabel('Number of iterations')
    plt.ylabel('FW gap')
    plt.title('FW on a Lasso problem')
    plt.grid()
    plt.show()
    sparsity = np.mean(sol.toarray().ravel() != 0)
    print('Sparsity of solution: %s%%' % (sparsity * 100))




########################################################################################
########################################################################################
def test():
    from util_feature import test_get_classification_data
    dfX, dfy = test_get_classification_data()
    cols     = list(dfX.columsn)
    ll       = [ ('pd_sample_imblearn', {}  )


               ]

    for fname, pars in ll :
        myfun = globals()[fname]
        res   = myfun(dfX, cols, pars)




    
    
"""


def pd_generic_transform(df, col=None, pars={}, model=None)  :
 
     Transform or Samples using  model.fit()   model.sample()  or model.transform()
    params:
            df    : (pandas dataframe) original dataframe
            col   : column name for data enancement
            pars  : (dict - optional) contains:                                          
                path_model_save: saving location if save_model is set to True
                path_model_load: saved model location to skip training
                path_data_new  : new data where saved 
    returns:
            model, df_new, col, pars
   
    path_model_save = pars.get('path_model_save', 'data/output/ztmp/')
    pars_model      = pars.get('pars_model', {} )
    model_method    = pars.get('method', 'transform')
    
    # model fitting 
    if 'path_model_load' in pars:
            model = load(pars['path_model_load'])
    else:
            log('##### Training Started #####')
            model = model( **pars_model)
            model.fit(df)
            log('##### Training Finshed #####')
            try:
                 save(model, path_model_save )
                 log('model saved at: ' + path_model_save  )
            except:
                 log('saving model failed: ', path_model_save)

    log('##### Generating Samples/transform #############')    
    if model_method == 'sample' :
        n_samples =pars.get('n_samples', max(1, 0.10 * len(df) ) )
        new_data  = model.sample(n_samples)
        
    elif model_method == 'transform' :
        new_data = model.transform(df.values)
    else :
        raise Exception("Unknown", model_method)
        
    log_pd( new_data, n=7)    
    if 'path_newdata' in pars :
        new_data.to_parquet( pars['path_newdata'] + '/features.parquet' ) 
        log('###### df transform save on disk', pars['path_newdata'] )    
    
    return model, df_new, col, pars



"""
    
    
    
