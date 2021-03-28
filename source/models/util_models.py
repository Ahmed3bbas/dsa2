# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""

"""
import logging, os, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split

####################################################################################################
VERBOSE = False

def log(*s):
    print(*s, flush=True)



####################################################################################################
def test_dataset_classifier_covtype(nrows=500):
    import wget
    # Dense features
    colnum = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",]

    # Sparse features
    colcat = ["Wilderness_Area1",  "Wilderness_Area2", "Wilderness_Area3",
        "Wilderness_Area4",  "Soil_Type1",  "Soil_Type2",  "Soil_Type3",
        "Soil_Type4",  "Soil_Type5",  "Soil_Type6",  "Soil_Type7",  "Soil_Type8",  "Soil_Type9",  ]

    # Target column
    coly        = ["Covertype"]

    log("start")
    global model, session

    root     = os.path.join(os.getcwd() ,"ztmp")
    BASE_DIR = Path.home().joinpath( root, 'data/input/covtype/')
    datafile = BASE_DIR.joinpath('covtype.data.gz')
    datafile.parent.mkdir(parents=True, exist_ok=True)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"

    # Download the dataset in case it's missing
    if not datafile.exists():
        wget.download(url, datafile.as_posix())

    # Read nrows of only the given columns
    feature_columns = colnum + colcat + coly
    df = pd.read_csv(datafile, header=None, names=feature_columns, nrows=nrows)
    #### Matching Big dict  ##################################################
    # X = df
    # y = df[coly].astype('uint8')
    return df, colnum, colcat, coly



def test_dataset_regress_fake(nrows=500):
    from sklearn import datasets as sklearn_datasets
    coly   = 'y'
    colnum = ["colnum_" +str(i) for i in range(0, 17) ]
    colcat = ['colcat_1']
    X, y    = sklearn_datasets.make_regression(
        n_samples=1000,
        n_features=17,
        n_targets=1,
        n_informative=17
    )
    df         = pd.DataFrame(X,  columns= colnum)
    df[coly]   = y.reshape(-1, 1)

    for ci in colcat :
      df[colcat] = np.random.randint(0,1, len(df))

    return df, colnum, colcat, coly




def test_dataset_classi_fake(nrows=500):
    from sklearn import datasets as sklearn_datasets
    ndim=11
    coly   = 'y'
    colnum = ["colnum_" +str(i) for i in range(0, ndim) ]
    colcat = ['colcat_1']
    X, y    = sklearn_datasets.make_classification(
        n_samples=1000,
        n_features=ndim,
        n_targets=1,
        n_informative=ndim
    )
    df         = pd.DataFrame(X,  columns= colnum)
    df[coly]   = y.reshape(-1, 1)

    for ci in colcat :
      df[colcat] = np.random.randint(0,1, len(df))

    return df, colnum, colcat, coly


def test_dataset_covtype(nrows=1000):

    # Dense features
    colnum = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am" , "Hillshade_Noon",  "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"]

    # Sparse features
    colcat = ["Wilderness_Area1",  "Wilderness_Area2", "Wilderness_Area3",
        "Wilderness_Area4",  "Soil_Type1",  "Soil_Type2",  "Soil_Type3", "Soil_Type4",  "Soil_Type5",  "Soil_Type6",  "Soil_Type7",  "Soil_Type8",  "Soil_Type9", "Soil_Type10",  "Soil_Type11",  "Soil_Type12",  "Soil_Type13",  "Soil_Type14", "Soil_Type15",  "Soil_Type16",  "Soil_Type17",  "Soil_Type18",  "Soil_Type19", "Soil_Type40",  ]

    # Target column
    coly        = ["Covertype"]

    log("start")
    global model, session

    root = os.path.join(os.getcwd() ,"ztmp")


    BASE_DIR = Path.home().joinpath( root, 'data/input/covtype/')
    datafile = BASE_DIR.joinpath('covtype.data.gz')
    datafile.parent.mkdir(parents=True, exist_ok=True)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"

    # Download the dataset in case it's missing
    if not datafile.exists():
        wget.download(url, datafile.as_posix())

    # Read nrows of only the given columns
    feature_columns = colnum + colcat + coly
    df = pd.read_csv(datafile, header=None, names=feature_columns, nrows=nrows)
    return df, colnum, colcat, coly



###################################################################################################################

def tf_data_create_sparse(cols_type_received:dict= {'cols_sparse' : ['col1', 'col2'],
                                                     'cols_num'    : ['cola', 'colb']

                                                     },
                           cols_ref:list=  [ 'col_sparse', 'col_num'  ], Xtrain:pd.DataFrame=None,
                           **kw):
    """

       Create sparse data struccture in KERAS  To plug with MODEL:
       No data, just virtual data
    https://github.com/GoogleCloudPlatform/data-science-on-gcp/blob/master/09_cloudml/flights_model_tf2.ipynb

    :return:
    """
    from tensorflow.feature_column import (categorical_column_with_hash_bucket,
        numeric_column, embedding_column, bucketized_column, crossed_column, indicator_column)

    ### Unique values :
    col_unique = {}

    if Xtrain is not None :
        for coli in cols_type_received['col_sparse'] :
                col_unique[coli] = int( Xtrain[coli].nunique())

    dict_cat_sparse, dict_dense = {}, {}
    for cols_groupname in cols_ref :
        assert cols_groupname in cols_type_received, "Error missing colgroup in config data_pars[cols_model_type] "

        if cols_groupname == "cols_sparse" :
           col_list = cols_type_received[cols_groupname]
           for coli in col_list :
               m_bucket = min(500, col_unique.get(coli, 500) )
               dict_cat_sparse[coli] = categorical_column_with_hash_bucket(coli, hash_bucket_size= m_bucket)

        if cols_groupname == "cols_dense" :
           col_list = cols_type_received[cols_groupname]
           for coli in col_list :
               dict_dense[coli] = numeric_column(coli)

        if cols_groupname == "cols_cross" :
           col_list = cols_type_received[cols_groupname]
           for coli in col_list :
               m_bucketi = min(500, col_unique.get(coli, 500) )
               m_bucketj = min(500, col_unique.get(coli, 500) )
               dict_cat_sparse[coli[0]+"-"+coli[1]] = crossed_column(coli[0], coli[1], m_bucketi * m_bucketj)

        if cols_groupname == "cols_discretize" :
           col_list = cols_type_received[cols_groupname]
           for coli in col_list :
               bucket_list = np.linspace(min, max, 100).tolist()
               dict_cat_sparse[coli +"_bin"] = bucketized_column(numeric_column(coli), bucket_list)


    #### one-hot encode the sparse columns
    dict_cat_sparse = { colname : indicator_column(col)  for colname, col in dict_cat_sparse.items()}

    ### Embed
    dict_cat_embed  = { 'em_{}'.format(colname) : embedding_column(col, 10) for colname, col in dict_cat_sparse.items()}


    #### TO Customisze
    #dict_dnn    = {**dict_cat_embed,  **dict_dense}
    # dict_linear = {**dict_cat_sparse, **dict_dense}

    return  dict_cat_sparse, dict_cat_embed, dict_dense,




def tf_data_pandas_to_dataset(training_df, colsX, coly):
    # tf.enable_eager_execution()
    # features = ['feature1', 'feature2', 'feature3']
    import tensorflow as tf
    print(training_df)
    training_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(training_df[colsX].values, tf.float32),
                tf.cast(training_df[coly].values, tf.int32)
            )
        )
    )

    for features_tensor, target_tensor in training_dataset:
        print(f'features:{features_tensor} target:{target_tensor}')
    return training_dataset



def tf_data_file_to_dataset(pattern, batch_size, mode=tf.estimator.ModeKeys.TRAIN, truncate=None):
    """  ACTUAL Data reading :
           Dataframe ---> TF Dataset  --> feed Keras model

    """
    import os, json, math, shutil
    import tensorflow as tf

    DATA_BUCKET = "gs://{}/flights/chapter8/output/".format(BUCKET)
    TRAIN_DATA_PATTERN = DATA_BUCKET + "train*"
    EVAL_DATA_PATTERN = DATA_BUCKET + "test*"

    CSV_COLUMNS  = ('ontime,dep_delay,taxiout,distance,avg_dep_delay,avg_arr_delay' + \
                    ',carrier,dep_lat,dep_lon,arr_lat,arr_lon,origin,dest').split(',')
    LABEL_COLUMN = 'ontime'
    DEFAULTS     = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],\
                    ['na'],[0.0],[0.0],[0.0],[0.0],['na'],['na']]

    def load_dataset(pattern, batch_size=1):
      return tf.data.experimental.make_csv_dataset(pattern, batch_size, CSV_COLUMNS, DEFAULTS)

    def features_and_labels(features):
      label = features.pop('ontime') # this is what we will train for
      return features, label

    dataset = load_dataset(pattern, batch_size)
    dataset = dataset.map(features_and_labels)

    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(batch_size*10)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(1)
    if truncate is not None:
        dataset = dataset.take(truncate)
    return dataset






def test_template(nrows=1000):
    """
        nrows : take first nrows from dataset
    """
    #### Regression PLEASE RANDOM VALUES AS TEST
    ### Fake Regression dataset
    df, colcat, colnum, coly = test_dataset_regress_fake()

    X = df[colcat+ colnum]
    y = df[coly]


    # Split the df into train/test subsets
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.05, random_state=2021, )#stratify=y) Regression no classes to stratify to
    X_train, X_valid, y_train, y_valid         = train_test_split(X_train_full, y_train_full, random_state=2021,)# stratify=y_train_full)
    # num_classes = len(set(y_train_full[coly].values.ravel()))
    log("X_train", X_train)


    cols_input_type_1 = []
    n_sample = 100
    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='boxcox')

    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='boxcox')


    m = {'model_pars': {
        ### LightGBM API model   #######################################
        # Specify the ModelConfig for pytorch_tabular
        'model_class':  ""

        # Type of target prediction, evaluation metrics
        ,'model_pars' : {'input_width': 18,  'y_width': 1 }


        , 'post_process_fun' : post_process_fun   ### After prediction  ##########################################
        , 'pre_process_pars' : {'y_norm_fun' :  pre_process_fun ,  ### Before training  ##########################

        ### Pipeline for data processing ##############################
        'pipe_list': [  #### coly target prorcessing
        {'uri': 'source/prepro.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },

        {'uri': 'source/prepro.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },
        {'uri': 'source/prepro.py::pd_colnum_binto_onehot',  'pars': {}, 'cols_family': 'colnum_bin', 'cols_out': 'colnum_onehot',  'type': ''             },

        #### catcol INTO integer,   colcat into OneHot
        {'uri': 'source/prepro.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },
        {'uri': 'source/prepro.py::pd_colcat_to_onehot',     'pars': {}, 'cols_family': 'colcat_bin', 'cols_out': 'colcat_onehot',  'type': ''             },

        ],
            }
        },

    'compute_pars': { 'metric_list': ['accuracy_score', 'median_absolute_error']
                    },

    'data_pars': { 'n_sample' : n_sample,

        'download_pars' : None,

        'cols_input_type' : cols_input_type_1,
        ### family of columns for MODEL  #########################################################
        'cols_model_group': [ 'colnum',
                              'colcat_binto_onehot',
                            ]

        ,'cols_model_group_custom' :  { 'colnum' : colnum,
                                        'colcat' : colcat,
                                        'coly' : coly
                                      }

        ###################################################
        ,'train': {'Xtrain': X_train,
                    'ytrain': y_train,
                        'Xtest': X_valid,
                        'ytest': y_valid},
                'eval': {'X': X_valid,
                        'y': y_valid},
                'predict': {'X': X_valid}

        ### Filter data rows   ##################################################################
        ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 },


        ### Added continuous & sparse features groups ###
        'cols_model_type2': {
            'colcontinuous':   colnum ,
            'colsparse' : colcat,
        },
        }
    }

    ##### Running loop
    ll = [
        'model_bayesian_pyro.py::BayesianRegression',
    ]
    for cfg in ll:

        # Set the ModelConfig
        m['model_pars']['model_class'] = cfg

        log('Setup model..')
        model = Model(model_pars=m['model_pars'], data_pars=m['data_pars'], compute_pars= m['compute_pars'] )

        log('\n\nTraining the model..')
        fit(data_pars=m['data_pars'], compute_pars= m['compute_pars'], out_pars=None)
        log('Training completed!\n\n')

        log('Predict data..')
        ypred, ypred_proba = predict(Xpred=None, data_pars=m['data_pars'], compute_pars=m['compute_pars'])
        log(f'Top 5 y_pred: {np.squeeze(ypred)[:5]}')


        log('Saving model..')
        save(path= "ztmp/data/output/torch_tabular")
        #  os.path.join(root, 'data\\output\\torch_tabular\\model'))

        log('Load model..')
        model, session = load_model(path="ztmp/data/output/torch_tabular")

        log('Model architecture:')
        log(model.model)

        reset()



if __name__ == "__main__":
    # import fire
    # fire.Fire()
    test()


