# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""


python model_vaem2.py test2



# # VAEM: a Deep Generative Model for Heterogeneous Mixed Type Data
# VAEM is an extension of variational autoencoders (VAEs) in order to handle such heterogeneous data. It is a deep generative model that is trained in a two stage manner. In the first stage we fit a different VAE independently to each data dimension $x_{nd}$. We call the resulting $D$ models marginal VAEs. Then, in the second stage, in order to capture the inter-variable dependencies, a new multi-dimensional VAE, called the dependency network, is build on top of the latent representations provided by the first-stage encoders. Finally, if the model is used in down stream tasks such as sequential active information acquisition, we often introduce a third stage, which is to add a new discriminator (preditor) model on top of the VAEM outputs. 
# 
# Different stages are referred as `1`,`2`, and `3` in `list_stages` in the `.json` files. 

# ## Usage

# To run the demo, you need to first download the [Bank Marketing UCI dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing), 
# and put the csv file under `data/bank`. You will need to preprocess the data into the format according to our example .csv 
# file (which does not contain any real data). This can done by splitting the text into columns using `;` as delimiters. Then,
# simply run `Main_Notebook.ipynb`. This notebook train/or load a VAEM model on Bank dataset, 
# and demonstrates how to perform sequential active information acquisition (SAIA) and imputation. 
# By default, it trains a new model on Bank dataset. If you would like to load a pre-trained model, by default it will load a pre-trained tensorflow model from `saved_weights/bank/`. Note that in order to perform 
# active information acquisition, an additional third stage training is required. This will add a discriminator (predictor)
# to the model, which is required for SAIA. The configurations for VAEM can be found in `.json` files in `hyperparameters/bank`, 
# which include:

# * "list_stage" : list of stages that you would like the model to be trained. stage 1 = training marginal VAEs, stage 2 = training dependency network,  stage 3 = add predictor and improve predictive performance. The default is [1,2]. 
# * "epochs" : number of epochs for training VAEM. If you would like to load a pretrained model rather than training a new one, you can simply set this to zero.
# * "latent_dim" : size of latent dimensions of dependency network, 
# * "p" : upper bound for artificial missingness probability. For example, if set to 0.9, then during each training epoch, the algorithm will randomly choose a probability smaller than 0.9, and randomly drops observations according to this probability. Our suggestion is that if original dataset already contains missing data, you can just set p to 0. 
# * "iteration" : iterations (number of mini batches) used per epoch. set to -1 to run the full epoch. If your dataset is large, please set to other values such as 10.
# * "batch_size" : iterations (number of mini batches) used per epoch. set to -1 to run the full epoch. If your dataset is large, please set to other values such as 10.
# * "K" : the dimension of the feature map (h) dimension of PNP encoder.
# * "M" : Number of MC samples when perform imputing. 
# * "repeat" : number of repeats.
# * "data_name" : name of the dataset being used. Our default is "bank".
# * "output_dir" : Directory where the model is stored. Our default is "./saved_weights/bank/",
# * "data_dir" : Directory where the data is stored. Our default is "./data/bank/",
# * "list_strategy" : list of strategies for active learning, 0 = random, 1 = single ordering. Default: [1]


"""
import numpy as np, sys,os, copy, random, pandas as pd, json

from scipy.stats import bernoulli
from random import sample
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
plt.switch_backend('agg')
# tfd = tf.contrib.distributions
import seaborn as sns; sns.set(style="ticks", color_codes=True)


### Need TF 1.4    pip install tensorflow=1.14
import tensorflow as tf
print(tf.__version__)




########################################################################################
sys.path.append( os.path.dirname(os.path.abspath(__file__)) + "/repo/VAEM/" )

import repo.VAEM.utils.process as process
import repo.VAEM.utils.params as params
import repo.VAEM.utils.active_learning as active_learning


#######################################################################################
verbosity =2

def log(*s):
    print(*s, flush=True)

def log2(*s):
    if verbosity >= 2 :
      print(*s, flush=True)


####################################################################################################
global model, session

def init(*kw, **kwargs):
    global model, session
    model = Model(*kw, **kwargs)
    session = None



####################################################################################################
##### Custom code  #################################################################################
cols_ref_formodel = ['none']  ### No column group

def load_data(): #(list_flt,list_cat,list_discrete,max_Data,min_Data,rs):
    args = params.Params('repo/VAEM/hyperparameters/bank_plot.json')

    if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    rs = 42 # random seed
    # the data will be mapped to interval [min_Data,max_Data]. Usually this will be [0,1] but you can also specify other values.
    max_Data = 0.7
    min_Data = 0.3
    list_discrete = np.array([8,9])

    # list of categorical variables
    list_cat = np.array([0,1,2,3,4,5,6,7])

    # list of numerical variables
    list_flt = np.array([8,9,10,11,12,13,14,15,16,17,18,19,20])
    #data_decode,records_d = load_data(list_flt,list_cat,list_discrete,max_Data,min_Data,rs)

    seed = 100
    bank_raw = pd.read_csv("repo/VAEM/data/bank/bank-additional.csv", sep=";")

    ### "age";"job";"marital";"education";"default";"balance";"housing";"loan";"contact";
    ### "day";"month";"duration";"campaign";"pdays";"previous";"poutcome";"y"
    """
     age    job marital education   default housing loan    contact month   day_of_week duration    campaign    pdays   previous    poutcome    emp.var.rate    cons.price.idx  cons.conf.idx   euribor3m   nr.employed y
     88  student married high.school unknown yes no  cellular    aug fri 43  7   999 1   success 1.4 93.444  -42.7   1.313   5228.1  yes
    """
    print(bank_raw.info())
    label_column="y"
    matrix1 = bank_raw.copy()

    matrix1 = matrix1.iloc[:5000,:]
    print(matrix1.shape, matrix1.columns)
    process.encode_catrtogrial_column(matrix1, ["job"])
    process.encode_catrtogrial_column(matrix1, ["marital"])
    process.encode_catrtogrial_column(matrix1, ["education"])
    process.encode_catrtogrial_column(matrix1, ["default"])
    process.encode_catrtogrial_column(matrix1, ["housing"])
    process.encode_catrtogrial_column(matrix1, ["loan"])
    process.encode_catrtogrial_column(matrix1, ["contact"])
    process.encode_catrtogrial_column(matrix1, ["month"])
    process.encode_catrtogrial_column(matrix1, ["day_of_week"])
    process.encode_catrtogrial_column(matrix1, ["poutcome"])
    process.encode_catrtogrial_column(matrix1, ["y"])

    Data = ((matrix1.values).astype(float))[0:,:]
    list_discrete_in_flt = (np.in1d(list_flt, list_discrete).nonzero()[0])
    list_discrete_comp   = list_discrete_in_flt + len(list_cat)

    if len(list_flt)>0 and len(list_cat)>0:
        list_var = np.concatenate((list_cat,list_flt))
    elif len(list_flt)>0:
        list_var = list_flt
    else:
        list_var = list_cat
    Data_sub = Data[:,list_var]
    dic_var_type = np.zeros(Data_sub.shape[1])
    dic_var_type[0:len(list_cat)] = 1

    # In this notebook we assume the raw data matrix is fully observed
    Mask = np.ones(Data_sub.shape)
    # Normalize/squash the data matrix
    Data_std       = (Data_sub - Data_sub.min(axis=0)) / (Data_sub.max(axis=0) - Data_sub.min(axis=0))
    scaling_factor = (Data_sub.max(axis=0) - Data_sub.min(axis=0))/(max_Data - min_Data)
    Data_sub       = Data_std * (max_Data - min_Data) + min_Data

    # decompress categorical data into one hot representation
    Data_cat  = Data[:,list_cat].copy()
    Data_flt  = Data[:,list_flt].copy()
    Data_comp = np.concatenate((Data_cat,Data_flt),axis = 1)
    Data_decomp, Mask_decomp, cat_dims, DIM_FLT = process.data_preprocess(Data_sub,Mask,dic_var_type)
    Data_train_decomp, Data_test_decomp, mask_train_decomp, mask_test_decomp,mask_train_comp, mask_test_comp,Data_train_comp, Data_test_comp = train_test_split(
            Data_decomp, Mask_decomp,Mask,Data_comp,test_size=0.1, random_state=rs)

    list_discrete = list_discrete_in_flt + (cat_dims.sum()).astype(int)

    Data_decomp = np.concatenate((Data_train_decomp, Data_test_decomp), axis=0)
    Data_train_orig = Data_train_decomp.copy()
    Data_test_orig = Data_test_decomp.copy()

    # Note that here we have added some noise to continuous-discrete variables to help training. Alternatively, you can also disable this by changing the noise ratio to 0.
    Data_noisy_decomp,records_d, intervals_d = process.noisy_transform(Data_decomp, list_discrete, noise_ratio = 0.99)
    noise_record                             = Data_noisy_decomp - Data_decomp
    Data_train_noisy_decomp                  = Data_noisy_decomp[0:Data_train_decomp.shape[0],:]
    Data_test_noisy_decomp                   = Data_noisy_decomp[Data_train_decomp.shape[0]:,:]
    
    data_decode = (Data_train_comp, Data_train_noisy_decomp,mask_train_decomp,Data_test_decomp,mask_test_comp,mask_test_decomp,cat_dims,DIM_FLT,dic_var_type)
    
    return data_decode,records_d



def encode2(data_decode,list_discrete,records_d,fast_plot):
    args = params.Params('repo/VAEM/hyperparameters/bank_plot.json')
    Data_train_decomp, Data_train_noisy_decomp,mask_train_decomp,Data_test_decomp,mask_test_comp,mask_test_decomp,cat_dims,DIM_FLT,dic_var_type = data_decode
    # vae = active_learning.p_vae_active_learning(Data_train_decomp, Data_train_noisy_decomp,mask_train_decomp,Data_test_decomp,mask_test_comp,mask_test_decomp,cat_dims,DIM_FLT,dic_var_type,args,list_discrete,records_d)

    model = p_vae_active_learning(Data_train_decomp,
                                  Data_train_noisy_decomp,
                                  mask_train_decomp,
                                  Data_test_decomp,
                                  mask_test_comp,
                                  mask_test_decomp,
                                  cat_dims,DIM_FLT,dic_var_type,
                                  args,list_discrete,records_d)

    tf.reset_default_graph()
    ### Impute missing data. Fthe mask to be zeros

    x_recon,z_posterior,x_recon_cat_p = model.get_imputation( Data_train_noisy_decomp, mask_train_decomp*0,cat_dims,dic_var_type) ## one hot already cpverted to integer


    x_real       = process.compress_data(Data_train_decomp,cat_dims, dic_var_type) ## x_real still needs conversion
    x_real_cat_p = Data_train_decomp[:,0:(cat_dims.sum()).astype(int)]

    # max_Data = 0.7
    # min_Data = 0.3
    Data_std = (x_real - x_real.min(axis=0)) / (x_real.max(axis=0) - x_real.min(axis=0))
    scaling_factor = (x_real.max(axis=0) - x_real.min(axis=0))/(max_Data - min_Data)
    Data_real = Data_std * (max_Data - min_Data) + min_Data
    sub_id = [1,2,10]

    if fast_plot ==1:
        Data_real = pd.DataFrame(Data_real[:,sub_id])
        g = sns.pairplot(Data_real.sample(min(1000,x_real.shape[0])),diag_kind = 'kde')
        g = g.map_diag(sns.distplot, bins = 50,norm_hist = True)
        g.set(xlim=(min_Data,max_Data), ylim = (min_Data,max_Data))
    else:
        Data_real = pd.DataFrame(Data_real[:,sub_id])
        g = sns.pairplot(Data_real.sample(min(10000,x_real.shape[0])),diag_kind = 'kde')
        g = g.map_diag(sns.distplot, bins = 50,norm_hist = True)
        g = g.map_upper(plt.scatter,marker='+')
        g = g.map_lower(sns.kdeplot, cmap="hot",shade=True,bw=.1)
        g.set(xlim=(min_Data,max_Data), ylim = (min_Data,max_Data))
    Data_fake_noisy = x_recon
    Data_fake       = process.invert_noise(Data_fake_noisy,list_discrete_comp,records_d)

    Data_std  = (Data_fake - x_real.min(axis=0)) / (x_real.max(axis=0) - x_real.min(axis=0))
    Data_fake = Data_std * (max_Data - min_Data) + min_Data
    sub_id    = [1,2,10]

    """
    if fast_plot ==1:
        g = sns.pairplot(pd.DataFrame(Data_fake[:,sub_id]).sample(min(1000,x_real.shape[0])),diag_kind = 'kde')
        g = g.map_diag(sns.distplot, bins = 50,norm_hist = True)
        g.set(xlim=(min_Data,max_Data), ylim = (min_Data,max_Data))
    else:
        g = sns.pairplot(pd.DataFrame(Data_fake[:,sub_id]).sample(min(1000,x_real.shape[0])),diag_kind = 'kde')
        g = g.map_diag(sns.distplot, bins = 50,norm_hist = True)
        g = g.map_upper(plt.scatter,marker='+')
        g = g.map_lower(sns.kdeplot, cmap="hot",shade=True,bw=.1)
        g.set(xlim=(min_Data,max_Data), ylim = (min_Data,max_Data))
    """
    return model, scaling_factor


def decode2(data_decode,scaling_factor,list_discrete,records_d):
    args = params.Params('repo/VAEM/hyperparameters/bank_SAIA.json')
    Data_train_comp, Data_train_noisy_decomp,mask_train_decomp,Data_test_decomp,mask_test_comp,mask_test_decomp,cat_dims,DIM_FLT,dic_var_type = data_decode

    # vae = active_learning.p_vae_active_learning(Data_train_comp, Data_train_noisy_decomp,
    vae = p_vae_active_learning(Data_train_comp, Data_train_noisy_decomp,
        mask_train_decomp, Data_test_decomp,
        mask_test_comp,mask_test_decomp,
        cat_dims,
        DIM_FLT,dic_var_type,
        args,list_discrete,records_d)


    """
    npzfile = np.load(args.output_dir+'/UCI_rmse_curve_SING.npz')
    IC_SING=npzfile['information_curve']*scaling_factor[-1]
    plt.figure(0)
    L = IC_SING.shape[1]
    fig, ax1 = plt.subplots()
    # These are in unitless percentages of the figure size. (0,0 is bottom left)
    left, bottom, width, height = [0.45, 0.4, 0.45, 0.45]
    ax1.plot(np.sqrt((IC_SING[:,:,0:]**2).mean(axis=1)).mean(axis=0),'ys',linestyle = '-.', label = 'VAEM-Bank dataset')
    ax1.errorbar(np.arange(IC_SING.shape[2]),np.sqrt((IC_SING[:,:,0:]**2).mean(axis=1)).mean(axis=0), yerr=np.sqrt((IC_SING[:,:,0:]**2).mean(axis=1)).std(axis = 0)/np.sqrt(IC_SING.shape[0]),ecolor='y',fmt = 'ys')
    plt.xlabel('Steps',fontsize=18)
    plt.ylabel('avg. test. RMSE',fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(bbox_to_anchor=(0.0, 1.02, 1., .102), mode = "expand", loc=3,
               ncol=1, borderaxespad=0.,prop={'size': 20}, frameon=False)
    ax1.ticklabel_format(useOffset=False)
    plt.show()
    """
    return vae


def save_model2(model,output_dir):
    model.save(output_dir)



##### Global Variables  ########################################################
rs = 42 # random seed
# the data will be mapped to interval [min_Data,max_Data]. Usually this will be [0,1] but you can also specify other values.
max_Data = 0.7
min_Data = 0.3
list_discrete = np.array([8,9])

# list of categorical variables
list_cat = np.array([0,1,2,3,4,5,6,7])

# list of numerical variables
list_flt = np.array([8,9,10,11,12,13,14,15,16,17,18,19,20])
# data_decode,records_d = load_data(list_flt,list_cat,list_discrete,max_Data,min_Data,rs)


def test2(fast_plot=0):
    """
    if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    rs = 42 # random seed
    # the data will be mapped to interval [min_Data,max_Data]. Usually this will be [0,1] but you can also specify other values.
    max_Data = 0.7
    min_Data = 0.3
    list_discrete = np.array([8,9])

    # list of categorical variables
    list_cat = np.array([0,1,2,3,4,5,6,7])
    # list of numerical variables
    list_flt = np.array([8,9,10,11,12,13,14,15,16,17,18,19,20])
    data_decode,records_d = load_data(list_flt,list_cat,list_discrete,max_Data,min_Data,rs)
    """
    args                  = params.Params('repo/VAEM/hyperparameters/bank_plot.json')
    if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)


    data_decode,records_d = load_data()
    Data_train_comp, Data_train_noisy_decomp,mask_train_decomp,Data_test_decomp,mask_test_comp,mask_test_decomp,cat_dims,DIM_FLT,dic_var_type = data_decode

    model_encode,scaling_factor = encode2(data_decode,list_discrete,records_d,fast_plot)
    model_decode                = decode2(data_decode, scaling_factor,list_discrete,records_d)

    save_model2(model, args.output_dir)






###############################################################################################
###############################################################################################
import sys, os, numpy as np, random
import tensorflow as tf
sys.path.append( os.path.dirname(os.path.abspath(__file__)) + "/")


import repo.VAEM.models.decoders as decoders
import repo.VAEM.models.encoders as encoders
import repo.VAEM.models.model as model
import repo.VAEM.utils.reward as reward

def p_vae_active_learning(Data_train_compressed, Data_train,mask_train,Data_test,mask_test_compressed,mask_test,cat_dims,dim_flt,dic_var_type,args,list_discrete,records_d, estimation_method=1):
    
    list_stage = args.list_stage
    list_strategy = args.list_strategy
    epochs = 1
    latent_dim = args.latent_dim
    batch_size = args.batch_size
    p = args.p
    K = args.K
    M = args.M
    Repeat = args.repeat
    iteration = args. iteration
    
    '''
    This function train or loads a VAEM model, and performs SAIA using SING or full EDDI strategy.
    Note that we assume that the last column of x is the target variable of interest
    :param Data_train_compressed: preprocessed traning data matrix without one-hot encodings. Note that we assume that the columns of the data matrix is re-ordered, so that the categorical variables appears first, and then the continuous variables afterwards.
    :param Data_train: preprocessed traning data matrix with one-hot encodings. Is is re-ordered as Data_train_compressed.
    :param mask_train: mask matrix that indicates the missingness of training data,with one-hot encodings. 1=observed, 0 = missing
    :param Data_test: test data matrix, with one-hot encodings.
    :param mask_test: mask matrix that indicates the missingness of test data, with one-hot encodings.. 1=observed, 0 = missing
    :param mask_test_compressed: mask matrix that indicates the missingness of test data, without one-hot encodings. 1=observed, 0 = missing.
    :param cat_dims: a list that indicates the number of potential outcomes for non-continuous variables.
    :param dim_flt: number of continuous variables.
    :param dic_var_type: a list that contains the statistical types for each variables
    :param args.epochs: number of epochs for training.
    :param args.latent_dim: latent dimension of partial VAE.
    :param args.K: dimension of feature map of PNP encoder
    :param args.M: number of samples used for MC sampling
    :param args.Repeat: number of repeats.
    :param estimation_method: what method to use for single ordering information reward estimation.
            In order to calculate the single best ordering, we need to somehow marginalize (average) the
            information reward over the data set (in this case, the test set).
            we provide two methods of marginalization.
            - estimation_method = 0: information reward marginalized using the model distribution p_{vae_model}(x_o).
            - estimation_method = 1: information reward marginalized using the data distribution p_{data}(x_o)
    :param args.list_stage: a list of stages that you wish to perform training. 1 = train marginal VAEs, 2 = train dependency VAEs, 3 = fine-tune, 4 = load a model without training.
    :param args.list_strategy: a list of strategies that is used for SAIA. 0 = Random, 1 = SING
    
    :return: None (active learning results are saved to args.output_dir)
    '''
    n_test = Data_test.shape[0]
    n_train = Data_train.shape[0]
    dim_cat = len(np.argwhere(cat_dims != -1))
    OBS_DIM = dim_cat + dim_flt
    al_steps = OBS_DIM
    # information curves
    information_curve_RAND = np.zeros(
        (Repeat, n_test, al_steps - 1 + 1))
    information_curve_SING = np.zeros(
        (Repeat, n_test, al_steps - 1 + 1))
    rmse_curve_RAND = np.zeros(
        (Repeat, n_test, al_steps - 1 + 1))
    rmse_curve_SING = np.zeros(
        (Repeat, n_test, al_steps - 1 + 1))
    # history of optimal actions
    action_SING = np.zeros((Repeat, n_test,
                            al_steps - 1))
    # history of information reward values
    R_hist_SING = np.zeros(
        (Repeat, al_steps - 1, n_test,
         OBS_DIM - 1))

    for r in range(Repeat):
        ## train partial VAE
        reward_estimation = reward.lindley(M, cat_dims, list_discrete, dic_var_type, records_d,)
        tf.reset_default_graph()
        for stag in range(len(list_stage)):
            stage = list_stage[stag]
            vae = train_p_vae(stage, Data_train, Data_train,mask_train, epochs, latent_dim,cat_dims,dim_flt,batch_size, p, K,iteration,list_discrete,records_d,args)  

        ## Perform active variable selection
        if len(list_strategy)>0:
            for strat in range(len(list_strategy)):
                strategy = list_strategy[strat]
                if strategy == 0:### random strategy
                    ## create arrays to store data and missingness
                    x = Data_test[:, :]  #
    #                 x = np.reshape(x, [n_test, OBS_DIM])
                    mask = np.zeros((n_test, OBS_DIM))
                    mask[:, -1] = 0  # we will never observe target value

                    ## initialize array that stores optimal actions (i_optimal)
                    i_optimal = [ nums for nums in range(OBS_DIM - 1 ) ]
                    i_optimal = np.tile(i_optimal, [n_test, 1])
                    random.shuffle([random.shuffle(c) for c in i_optimal])

                    ## evaluate likelihood and rmse at initial stage (no observation)
                    negative_predictive_llh, predictive_rmse = vae.predictive_loss(
                        x, mask,cat_dims, dic_var_type, M)
                    information_curve_RAND[r, :, 0] = negative_predictive_llh
                    rmse_curve_RAND[r, :, 0] = predictive_rmse
                    for t in range(al_steps - 1 ):
                        print("Repeat = {:.1f}".format(r))
                        print("Strategy = {:.1f}".format(strategy))
                        print("Step = {:.1f}".format(t))
                        io = np.eye(OBS_DIM)[i_optimal[:, t]]
                        mask = mask + io
                        negative_predictive_llh, predictive_rmse = vae.predictive_loss(
                            x, mask, cat_dims, dic_var_type,M)
                        information_curve_RAND[r, :, t +1] = negative_predictive_llh
                        rmse_curve_RAND[r, :, t+1] = predictive_rmse
                    np.savez(os.path.join(args.output_dir, 'UCI_information_curve_RAND.npz'),
                             information_curve=information_curve_RAND)
                    np.savez(os.path.join(args.output_dir, 'UCI_rmse_curve_RAND.npz'),
                             information_curve=rmse_curve_RAND)

                if strategy == 1:### single ordering strategy
                    im_SING = np.zeros((Repeat, al_steps - 1 , M,
                                    n_test, Data_train.shape[1] ))
                    #SING is obtrained by maximize mean information reward for each step for the test set to be consistant with the description in the paper.
                    #We can also get this order by using a subset of training set to obtain the optimal ordering and apply this to the testset.
                    x = Data_test[:, :]  #
#                     x,_, _ = noisy_transform(x, list_discrete, noise_ratio)
    #                 x = np.reshape(x, [n_test, OBS_DIM])
                    mask = np.zeros((n_test, OBS_DIM)) # this stores the mask of missingness (stems from both test data missingness and unselected features during active learing)
                    mask2 = np.zeros((n_test, OBS_DIM)) # this stores the mask indicating that which features has been selected of each data
                    mask[:, -1] = 0  # Note that no matter how you initialize mask, we always keep the target variable (last column) unobserved.
                    negative_predictive_llh, predictive_rmse = vae.predictive_loss(
                        x, mask,cat_dims, dic_var_type, M)
                    information_curve_SING[r, :, 0] = negative_predictive_llh
                    rmse_curve_SING[r, :, 0] = predictive_rmse

                    for t in range(al_steps - 1 ): # t is a indicator of step
                        print("Repeat = {:.1f}".format(r))
                        print("Strategy = {:.1f}".format(strategy))
                        print("Step = {:.1f}".format(t))
                        ## note that for single ordering, there are two rewards.
                        # The first one (R) is calculated based on no observations.
                        # This is used for active learning phase, since single ordering should not depend on observations.
                        # The second one (R_eval) is calculated in the same way as chain rule approximation. This is only used for visualization.
                        if t ==-1:
                            im_0 = Data_train_compressed.reshape((1,Data_train_compressed.shape[0],-1))
                            im = Data_train_compressed.reshape((1,Data_train_compressed.shape[0],-1))
                            R = -1e40 * np.ones((Data_train_compressed.shape[0], OBS_DIM - 1))
                            for u in range(OBS_DIM - 1): # u is the indicator for features. calculate reward function for each feature candidates
                                loc = np.where(mask2[:, u] == 0)[0]
                                if estimation_method == 0:
                                    R[loc, u] = reward_estimation.R_lindley_chain(u, x, mask,  vae, im_0,loc)
                                else:
                                    R[loc, u] = reward_estimation.R_lindley_chain(u, x, mask, vae, im, loc)
                        else:
                            R = -1e40 * np.ones((n_test, OBS_DIM - 1))
                            im_0 = reward_estimation.completion(x, mask*0,vae) # sample from model prior
                            im = reward_estimation.completion(x, mask, vae) # sample conditional on observations
                            im_SING[r, t, :, :, :] = im
                            for u in range(OBS_DIM - 1): # u is the indicator for features. calculate reward function for each feature candidates
                                loc = np.where(mask2[:, u] == 0)[0]
                                if estimation_method == 0:
                                    R[loc, u] = reward_estimation.R_lindley_chain(u, x, mask,  vae, im_0,loc)
                                else:
                                    R[loc, u] = reward_estimation.R_lindley_chain(u, x, mask, vae, im, loc)
                            R_hist_SING[r, t, :, :] = R
                        
                        i_optimal = (R.mean(axis=0)).argmax() # optimal decision based on reward averaged on all data
                        i_optimal = np.tile(i_optimal, [n_test])
                        io = np.eye(OBS_DIM)[i_optimal]
                        action_SING[r, :, t] = i_optimal
                        mask = mask + io*mask_test_compressed # this mask takes into account both data missingness and missingness of unselected features
                        negative_predictive_llh, predictive_rmse = vae.predictive_loss(
                            x, mask,cat_dims, dic_var_type, M)
                        mask2 = mask2 + io # this mask only stores missingess of unselected features, i.e., which features has been selected of each data
                        information_curve_SING[r, :, t +
                                               1] = negative_predictive_llh
                        rmse_curve_SING[r, :, t+1] = predictive_rmse
                    np.savez(os.path.join(args.output_dir, 'UCI_information_curve_SING.npz'),
                             information_curve=information_curve_SING)
                    np.savez(os.path.join(args.output_dir, 'UCI_rmse_curve_SING.npz'),
                             information_curve=rmse_curve_SING)
                    np.savez(os.path.join(args.output_dir, 'UCI_action_SING.npz'), action=action_SING)
                    np.savez(os.path.join(args.output_dir, 'UCI_R_hist_SING.npz'), R_hist=R_hist_SING)

#             Save results
    return vae



def train_p_vae(stage, x_train, Data_train,mask_train, epochs, latent_dim,cat_dims,dim_flt,batch_size, p, K,iteration,list_discrete,records_d,args):
    '''
        This function trains the partial VAE.
        :param stage: stage of training 
        :param x_train: initial inducing points
        :param Data_train: training Data matrix, N by D
        :param mask_train: mask matrix that indicates the missingness. 1=observed, 0 = missing
        :param epochs: number of epochs of training
        :param LATENT_DIM: latent dimension for partial VAE model
        :param cat_dims: a list that indicates the number of potential outcomes for non-continuous variables.
        :param dim_flt: number of continuous variables.
        :param batch_size: batch_size.
        :param p: dropout rate for creating additional missingness during training
        :param K: dimension of feature map of PNP encoder
        :param iteration: how many mini-batches are used each epoch. set to -1 to run the full epoch.
        :return: trained VAE, together with the test data used for testing.
        '''
    # we have three stages of training.
    # stage 1 = training marginal VAEs, stage 2 = training dependency network, (see Section 2 in our paper)
    # stage 3 = add predictor and improve predictive performance (See Appendix C in our paper)
    if stage ==1:
        load_model = 0
        disc_mode = 'non_disc'
    elif stage == 2:
        load_model = 1
        disc_mode = 'non_local'
    elif stage == 3:
        load_model = 1
        disc_mode = 'joint'
    obs_dim = Data_train.shape[1]
    n_train = Data_train.shape[0]
    list_train = np.arange(n_train)
    batch_size = np.minimum(batch_size,n_train)
    ####### construct
    kwargs = {
        'stage':stage,
        'K': K,
        'latent_dim': latent_dim,
        'batch_size': batch_size,
        'encoder': encoders.vaem_encoders(obs_dim,cat_dims,dim_flt,K, latent_dim ),
        'decoder': decoders.vaem_decoders(obs_dim,cat_dims,list_discrete,records_d),
        'obs_dim': obs_dim,
        'cat_dims': cat_dims,
        'dim_flt': dim_flt,
        'load_model':load_model,
        'decoder_path': os.path.join(args.output_dir, 'generator.tensorflow'),
        'encoder_path': os.path.join(args.output_dir, 'encoder.tensorflow'),
        'x_train':x_train,
        'list_discrete':list_discrete,
        'records_d':records_d,
    }
    vae = model.partial_vaem(**kwargs)
    if iteration == -1:
        n_it = int(np.ceil(n_train / kwargs['batch_size']))
    else:
        n_it = iteration
    hist_loss_full = np.zeros(epochs)
    hist_loss_cat = np.zeros(epochs)
    hist_loss_flt = np.zeros(epochs)
    hist_loss_z_local = np.zeros(epochs)
    hist_loss_kl = np.zeros(epochs)

    if stage == 3:
        ## after stage 2, do discriminative training of y. Please refer to Section 3.4 and Appendix C in our paper.
        for epoch in range(epochs):
            train_loss_full = 0.
            train_loss_cat = 0.
            train_loss_flt = 0.
            train_loss_z_local = 0.
            for it in range(n_it):
                if iteration == -1:
                    batch_indices = list_train[it * kwargs['batch_size']:min(it * kwargs['batch_size'] + kwargs['batch_size'], n_train - 1)]
                else:
                    batch_indices = sample(range(n_train), kwargs['batch_size'])
                x = Data_train[batch_indices, :]
                mask_train_batch = mask_train[batch_indices, :]
                DROPOUT_TRAIN = np.minimum(np.random.rand(mask_train_batch.shape[0], obs_dim), p)
                while True:
                    mask_drop = bernoulli.rvs(1 - DROPOUT_TRAIN)
                    if np.sum(mask_drop > 0):
                        break

                mask_drop = mask_drop.reshape([kwargs['batch_size'], obs_dim])
                _ = vae.update(x, mask_drop * mask_train_batch, 'disc')
                loss_full, loss_cat, loss_flt, loss_z_local, loss_kl, stds, _, _ = vae.full_batch_loss(x,mask_drop * mask_train_batch)
                train_loss_full += loss_full
                train_loss_cat += loss_cat
                train_loss_flt += loss_flt
                train_loss_z_local += loss_z_local

            # average loss over most recent epoch
            train_loss_full /= n_it
            train_loss_cat /= n_it
            train_loss_flt /= n_it
            train_loss_z_local /= n_it
            print(
                'Epoch: {} \tnegative training ELBO per observed feature: {:.2f}, Cat_term: {:.2f}, Flt_term: {:.2f},z_term: {:.2f}'
                    .format(epoch, train_loss_full, train_loss_cat, train_loss_flt, train_loss_z_local))

    for epoch in range(epochs):
        train_loss_full = 0. # full training loss
        train_loss_cat = 0. # reconstruction loss for non_continuous likelihood term
        train_loss_flt = 0. # reconstruction loss for continuous likelihood term
        train_loss_z_local = 0. # reconstruction loss for second stage on z space (gaussian likelihood)
        train_loss_kl = 0 # loss for KL term

        for it in range(n_it):
            if iteration == -1:
                batch_indices = list_train[it*kwargs['batch_size']:min(it*kwargs['batch_size'] + kwargs['batch_size'], n_train - 1)]
            else:
                batch_indices = sample(range(n_train), kwargs['batch_size'])

            x = Data_train[batch_indices, :]
            mask_train_batch = mask_train[batch_indices, :]
            DROPOUT_TRAIN = np.minimum(np.random.rand(mask_train_batch.shape[0], obs_dim), p)
            while True:
                mask_drop = bernoulli.rvs(1 - DROPOUT_TRAIN)
                if np.sum(mask_drop > 0):
                    break

            mask_drop = mask_drop.reshape([kwargs['batch_size'], obs_dim])
            _ = vae.update(x, mask_drop*mask_train_batch,disc_mode)
            loss_full, loss_cat, loss_flt,loss_z_local, loss_kl, stds, _,_ = vae.full_batch_loss(x,mask_drop*mask_train_batch)
            train_loss_full += loss_full
            train_loss_cat += loss_cat
            train_loss_flt += loss_flt
            train_loss_z_local += loss_z_local
            train_loss_kl += loss_kl

          # average loss over most recent epoch
        train_loss_full /= n_it
        train_loss_cat /= n_it
        train_loss_flt /= n_it
        train_loss_z_local /= n_it
        train_loss_kl /= n_it
        hist_loss_full[epoch] = train_loss_full
        hist_loss_cat[epoch] = train_loss_cat
        hist_loss_flt[epoch] = train_loss_flt
        hist_loss_z_local[epoch] = train_loss_z_local
        hist_loss_kl[epoch] = train_loss_kl

        print('Epoch: {} \tnegative training ELBO per observed feature: {:.2f}, Cat_term: {:.2f}, Flt_term: {:.2f},z_term: {:.2f}'
            .format(epoch, train_loss_full,train_loss_cat,train_loss_flt, train_loss_z_local))

    if stage <= 2:
        vae.save_generator(os.path.join(args.output_dir, 'generator.tensorflow'))
        vae.save_encoder(os.path.join(args.output_dir, 'encoder.tensorflow'))

    return vae



















##################################################################################################
##################################################################################################
class Model(object):
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None):
        self.model_pars, self.compute_pars, self.data_pars = model_pars, compute_pars, data_pars
        self.history = None
        if model_pars is None:
            self.model = None
            return
        # log2("data_pars", data_pars)
        model_class = model_pars['model_class']  #

        ### get model params  #######################################################
        mdict_default = {
             'class_num':           5
            ,'intermediate_dim':    64
        }
        mdict = model_pars.get('model_pars', mdict_default)

        ### Dynamic Dimension : data_pars  ---> model_pars dimension  ###############
        data_decode,records_d = load_data()




        #### Model setup ################################
        list_discrete = np.array([8,9])
        args = params.Params('./hyperparameters/bank_plot.json')
        model1,scaling_factor = encode2(data_decode,list_discrete,records_d,fast_plot)
        model2                = decode2(data_decode, scaling_factor,list_discrete,records_d)

        self.model ={
            'encode' :  model1,
            'decode' :  model2
        }

        log2(self.model_pars, self.model)
        self.model.summary()


def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """
    global model, session
    session = None  # Session type for compute

    Xtrain_tuple, ytrain, Xtest_tuple, ytest = get_dataset(data_pars, task_type="train",)
    cpars          = copy.deepcopy( compute_pars.get("compute_pars", {}))   ## issue with pickle

    ### Fake label
    Xtrain_dummy = np.ones((Xtrain_tuple.shape[0], 1))

    assert 'epochs' in cpars, 'epoch missing'
    model.model['encode'].fit([Xtrain_tuple, Xtrain_dummy],  **cpars)
    model.model['decode'].fit([Xtrain_tuple, Xtrain_dummy],  **cpars)



def encode(Xpred=None, data_pars=None, compute_pars={}, out_pars={}, **kw):
    global model, session
    if Xpred is None:
        # data_pars['train'] = False
        Xpred_tuple = get_dataset(data_pars, task_type="predict")

    else :
        cols_type   = data_pars.get('cols_model_type2', {})  ##
        Xpred_tuple = get_dataset_tuple(Xpred, cols_type, cols_ref_formodel)

    log2(Xpred_tuple)
    Xdummy = np.ones((Xpred_tuple.shape[0], 1))
    ypred = model.model.encode([Xpred_tuple, Xdummy ] )

    return ypred


def decode(Xpred=None, data_pars=None, compute_pars={}, out_pars={}, **kw):
    global model, session
    if Xpred is None:
        # data_pars['train'] = False
        Xpred_tuple = get_dataset(data_pars, task_type="predict")

    else :
        cols_type   = data_pars.get('cols_model_type2', {})  ##
        Xpred_tuple = get_dataset_tuple(Xpred, cols_type, cols_ref_formodel)

    log2(Xpred_tuple)
    Xdummy = np.ones((Xpred_tuple.shape[0], 1))
    ypred = model.model.decode([Xpred_tuple, Xdummy ] )

    return ypred



####################################################################################################
def get_dataset_tuple(Xtrain, cols_type_received, cols_ref):
    """  Split into Tuples to feed  Xyuple = (df1, df2, df3)
    :param Xtrain:
    :param cols_type_received:
    :param cols_ref:
    :return:
    """
    if len(cols_ref) <= 1 :
        return Xtrain

    Xtuple_train = []
    for cols_groupname in cols_ref :
        assert cols_groupname in cols_type_received, "Error missing colgroup in config data_pars[cols_model_type] "
        cols_i = cols_type_received[cols_groupname]
        Xtuple_train.append( Xtrain[cols_i] )

    return Xtuple_train



def get_dataset(data_pars=None, task_type="train", **kw):
    """
      return tuple of dataframes
    """
    # log(data_pars)
    if data_pars.get('dataset_name', '') == 'correlation' :
       x_train, ytrain = load_data(data_pars)
       return x_train, ytrain


    data_type = data_pars.get('type', 'ram')
    cols_ref  = cols_ref_formodel

    if data_type == "ram":
        # cols_ref_formodel = ['cols_cross_input', 'cols_deep_input', 'cols_deep_input' ]
        ### dict  colgroup ---> list of colname
        cols_type_received     = data_pars.get('cols_model_type2', {} )  ##3 Sparse, Continuous

        if task_type == "predict":
            d = data_pars[task_type]
            Xtrain       = d["X"]
            Xtuple_train = get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
            return Xtuple_train

        if task_type == "eval":
            d = data_pars[task_type]
            Xtrain, ytrain  = d["X"], d["y"]
            Xtuple_train    = get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
            return Xtuple_train, ytrain

        if task_type == "train":
            d = data_pars[task_type]
            Xtrain, ytrain, Xtest, ytest  = d["Xtrain"], d["ytrain"], d["Xtest"], d["ytest"]

            ### dict  colgroup ---> list of df
            Xtuple_train = get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
            Xtuple_test  = get_dataset_tuple(Xtest, cols_type_received, cols_ref)
            #flog2("Xtuple_train", Xtuple_train)

            return Xtuple_train, ytrain, Xtuple_test, ytest


    elif data_type == "file":
        raise Exception(f' {data_type} data_type Not implemented ')

    raise Exception(f' Requires  Xtrain", "Xtest", "ytrain", "ytest" ')

######################################################################################
def reset():
    global model, session
    model, session = None, None

def save(path=None, info=None):
    import dill as pickle, copy
    global model, session
    os.makedirs(path, exist_ok=True)

    model.model.save(f"{path}/model_keras.h5")
    model.model.save_weights(f"{path}/model_keras_weights.h5")

    modelx = Model()  # Empty model  Issue with pickle
    modelx.model_pars   = model.model_pars
    modelx.data_pars    = model.data_pars
    modelx.compute_pars = model.compute_pars
    # log('model', modelx.model)
    pickle.dump(modelx, open(f"{path}/model.pkl", mode='wb'))  #

    pickle.dump(info, open(f"{path}/info.pkl", mode='wb'))  #


def load_model(path=""):
    global model, session
    import dill as pickle

    model0      = pickle.load(open(f"{path}/model.pkl", mode='rb'))

    model = Model()  # Empty model
    model.model        = get_model( model0.model_pars)
    model.model_pars   = model0.model_pars
    model.compute_pars = model0.compute_pars

    model.model.load_weights( f'{path}/model_keras_weights.h5')

    log(model.model.summary())
    #### Issue when loading model due to custom weights, losses, Keras erro
    #model_keras = get_model()
    #model_keras = keras.models.load_model(path + '/model_keras.h5' )
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



##################################################################################

def test():
    ### Custom dataset
    adata_pars = {'dataset_name':  'correlation'}
    adata_pars['state_num']           = 10
    X,y = load_data()


    ####
    d = {'task_type' : 'train', 'data_type': 'ram',}
    d['signal_dimension'] = 15

    d["train"] ={
      "Xtrain":  X[:10,:],
      "ytrain":  y[:10,:],        ## Not used
      "Xtest":   X[10:1000,:],
      "ytest":   y[10:1000,:],    ## Nor Used
    }

    data_pars= d
    m                       = {}
    m['original_dim']       = np.uint32( adata_pars['signal_dimension']*(adata_pars['signal_dimension']-1)/2)
    m['class_num']          = 5
    model_pars = {'model_pars'  : m,
                  'model_class' : "class_VAEMDN"
                 }

    compute_pars = {}
    compute_pars['compute_pars'] = {'epochs': 1, }   ## direct feed


    ### Meta Class #########################################################
    X = load_data()
    test_helper(model_pars, data_pars, compute_pars, X)



def test_helper(model_pars, data_pars, compute_pars, Xpred):
    global model, session
    init()
    root  = "ztmp/"
    model = Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)

    log('\n\nTraining the model..')
    fit(data_pars=data_pars, compute_pars=compute_pars, out_pars=None)


    log('Predict data..')
    ypred, ypred_proba = encode(Xpred=Xpred, data_pars=data_pars, compute_pars=compute_pars)
    log(f'Top 5 y_pred: {np.squeeze(ypred)[:3]}')

    #
    log('Saving model..')
    log( model.model )
    save(path= root + '/model_dir/')


    log('Load model..')
    model, session = load_model(path= root + "/model_dir/")
    log('Model successfully loaded!\n\n')

    log('Model architecture:')
    log(model.model)

    log('Predict data..')
    ypred, ypred_proba = encode(Xpred=Xpred, data_pars=data_pars, compute_pars=compute_pars)







if __name__ == "__main__":
    test2()



