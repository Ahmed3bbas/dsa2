import tensorflow as tf
import pandas as pd
import pprint
# dst = {}

class dictEval(object):
    dst= {}
    def eval_dict(self, ddict):
        for key, value in ddict.items():
            if isinstance(value, dict):
                node = dst.setdefault(key, {})
                self.eval_dict(value)
            else:
                if "@lazy" in key :
                    dst[key] = value
                else :
                    if 'pandas' in key :
                        key2 = key.split(":")[-1]
                        dst[key2] = pd.read_csv(value)
        return dst




data_pars= {

'train' :

{
	'@lazy_pandas:Xtrain' : folder_parquet_zip,

	'@lazy_pandas:Xtest' :  folder_parquet_zip,

    '@lazy_tfdataset:Xtest' :  folder_parquet,

    '@lazy_pyarrow:Xtest' :  folder_parquet,

},


'pars' : 23,


}
test = dictEval()
dict_new = test.eval_dict(data_pars)





test = dictEval()
dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
csv_file    = 'datasets/petfinder-mini/petfinder-mini.csv'
tf.keras.utils.get_file('petfinder_mini.zip', dataset_url,extract=True, cache_dir='.')
src = {
    '@lazy':{
        'pandas.read_csv:petfinder':csv_file
    }
}

pprint.pprint(test.eval_dict(src))
