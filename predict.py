# #! /usr/bin/python
# # sys.argv[1] : filePath
# # sys.argv[2] : modelType
import sys
import os
import autogluon as ag
# from autogluon import TabularPrediction as task
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd ## read/write data
import re ## regex search for features name
from sklearn import preprocessing ## standardization
import math ## take log10
import numpy as np ## data handling
from autogluon.tabular import TabularDataset, TabularPredictor


def pred(dir_v,refFeatures,test_data):
    predictor = TabularPredictor.load(dir_v)
    # write predict res to file
    p_model_test = predictor.predict(test_data,model='WeightedEnsemble_L2')
    return p_model_test
def refFeaturesArr(model):
    if model == "E":
        refFeatures = ['EtOH-eq (%)(v/v)']
    elif model == "ET":
        refFeatures = ['EtOH-eq (%)(v/v)','Temp (C)']
    elif model == "ETM":
        refFeatures = ['EtOH-eq (%)(v/v)', 'Material_consolidated', 'Temp (C)']
    elif model == "ETML":
        refFeatures = ['EtOH-eq (%)(v/v)', 'Material_consolidated', 'logKow_25C','Temp (C)']
    return refFeatures
def AD(exclude_rule,features,test_data,fp_column):
    tmp_exclude_rule = exclude_rule[(exclude_rule.way == features)]
    test_data["AD"] = True
    for index_td, item in test_data.iterrows():
        fpbitset = set(np.where(np.array(list(item[list(fp_column)]))=='1')[0])#check fingerprint in domain
        for index, row in tmp_exclude_rule.iterrows():
            if features == "ETML":#check logKow in domain
                if row['Lgte']==row['Lgte']:
                    if item['logKow_25C']<row['Lgte']:
                        continue
                if row['Llt']==row['Llt']:
                    if item['logKow_25C']>=row['Llt']:
                        continue            
            if (features == "ETM") or (features == "ETML"):#check Material in domain
                if row['M']==row['M']:
                    material = set()
                    material = set([i for i in row['M'].split("|")])
                    if item['Material_consolidated'] not in material:
                        continue
            if features == "ET" or (features == "ETM") or (features == "ETML"):#check Temp in domain
                if row['Tgte']==row['Tgte']:
                    if item['Temp (C)']<row['Tgte']:
                        continue
                if row['Tlt']==row['Tlt']:
                    if item['Temp (C)']>=row['Tlt']:
                        continue
            if row['Egte']==row['Egte']:#check EtOH-eq in domain
                if item['EtOH-eq (%)(v/v)']<row['Egte']:
                    continue
            if row['Elt']==row['Elt']:
                if item['EtOH-eq (%)(v/v)']>=row['Elt']:
                    continue
            human_0 = set()
            human_1 = set()
            if row['human_0']==row['human_0']: # not NaN
                human_0 = set([i for i in row['human_0'].split(" ,")])
                human_0 = set(map(int, human_0))
            if row['human_1']==row['human_1']: # not NaN
                human_1 = set([i for i in row['human_1'].split(" ,")])
                human_1 = set(map(int, human_1))
            not_contain_set = human_0 & fpbitset
            diff_contain_set = human_1 - fpbitset
            if (len(not_contain_set)==0) & (len(diff_contain_set)==0):
                test_data.at[index_td,"AD"] = False
                break
    return pd.Series(test_data["AD"])

test_data = TabularDataset(sys.argv[1])
exclude_rule = pd.read_csv("./AD_exclude_rule.csv")
fp_column = list(filter(re.compile("^V[0-9]+$").match, list(test_data.columns)))

for fp_v in fp_column:
    test_data[fp_v] = test_data[fp_v].apply(str)
if sys.argv[2] == "batch":
    for features in ["E","ET","ETM","ETML"]:
        refFeatures = refFeaturesArr(features)
        tmp_test = test_data[(test_data.modelType == features)]
        if len(tmp_test.index) >0:
            df = pd.DataFrame(columns=["pred_logKPF","AD"])
            dir_v = "./autogluon_model/" +features+"/"
            df["pred_logKPF"] = pred(dir_v, refFeatures, test_data = tmp_test)
            df["AD"] = AD(exclude_rule,features,test_data = tmp_test,fp_column=fp_column)
            predCols = df.columns
            df = pd.concat([tmp_test.iloc[:, 0:7], df], axis=1)
            df.columns = [*['Food_liquid','EtOH-eq','Material_consolidated','logKow_25C','Temp','SMILES','modelType'],*predCols]
            filepath = sys.argv[1]+'_res.csv'
            if os.path.isfile(filepath):
                df.to_csv(filepath ,mode='a', index = True, header=False)
            else:
                df.to_csv(filepath ,mode='a', index = True, header=True)
    print(filepath)
else:
    refFeatures = refFeaturesArr(sys.argv[2])
    df = pd.DataFrame(columns=["pred_logKPF","AD"])
    dir_v = "./autogluon_model/" + sys.argv[2]+"/"
    df["pred_logKPF"] = pred(dir_v, refFeatures, test_data = test_data)
    df["AD"] = AD(exclude_rule,sys.argv[2],test_data = test_data,fp_column=fp_column)
    predCols = df.columns
    df = pd.concat([test_data.iloc[:, 0:5], df], axis=1)
    df.columns = [*['Food_liquid','EtOH-eq','Material_consolidated','logKow_25C','Temp'], *predCols]
    #df.to_csv(sys.argv[1]+'_res.csv',mode='a', index = False, header=True)
    print(df.to_json(orient="records"))
