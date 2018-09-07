#########################################################################################################################################
#########################################################################################################################################
###                  Classification on Multiple Target Variables on the Same Dataset                                                  ###
###                         Auxilliary File                                                                                           ### 
###                       Programmed by E. M. (CMTVEM)                                                                                ###
###                                   July 2018                                                                                       ###
###   preprocessing: Performs preprocessing on training or test data according to configurations parameters                           ###
###   Input: configuration parameters dictionary, training_flag (True for training set, False for test set)                           ###
###   Output: preprocessed dataframe, updated configuration parameters                                                                ###
#########################################################################################################################################
#########################################################################################################################################
#Importing libraries
import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

#importing functions from accompanying CMTVEM .py files
from CMTVEM_auxilliary import output_writer


def preprocessing(configparms,training_preprocessing_flag):
    
    if training_preprocessing_flag:
        df = pd.read_csv(configparms['training_file'])
    else:
        #empty list since test data does not include target variables
        configparms['target_name_list'] = list()        
        df = pd.read_csv(configparms['test_file'])
        df['stabilityVec']=0
    
    reportCMTVEM = list()    
    reportCMTVEM.append("---------------------------------------------------------------")
    reportCMTVEM.append("input parameters from configuration file")
    reportCMTVEM.append(configparms)
    
    #removing chemically/physically meaningless features
    df.drop(list(configparms['remove_features'].values()), axis=1, inplace=True)
    
    #revise features that need corrections
    for item in list(configparms['revised_features'].values()):
        feature_name_temp = item.split(',')[0].split()[0]
        element_name_temp = item.split(',')[1].split()[0]
        correct_value_temp = item.split(',')[2].split()[0]
        df.loc[df.formulaA == element_name_temp, feature_name_temp] = float(correct_value_temp)
        df.loc[df.formulaB == element_name_temp, feature_name_temp] = float(correct_value_temp)
    
    # revise features that are non-numerical and making categorical features in their stead on the data-frame 
    df_added_categoricals = pd.get_dummies(df, columns=list(configparms['categorical_features'].values()))
    configparms['categorical_features_fullnames']=list(set(df_added_categoricals.columns) - set(df.columns))
    #copy back
    df = copy.deepcopy(df_added_categoricals) 
    

    #Handling missing values
    #since the zero values in certain features are physically meaningless (eg. bulk modulus) it needs to be changed to NAN
    if configparms['imputation_method'] != 'none':
        for item in list(configparms['impute_features'].values()):
            df[item] = df[item].replace(0 , np.nan)
    
    df_features_data = df.drop(['formulaA','formulaB','stabilityVec'], axis=1)

    if configparms['imputation_method']=='mn':
        imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)
        for item in list(configparms['impute_features'].values()):
            df_features_data[item] = imputer.fit_transform(df_features_data[item].values.reshape(-1,1)).ravel()
    
               
    #Reflect the changes on the original data frame
    df[df_features_data.columns] = df_features_data[df_features_data.columns]
    
    if training_preprocessing_flag:
        #if training set is under preprocessing
        #This section is for preparing the target variable for training set only
        for item in list(configparms['element_removal'].values()):
            df = df[df.formulaA.str.contains(item) == False]
            df= df.reset_index(drop=True)

        df['target_vector'] = df['target_vector'].map(lambda x: x.lstrip('[').rstrip(']'))
        configparms['target_name_list'] = ['target1','target2','target3','target4','target5'] 
        df[configparms['target_name_list']]=df['target_vector'].str.split(',', expand=True)
        df[configparms['target_name_list']]=df[configparms['target_name_list']].astype(np.float)
            
        #Writes the preprocessed training data into a csv file
        df.to_csv('training_preprocessed.csv') 
        output_writer(reportCMTVEM,configparms)
    
    return(df,configparms)    


