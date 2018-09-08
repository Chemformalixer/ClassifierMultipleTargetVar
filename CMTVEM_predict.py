#########################################################################################################################################
#########################################################################################################################################
###                  Classification on Multiple Target Variables on the Same Dataset                                                  ###
###                         Prediction file                                                                                           ### 
###                       Programmed by E. M. (CMTVEM)                                                                                ###
###                                   July 2018                                                                                       ###
###                                                                                                                                   ###
###   predict: Predicts stability vector and saves it on the test file                                                                ###
###   Input: preprocessed dataframe from test set, configuration parameters                                                           ###
###   Output: updated configuration parameters which includes updated hyper-parameters                                                ###
#########################################################################################################################################
#########################################################################################################################################

#Importing libraries
import pandas as pd
import datetime
import time
from sklearn.preprocessing import StandardScaler

#importing functions from accompanying CCEM .py files
from CMTVEM_auxilliary import output_writer

#suppressing copy warning
pd.options.mode.chained_assignment = None


def predict(configparms,df_unkown,dispatcher,start_time):
    reportCMTVEM = list()    
    reportCMTVEM.append("---------------------------------------------------------------")
    
    #features_list = [x for x in df_unkown.columns if x not in ['feature_idx','target_vector']]
    target_pred = pd.DataFrame(index = range(0,len(df_unkown[features_list[1]])), columns = configparms['target_name_list'])
    if configparms['scale']=='yes':
        scaler = StandardScaler()
        reportCMTVEM.append("test data scaled")
        reportCMTVEM.append(scaler.fit(df_unkown[features_list]))
        df_unkown[features_list]=pd.DataFrame(scaler.transform(df_unkown[features_list]))

    for key, value in dispatcher.items():
        if key == 'average_Matthews_score':
            continue
        
        target_pred[key] = dispatcher[key].predict(df_unkown[features_list])
    
    
    # if certain target values are known from off-training knowledge
    target_pred['feature_idx']=df_unkown['feature_idx']
    target_pred.loc[target_pred.feature_idx.isin(configparms['element_removal'].values()), configparms['target_name_list']] = 0
    target_pred['target_vector'] = '['+target_pred['target1'].map(str)+','+target_pred['target2'].map(str)+','+target_pred['target3'].map(str)+','+target_pred['target4'].map(str)+','+target_pred['target5'].map(str)+',]'
    
    df_original_test = pd.read_csv(configparms['test_file'])
    df_original_test['target_vector'] = target_pred['target_vector']
    df_original_test.to_csv(configparms['test_file'], index=False)
    reportCMTVEM.append("Test file prediction commenced at")    
    reportCMTVEM.append(str(datetime.datetime.now()))
    elapsed_time=int(time.time()-start_time)
    reportCMTVEM.append("total cpu time elapsed in seconds:")    
    reportCMTVEM.append(str(elapsed_time))
    reportCMTVEM.append("---------------------------------------------------------------")
    output_writer(reportCMTVEM,configparms)

    return()
    
