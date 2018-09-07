################################################################################################################################
################################################################################################################################
###                                                                                                                          ###
###                                                                                                                          ###
###                     Classification on Multiple Target Variables on the Same Dataset                                      ### 
###             Automated Preprocessing, Balancing, Training and Validation and in House Hyper-parameter Tuning              ###
###                                 Programmed by E. M.   (CMTVEM)                                                           ###
###                                   July 2018                                                                              ###
###   Python 3.5.2                                                                                                           ###
###   Main programs calls the following functions :                                                                          ###
###   1-inp_read_parm: Determine global parameters in configuration file by default 'configuration.txt'                      ###
###   2-preprocessing: Performs preprocessing on training and test data according to configurations parameters               ###
###   3-train_validate: Constructs, trains and validates the model                                                           ###
###   4-parm_tune: Performs hyperparameter tuning                                                                            ###
###   5-predict: Predicts stability vector and saves it on the test file                                                     ###
###                                                                                                                          ###
###                                                                                                                          ###
################################################################################################################################
################################################################################################################################

#Importing CMTVEM functions (see above)
from CMTVEM_auxilliary import inp_read_parm
from CMTVEM_preprocessing import preprocessing
from CMTVEM_train_validate import train_validate
from CMTVEM_parm_tune import parm_tune
from CMTVEM_predict import predict


def main():
    #Reads input parameters to configure preprocessing and modeling procedure
    start_time,configparms=inp_read_parm('configuration.txt') 
    #training_file_preprocessing_flag = True b/c it is preprocessing the training data
    df,configparms, stab_sum_gen = preprocessing(configparms,True)  
    #model_dispatcher is the dictionary of all N models for N targets
    model_dispatcher = train_validate(df,configparms)
    #Updates hyper parameters by tuning
    configparms = parm_tune(df, configparms)
    
    #Prediction:
    #training_file_preprocessing_flag = False b/c test data is undergoing preprocessing
    df_unknown,_,_ = preprocessing(configparms,False)
    #Predicts target stability vector on test set and writes the updated test csv file
    predict(configparms,df_unknown,model_dispatcher,start_time)

################################################################################################################################
if __name__ == '__main__':
    main()
