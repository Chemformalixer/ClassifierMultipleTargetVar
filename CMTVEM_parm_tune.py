################################################################################################################################
################################################################################################################################
###                  Classification on Multiple Target Variables on the Same Dataset                                         ###                                                                    ###
###                         Parameter Tuning File                                                                            ###                            ### 
###                       Programmed by E. M. (CMTVEM)                                                                       ###                         ###
###                                   July 2018                                                                              ###                      ###
###                                                                                                                          ###
###   parm_tune: Performs hyperparameter tuning                                                                              ###
###   Input: preprocessed training dataframe, configuration parameters                                                       ###
###   Output: updated configuration parameters which includes updated hyper parameters                                       ###
################################################################################################################################
################################################################################################################################

#Importing libraries
import numpy as np
import pandas as pd
import math

#importing functions from accompanying CMTVEM .py files
from CMTVEM_auxilliary import output_writer
from CMTVEM_train_validate import train_validate
#suppressing copy warning
pd.options.mode.chained_assignment = None

def parm_tune(df, configparms):
    reportCMTVEM = list()    
    reportCMTVEM.append("---------------------------------------------------------------")

    if not(configparms['hyper_parameter_tuning_model'] == 'ga' and configparms['learning_model'] == 'ev'):
        reportCMTVEM = list()    
        reportCMTVEM.append("---------------------------------------------------------------")
        reportCMTVEM.append("No hyper parameter tuning is performed.")
        reportCMTVEM.append("If you need tuning choose ev option for model and yes for tuning in config.txt file")
        output_writer(reportCMTVEM,configparms)
        return(configparms)

    reportCMTVEM.append("hyper parameter tuning")

    def gradient_model(coefs):
        configparms['hyper_parameters']=coefs
        temp_f1_item = 0
        for item in range(0,3):
            model = train_validate(df,configparms)
            temp_f1_item = temp_f1_item + model['average_Matthews_score'] 
        f1_score_average = (temp_f1_item/3)
        return(f1_score_average)
    def gard_calc(coefs):
        #h small change for float interval
        h=1
        #uh small change for integer optimization
        uh=1
        u1,u2,w1,w2,w3 = coefs
        grad_u1=(gradient_model([u1+uh,u2,w1,w2,w3])-gradient_model([u1-uh,u2,w1,w2,w3]))/(2*uh)
        grad_u2=(gradient_model([u1,u2+uh,w1,w2,w3])-gradient_model([u1,u2-uh,w1,w2,w3]))/(2*uh)
        grad_w1=(gradient_model([u1,u2,w1+h,w2,w3])-gradient_model([u1,u2,w1-h,w2,w3]))/(2*h)
        grad_w2=(gradient_model([u1,u2,w1,w2+h,w3])-gradient_model([u1,u2,w1,w2-h,w3]))/(2*h)
        grad_w3=(gradient_model([u1,u2,w1,w2,w3+h])-gradient_model([u1,u2,w1,w2,w3-h]))/(2*h)
        return([grad_u1, grad_u2, grad_w1, grad_w2, grad_w3])
    
    coefs_history=[]
    coefs_current = configparms['hyper_parameters']
    
    #eta controls the size of steps in gradient optimization
    eta = 10
    
    #maximum number of iterations
    max_iterations=30
    
    #stopping condition tolerance
    epsilon=0.00001
    
    for item in range(0,max_iterations):
        
        coefs_history.append(coefs_current)
        
        coefs_next = [i + eta*j for i, j in zip(coefs_current, gard_calc(coefs_current))]
        
        #integer optimization for first two coefs
        #int_opt_t is integer optimization threshold
        int_opt_t = 0.1 
        if coefs_next[0]>coefs_current[0]+int_opt_t:
            coefs_next[0] = math.floor(coefs_next[0])+1
        elif coefs_next[0]<coefs_current[0]-int_opt_t:
            coefs_next[0] = math.floor(coefs_next[0])
        else:
            coefs_next[0] = coefs_current[0]
        if coefs_next[1]>coefs_current[1]+int_opt_t:
            coefs_next[1] = math.floor(coefs_next[1])+1
        elif coefs_next[1]<coefs_current[1]-int_opt_t:
            coefs_next[1] = math.floor(coefs_next[1])
        else:
            coefs_next[1] = coefs_current[1]
           
        #stopping condition
        if np.linalg.norm(list(i - j for i,j in zip(coefs_next, coefs_current))) <= epsilon * np.linalg.norm(coefs_current):
            break
        coefs_current = coefs_next
        str_temp=', '.join(str(e) for e in coefs_current)
        f1_score_of_iteration = gradient_model(coefs_current)
        reportCMTVEM.append("updated coefficients after "+ str(item)+ " iterations " + str_temp + ", Matthews score= " + str(f1_score_of_iteration))
        output_writer(reportCCEM,configparms)

        
    print("gradient descent finished after " + str(item) + " iterations" + str_temp+ ", Matthews score= " + str(f1_score_of_iteration))
    reportCMTVEM.append("gradient descent finished after " + str(item) + " iterations "+ str_temp+ ", Matthews score= " + str(f1_score_of_iteration))
    configparms['hyper_parameters']=coefs_current
    return(configparms)

