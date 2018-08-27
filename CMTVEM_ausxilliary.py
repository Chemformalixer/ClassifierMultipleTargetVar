##############################################################################################################################################################
##############################################################################################################################################################
###                  Classification on Multiple Target Variables on the Same Dataset                                                                       ###
###                         Auxilliary File                                                                                                                ### 
###                       Programmed by E. M. (CMTVEM)                                                                                                     ###
###                                   July 2018                                                                                                            ###
###   inp_read_parm: reads global parameters in configuration file by default 'configuration.txt'                                                          ###
###   Input: configuration parameters file name (default = 'configuration.txt')                                                                            ###
###   Output: start_time of the program, configuration parameters                                                                                          ###
##############################################################################################################################################################
###   sub-functions:                                                                                                                                       ###
###   output: writes the output report on the report file                                                                                                  ###
###   error_message_generator: generates necessary error messages and writes them into report file                                                         ###
##############################################################################################################################################################
##############################################################################################################################################################

#Importing libraries
import os
import sys
import datetime
import time

def inp_read_parm(configuration_file_name):

    reportCMTVEM = list()#starting the report, reportCMTVEM will be written on the report txt file at every stage of the program
    reportCMTVEM.append("---------------------------------------------------------------")
    reportCMTVEM.append("CMTVEM report file generated at:")    
    start_time = time.time()
    reportCMTVEM.append(str(datetime.datetime.now()))
    print(sys.version)
    if sys.version[0] is not '3':
        error_message_generator(0)#Prints an error message if the python version is not the same as this program was written for
        sys.exit()
        

    cwd = os.getcwd()  # current working directory (cwd)
    cwdfiles = os.listdir(cwd)  # get all the files in that directory
    if not(configuration_file_name in cwdfiles) or not('training_data.csv' in cwdfiles) or not('test_data.csv' in cwdfiles) or not('performance_report.txt' in cwdfiles):
        error_message_generator(1)#prints an error message if necessary files are not present in the working directory 
        sys.exit()

    with open(configuration_file_name) as input_parameters_file:#Starts reading the configuration file and all parameters
        content = input_parameters_file.readlines()
        content = [x.strip() for x in content]
    nblock_inputfile = 0
    #number blocks of variable dictionaries in configuration file (after line 25 in configuration.txt)
    index_list = list(range(0,5)) 
    configparms = {}
    configparms['training_file']=('{}'.format(content[1]))
    configparms['test_file']=('{}'.format(content[3]))
    configparms['performance_report_file']=('{}'.format(content[5]))
    configparms['imputation_method']=('{}'.format(content[7]))
    configparms['learning_model']=('{}'.format(content[9]))
    #double number of entries by considering switching A and B elements 
    configparms['binary_combine']=('{}'.format(content[11])) 
    configparms['target_variable_definition']=('{}'.format(content[13]))
    configparms['balance_sampling']=('{}'.format(content[15]))
    configparms['scale']=('{}'.format(content[17]))
    configparms['hyper_parameter_tuning_model']=('{}'.format(content[19]))
    configparms['hyper_parameters']=list(map(float,('{}'.format(content[21])).split(',')))
    configparms['roc_plot']=('{}'.format(content[23]))
    # building nested structure for variable number of features that need alteration
    configparms['remove_features'] = {} 
    configparms['impute_features'] = {}
    configparms['categorical_features'] = {}
    configparms['revised_features'] = {}
    configparms['element_removal']={}
    item2=0
    #changeable number in the configuration file for features that need modification
    #25 is the line number in configuration file where changeable number of features begin
    for item1 in range(25,len(content)): 
        if content[item1].startswith('#'): 
            #  variable-size parameters in config file start at line 25
            index_list[nblock_inputfile]=item1-25 
            nblock_inputfile = nblock_inputfile +1
            item2 = 0
            continue
        #first block of changeable number of features; marked for removal
        if nblock_inputfile == 0: 
            configparms['remove_features'][item2] = content[item1] 
            item2 = item2 +1
        #second block of changeable number of features; marked for imputation
        if nblock_inputfile == 1:
            configparms['impute_features'][item2] = content[item1]
            item2 = item2 +1
        #third block of changeable number of features; marked for making categorical
        if nblock_inputfile == 2:
            configparms['categorical_features'][item2] = content[item1]
            item2 = item2 +1
        #fourth block of changeable number of features; mark for revision
        if nblock_inputfile == 3:
            configparms['revised_features'][item2] = content[item1]
            item2 = item2 +1
        #fifth block of changeable number of elements; marked for removal
        if nblock_inputfile == 4:
            configparms['element_removal'][item2] = content[item1]
            item2 = item2 +1
            
    #Writes all the parameters by appending the output report file, this is helpful to check performance vs configuration parameters in case of multiple runs
    output_writer(reportCMTVEM,configparms)
    return(start_time,configparms)

#Writes the report by appending the report file
def output_writer(reportCMTVEM,configparms):
    output_file_name = configparms['performance_report_file']
    outputfile_CMTVEM = open(output_file_name, 'a')
    
    for item in reportCMTVEM:
        outputfile_CMTVEM.write("%s\n" % item)
    
    return()

def error_message_generator(error_code):
    if error_code ==0:
        error_message = "Error #0: Program was originally written for python version 3.5.2 make sure, your version might be incompatible."
    elif error_code == 1:
        error_message = "Error #1: Make sure all necessary files configuration.txt, performance_report.txt, training_data.csv, and test_data.csv are in the working directory."
    output_writer(error_message,None)
    print(error_message)
    return(error_message)
