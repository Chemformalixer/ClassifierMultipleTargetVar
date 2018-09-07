#####################################################################################################################################################
#####################################################################################################################################################
###                                                                                                                                               ###
###                                                                                                                                               ###
###                     Classification on Multiple Target Variables on the Same Dataset                                                           ### 
###             Automated Preprocessing, Balancing, Training and Validation and in House Hyper-parameter Tuning                                   ###
###                                     Training and validation file                                                                                  ###
###                                 Programmed by E. M.   (CMTVEM)                                                                                ###
###                                   July 2018                                                                                                   ###
###   train_validate: Constructs, trains and validates the model                                                                                  ###
###   Input: preprocessed training dataframe, configuration parameters                                                                            ###
###   Output: trained models in the form of a dictionary(model_dispatcher) of 9 classification functions along with average Matthews score        ###
#####################################################################################################################################################
#####################################################################################################################################################


#Importing libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, auc, matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

#importing functions from accompanying CMTVEM .py files
from CMTVEM_auxilliary import output_writer

#suppressing copy warning
pd.options.mode.chained_assignment = None

def train_validate(df,configparms):
    
    #starting the report, reportCMTVEM will be written on the report txt file at every stage of the program  
    reportCMTVEM = list()
    reportCMTVEM.append("---------------------------------------------------------------")
    reportCMTVEM.append("performance of the model")
    #initialize classifier dictionary
    model_dispatcher={}

    X = df.drop(configparms['target_name_list'], axis = 1)
    X = X.drop(['formulaA','formulaB'], axis = 1)
    matthews_score_array = []
    f1_score_array = []
    precision_array = []
    recall_array = []
    accuracy_array = []
    n_item = 0
    colors = ['aqua', 'green', 'blue', 'red', 'sienna']
    phases = ['target1','target2', 'target3', 'target4', 'target5']
    linestyles = ['--',':','-.','--','-']

    if configparms['scale']=='yes':
        scaler = StandardScaler()
        reportCMTVEM.append("train data scaled")
        #remove categorical variables
        reportCMTVEM.append(scaler.fit(X.drop(configparms['categorical_features_fullnames'], axis = 1)))
        #put data back together
        X=pd.concat([pd.DataFrame(scaler.transform(X.drop(configparms['categorical_features_fullnames'], axis = 1))), X[configparms['categorical_features_fullnames']]], axis=1)
    for item in configparms['target_name_list']:
        if item=='stabilitysum':
            continue
        #if item == 'stability_5th_interval': #this conditional block can be used for testing the code with parm tuner at a faster pace
        #    break
        y=df[item]
        [X_train, X_Validation, y_train, y_Validation] = train_test_split(X, y, test_size = 0.25)#Splitting train and validation sets
        #balancing because the data is dominated by unstable binary states
        if configparms['balance_sampling']!='n' :
            #kind = type of balancing
            smethod = SMOTE(kind=configparms['balance_sampling'])
            #only train data should be synthetically modified to become balanced, validation data should not be touched
            [X_res, y_res] = smethod.fit_sample(X_train,y_train)
            X_train=pd.DataFrame(X_res)
            y_train=pd.DataFrame(y_res)
        #random forest
        if configparms['learning_model']=='rf':
            classifier = RandomForestClassifier(n_estimators = 300, max_features=15, criterion = 'entropy')
        #support vector machine classifier
        elif configparms['learning_model']=='svc':
            classifier = SVC(kernel='poly', max_iter=100, gamma = 2)
        #k nearest neighbor
        elif configparms['learning_model']=='knn':
            classifier = KNeighborsClassifier(n_neighbors=5)
        #gaussian process
        elif configparms['learning_model']=='gp':
            classifier = GaussianProcessClassifier(1.0 * RBF(1.0), max_iter_predict = 10)
        #multilayer perceptron
        elif configparms['learning_model']=='mlp':
            classifier = MLPClassifier(alpha=1,hidden_layer_sizes=(95,75))
        #ensemble voting
        elif configparms['learning_model']=='ev':
            #ev does not include gaussian process due to heavy computational cost, does not include svc either for limited accuracy
            #n_estimators = 301 is chosen empirically, after 300 not much is gained in accuracy
            classifier1 = RandomForestClassifier(n_estimators = 301, max_features=int(configparms['hyper_parameters'][0]), criterion = 'entropy') 
            classifier3 = KNeighborsClassifier(n_neighbors=int(configparms['hyper_parameters'][1]))
            classifier4 = MLPClassifier(alpha=1,hidden_layer_sizes=(95,75))
            classifier = VotingClassifier(estimators=[('rf',classifier1),('knn', classifier3), ('mlp', classifier4)], voting='soft', weights=configparms['hyper_parameters'][2:5],flatten_transform=True)
       
            
        classifier.fit(X_train,y_train.values.ravel())
        model_dispatcher[item] = classifier
        y_pred = classifier.predict(X_Validation)
        cm = confusion_matrix(y_Validation, y_pred)
        matthews_score_array.append(matthews_corrcoef(y_Validation, y_pred))
        f1_score_array.append(f1_score(y_Validation, y_pred))
        precision_array.append(precision_score(y_Validation, y_pred))
        recall_array.append(recall_score(y_Validation, y_pred))
        accuracy_array.append(accuracy_score(y_Validation, y_pred))
        reportCMTVEM.append("confusion matrix for "+item)
        print("confusion matrix for "+item)
        reportCMTVEM.append(cm)
        print(cm)
        
        # making the ROC plot: To make all curves for stable phases appear in a single plot saving the plot occurs after the loop 
        if configparms['roc_plot'] != 'none':
            fpr, tpr, _ = metrics.roc_curve(y_Validation, y_pred, pos_label=1)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[n_item], linestyle=linestyles[n_item], label='ROC curve of {0} (area = {1:0.2f})'''.format(phases[n_item], roc_auc))
            n_item=n_item+1
    #saving the roc curve in png format
    if configparms['roc_plot'] != 'none':
        plt.plot([0, 1], [0, 1], color='black', linestyle='-', linewidth=4.0)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic plot')
        plt.legend(loc="lower right")
        plt.savefig(configparms['roc_plot'])

    #recording the performance metrics
    reportCMTVEM.append("---------------------------------------------------------------")
    #mattews_score_array mean as the target of maximization of parameter tuning is passed to model dispatcher
    reportCMTVEM.append("average Matthews score")
    model_dispatcher['average_Matthews_score'] = np.mean(matthews_score_array)
    reportCMTVEM.append(model_dispatcher['average_Matthews_score'])
    reportCMTVEM.append("average f1 score")
    reportCMTVEM.append(np.mean(f1_score_array))
    reportCMTVEM.append("average precision score")
    reportCMTVEM.append(np.mean(precision_array))
    reportCMTVEM.append("average recall score")
    reportCMTVEM.append(np.mean(recall_array))
    reportCMTVEM.append("average accuracy score")
    reportCMTVEM.append(np.mean(accuracy_array))
    
    print("average Matthews score:", np.mean(matthews_score_array))
    print("average f1 score:", np.mean(f1_score_array))
    print("average precision score", np.mean(precision_array))
    print("average recall score:", np.mean(recall_array))
   
    #If we were to choose a useful metric, f1_score and Matthews score are the most relevant and statistically useful
    output_writer(reportCMTVEM,configparms)
    #model_dispatcher is a dictionary of all 9 models for 9 target variables
    return(model_dispatcher)


