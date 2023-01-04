import os
import sys
import h5py
import re
import csv
import numpy as np

from glob import glob
import scipy
import pickle
import pandas as pd
import read_data_mol as read_data

def init(mdlParams_):

    mdlParams = {}
    #print()
    # Save summaries and model here

    #mdlParams_['pathBase'] = '/mnt/turing1/satish/'
    #mdlParams['saveDir'] = mdlParams_['pathBase']+'/models/'
    mdlParams['saveDir'] = mdlParams_['pathBase']+'/models/'
    mdlParams['input_file_dimension'] =     mdlParams_['input_file_dimension']
    mdlParams['use_and_mix_all'] = False
    mdlParams['use_and_mix_all_split'] = 0.20

    mdlParams['setMean'] = np.array([0,0,0])


    # Save all im paths here
    task = mdlParams_['task_num']
    if task==0:
        mdlParams['class_names'] = ['Klassisch', 'DesmoNodlaer']
        mdlParams['numClasses'] = len(mdlParams['class_names'])
        mdlParams['numOut'] = mdlParams['numClasses']
        mdlParams['Task'] = 'Histologischer Subtyp'
        mdlParams['data_split'] = [10,5]


    elif task == 1:
        mdlParams['class_names'] = ['WNT', 'SHH', '3/4']
        mdlParams['numClasses'] = len(mdlParams['class_names'])
        mdlParams['numOut'] = mdlParams['numClasses']
        mdlParams['Task'] = 'Molekularer Subtyp (Method of detection)'




    elif task == 2:
        mdlParams['class_names'] = ['DEAD',  'others']
        mdlParams['numClasses'] = len(mdlParams['class_names'])
        mdlParams['numOut'] = mdlParams['numClasses']
        mdlParams['Task'] = 'Follow-up [months]'


    print('The task: ', mdlParams['Task'], ' Classes: ',mdlParams['class_names'] )
    # First: get all paths into dict
    # Create labels array
    mdlParams['im_paths'] = []
    mdlParams['labels_list'] = []
    print('Image path--',mdlParams_['data_dir'])
    mdlParams['labels_array'], mdlParams['im_paths'], mdlParams['patientID'], mdlParams['patientID_unique']  = \
    read_data.load_image_path(path_init= mdlParams_['meta_data'], data_Dir= mdlParams_['data_dir'], mdlParams = mdlParams)
    #print("Labels shape",mdlParams['labels_array'].shape)

    num_images = len(mdlParams['patientID_unique'][0])
    print('number of different patietns', num_images)
    label_frequency = np.sum(np.array(mdlParams['patientID_unique'][2]))

    # 15 Patients as Val and Test Set each

    print('Number of different Patients in Data Set', num_images)
    #print('Path Files', mdlParams['im_paths'])

    # Differentiate: randomize and mix, single validation, cross-validation
    # Use Single Split for now
    mdlParams['trainIndCV'] = []
    mdlParams['valIndCV'] = []
    mdlParams['testIndCV'] =[]

    mdlParams['Val_numPatches_uniqueCV'] = []
    mdlParams['Test_numPatches_uniqueCV'] = []
    mdlParams['Train_numPatches_uniqueCV'] = []

    mdlParams['Val_Label_uniqueCV'] = []
    mdlParams['Test_Label_uniqueCV'] = []
    mdlParams['Train_Label_uniqueCV'] = []

    mdlParams['ValInd_ID_uniqueCV'] = []
    mdlParams['TestInd_ID_uniqueCV'] = []
    mdlParams['TrainInd_ID_uniqueCV'] = []

    val_path =[]
    train_path =[]
    test_path =[]

    mdlParams['numCV'] = 1

    patientID_unique_b = np.array(mdlParams['patientID_unique'][0])
    patientnumPatches_unique_b =  np.array(mdlParams['patientID_unique'][1])
    patientID_Label_unique_b =  np.array(mdlParams['patientID_unique'][2])
    print('Patient Id',mdlParams['patientID_unique'][0],len(mdlParams['patientID_unique'][0]))

    # Shuffle Data
    #print('new shuffle 3')
    cv_shuffle = np.array([40,24,95,117,103,85,47,37,152,96,126,62,23,87,48,84,127,18,92,100,7,65,169,105,19,162,173,108,168,121,111,151,72,153,164,109,136,31,46,150,129,33,146,63,58,78,125,137,145,5,67,115,93,64,99,106,49,53,140,55,76,73,128,149,124,101,59,44,154,43,156,94,131,27,11,102,66,22,0,10,26,158,39,32,54,155,138,144,81,147,110,14,57,160,134,98,41,139,56,38,80,83,170,91,165,163,29,166,132,82,6,174,133,79,25,2,142,36,114,71,9,148,1,42,51,16,35,120,34,74,89,17,107,116,104,122,130,13,141,112,45,50,171,97,15,157,119,90,20,4,3,12,118,69,75,159,68,113,77,61,86,60,172,123,21,8,88,135,30,70,28,52,167,161,143])
    print(len(cv_shuffle))

    cv_shuffle_new = []

    for index in cv_shuffle:
        if index < int(patientID_unique_b.shape[0]):
            cv_shuffle_new.append(index)

    cv_shuffle = np.array(cv_shuffle_new)
    print(len(cv_shuffle))

    #print(len(patientID_unique_b))
    patientID_unique = patientID_unique_b[cv_shuffle]
    patientnumPatches_unique =  patientnumPatches_unique_b[cv_shuffle]
    patientID_Label_unique =  patientID_Label_unique_b[cv_shuffle]
    ID_list_CV = np.arange(num_images)

    cv_test_dict= {}
    cv_train_dict= {}
    for cv in range(mdlParams['numCV']):
        #print('CV:', cv)
        val_path =[]
        train_path =[]
        test_path =[]

        # Define randomized inds
        # get validation path

        mdlParams['ValInd_Patient_List'] =  []
        mdlParams['TestInd_Patient_List'] =  []
        mdlParams['TrainInd_Patient_List'] =  []

        c_label = [0]*mdlParams['numClasses']
        c_t_label = [0]*mdlParams['numClasses']
        for i in ID_list_CV: # loop over all patient IDs
                # Val Set

            # Split the patients based on the distrubition of the training data into test and val sets
            current = patientID_Label_unique[i]
            #print('Current', current, i)
            if mdlParams['Task'] == 'Histologischer Subtyp':
                if mdlParams['numClasses']==2:
                    if current[0]==1 and c_label[0] < 5:
                            c_label = c_label + patientID_Label_unique[i]
                            mdlParams['ValInd_Patient_List'].append(i)
                    elif current[1]==1 and c_label[1] < 2:
                            c_label = c_label + patientID_Label_unique[i]
                            mdlParams['ValInd_Patient_List'].append(i)

                    elif current[0]==1 and c_t_label[0] < 5:
                            c_t_label = c_t_label + patientID_Label_unique[i]
                            mdlParams['TestInd_Patient_List'].append(i)
                    elif current[1]==1 and c_t_label[1] < 2:
                            c_t_label = c_t_label + patientID_Label_unique[i]
                            mdlParams['TestInd_Patient_List'].append(i)
                    else:
                        mdlParams['TrainInd_Patient_List'].append(i)
                elif mdlParams['numClasses']==3:
                    if current[0]==1 and c_label[0] < 0:
                            c_label = c_label + patientID_Label_unique[i]
                            mdlParams['ValInd_Patient_List'].append(i)
                    elif current[1]==1 and c_label[1] < 0:
                            c_label = c_label + patientID_Label_unique[i]
                            mdlParams['ValInd_Patient_List'].append(i)
                    elif current[2]==1 and c_label[2] < 0:
                            c_label = c_label + patientID_Label_unique[i]
                            mdlParams['ValInd_Patient_List'].append(i)

                    elif current[0]==1 and c_t_label[0] < 11:
                            c_t_label = c_t_label + patientID_Label_unique[i]
                            mdlParams['TestInd_Patient_List'].append(i)
                    elif current[1]==1 and c_t_label[1] < 5:
                            c_t_label = c_t_label + patientID_Label_unique[i]
                            mdlParams['TestInd_Patient_List'].append(i)
                    elif current[2]==1 and c_t_label[2] < 1:
                            c_t_label = c_t_label + patientID_Label_unique[i]
                            mdlParams['TestInd_Patient_List'].append(i)
                    else:
                        mdlParams['TrainInd_Patient_List'].append(i)
            elif mdlParams['Task'] == 'Molekularer Subtyp (Method of detection)':
                if mdlParams['numClasses'] ==2:
                    if current[0]==1 and c_label[0] < 2:
                            c_label = c_label + patientID_Label_unique[i]
                            mdlParams['ValInd_Patient_List'].append(i)
                    elif current[1]==1 and c_label[1] < 3:
                            c_label = c_label + patientID_Label_unique[i]
                            mdlParams['ValInd_Patient_List'].append(i)

                    elif current[0]==1 and c_t_label[0] < 2:
                            c_t_label = c_t_label + patientID_Label_unique[i]
                            mdlParams['TestInd_Patient_List'].append(i)
                    elif current[1]==1 and c_t_label[1] < 4:
                            c_t_label = c_t_label + patientID_Label_unique[i]
                            mdlParams['TestInd_Patient_List'].append(i)

                    else:
                        mdlParams['TrainInd_Patient_List'].append(i)
                elif mdlParams['numClasses'] ==3:
                    if current[0]==1 and c_label[0] < 1:
                            c_label = c_label + patientID_Label_unique[i]
                            mdlParams['ValInd_Patient_List'].append(i)
                    elif current[1]==1 and c_label[1] < 2:
                            c_label = c_label + patientID_Label_unique[i]
                            mdlParams['ValInd_Patient_List'].append(i)
                    elif current[2]==1 and c_label[2] < 3:
                            c_label = c_label + patientID_Label_unique[i]
                            mdlParams['ValInd_Patient_List'].append(i)

                    elif current[0]==1 and c_t_label[0] < 1:
                            c_t_label = c_t_label + patientID_Label_unique[i]
                            mdlParams['TestInd_Patient_List'].append(i)
                    elif current[1]==1 and c_t_label[1] < 2:
                            c_t_label = c_t_label + patientID_Label_unique[i]
                            mdlParams['TestInd_Patient_List'].append(i)
                    elif current[2]==1 and c_t_label[2] < 3:
                            c_t_label = c_t_label + patientID_Label_unique[i]
                            mdlParams['TestInd_Patient_List'].append(i)


                    else:
                        mdlParams['TrainInd_Patient_List'].append(i)

            elif mdlParams['Task'] == 'Follow-up [months]':


                if current[0]==1 and c_label[0] < 1:
                        c_label = c_label + patientID_Label_unique[i]
                        mdlParams['ValInd_Patient_List'].append(i)
                elif current[1]==1 and c_label[1] < 2:
                        c_label = c_label + patientID_Label_unique[i]
                        mdlParams['ValInd_Patient_List'].append(i)

                elif current[0]==1 and c_t_label[0] < 2:
                        c_t_label = c_t_label + patientID_Label_unique[i]
                        mdlParams['TestInd_Patient_List'].append(i)
                elif current[1]==1 and c_t_label[1] < 2:
                        c_t_label = c_t_label + patientID_Label_unique[i]
                        mdlParams['TestInd_Patient_List'].append(i)
                else:
                    mdlParams['TrainInd_Patient_List'].append(i)

            else:
                if current[0]==1 and c_label[0] < 1:
                        c_label = c_label + patientID_Label_unique[i]
                        mdlParams['ValInd_Patient_List'].append(i)
                elif current[1]==1 and c_label[1] < 2:
                        c_label = c_label + patientID_Label_unique[i]
                        mdlParams['ValInd_Patient_List'].append(i)

                elif current[0]==1 and c_t_label[0] < 2:
                        c_t_label = c_t_label + patientID_Label_unique[i]
                        mdlParams['TestInd_Patient_List'].append(i)
                elif current[1]==1 and c_t_label[1] < 2:
                        c_t_label = c_t_label + patientID_Label_unique[i]
                        mdlParams['TestInd_Patient_List'].append(i)
                else:
                    mdlParams['TrainInd_Patient_List'].append(i)



        if cv>0:
            mdlParams['TrainInd_Patient_List'] = mdlParams['TrainInd_Patient_List'] + test_val_ind # test_val_ind from previous fold
            test_val_ind = test_val_ind + mdlParams['ValInd_Patient_List'] + mdlParams['TestInd_Patient_List']
        else:
            test_val_ind = mdlParams['ValInd_Patient_List'] + mdlParams['TestInd_Patient_List']

        ID_list_CV = np.setdiff1d(np.arange(num_images),test_val_ind) # all the samples that have not been used for validation or testing so far
        #print('Still left', ID_list_CV.shape[0] )
        #print('test list', mdlParams['TestInd_Patient_List'] )
        #mdlParams['ValInd_Patient_List'] =  np.array(mdlParams['ValInd_Patient_List'])
        mdlParams['ValInd_Patient_List'] =  np.array(mdlParams['ValInd_Patient_List']) ##edited
        mdlParams['TestInd_Patient_List'] =  np.array(mdlParams['TestInd_Patient_List'])
        mdlParams['TrainInd_Patient_List'] =  np.array(mdlParams['TrainInd_Patient_List'])
        #print('TestInd_Patient_List', mdlParams['TestInd_Patient_List'])
        # get the patient index for excel sheet
        mdlParams['ValInd_ID_unique'] = patientID_unique[mdlParams['ValInd_Patient_List']] ##edited
        mdlParams['TestInd_ID_unique'] = patientID_unique[mdlParams['TestInd_Patient_List']]
        #print('lenTestInd_ID_unique',len(mdlParams['TestInd_ID_unique']))
        #print('TestInd_ID_unique', mdlParams['TestInd_ID_unique'])
        #print('**ValInd_ID_unique', mdlParams['ValInd_ID_unique'])
        mdlParams['TrainInd_ID_unique'] = patientID_unique[mdlParams['TrainInd_Patient_List']]

        mdlParams['Val_numPatches_unique'] = patientnumPatches_unique[mdlParams['ValInd_Patient_List']]  ##edited
        mdlParams['Test_numPatches_unique'] = patientnumPatches_unique[mdlParams['TestInd_Patient_List']]
        mdlParams['Train_numPatches_unique'] = patientnumPatches_unique[mdlParams['TrainInd_Patient_List']]

        mdlParams['Val_Label_unique'] = patientID_Label_unique[mdlParams['ValInd_Patient_List']]  ##edited
        mdlParams['Test_Label_unique'] = patientID_Label_unique[mdlParams['TestInd_Patient_List']]
        mdlParams['Train_Label_unique'] = patientID_Label_unique[mdlParams['TrainInd_Patient_List']]

        #print('Label Frequency of Val given patients: ', np.sum(np.array(mdlParams['Val_Label_unique']), axis=0))
        #print('Label Frequency of Test given patients: ', np.sum(np.array(mdlParams['Test_Label_unique']), axis=0))
        #print('Label Frequency of Train given patients: ', np.sum(np.array(mdlParams['Train_Label_unique']), axis=0))

        # set up index arrays for subimage paths and label arrays

        for i in mdlParams['ValInd_ID_unique']:
            im_name = int(i)
            #print(im_name)
            for k in range(len(mdlParams['patientID'])):
                if im_name == mdlParams['patientID'][k]:
                    val_path.append(k)

        # get test path
        for i in mdlParams['TestInd_ID_unique']:
            im_name = int(i)
            for k in range(len(mdlParams['patientID'])):
                if im_name == mdlParams['patientID'][k]:
                    test_path.append(k)
                    #print('Test')

        # get training path
        for i in mdlParams['TrainInd_ID_unique']:
            im_name = int(i)
            for k in range(len(mdlParams['patientID'])):
                if im_name == mdlParams['patientID'][k]:
                    train_path.append(k)


                    #print('Train')

        size_val = len(val_path)
        mdlParams['valIndCV'].append(np.array(val_path))
        mdlParams['testIndCV'].append(np.array(test_path))
        mdlParams['trainIndCV'].append(np.array(train_path))

        mdlParams['Val_numPatches_uniqueCV'].append(np.array(mdlParams['Val_numPatches_unique']))
        mdlParams['Test_numPatches_uniqueCV'].append(np.array(mdlParams['Test_numPatches_unique']))
        mdlParams['Train_numPatches_uniqueCV'].append(np.array(mdlParams['Train_numPatches_unique']))

        mdlParams['Val_Label_uniqueCV'].append(np.array(mdlParams['Val_Label_unique']))
        mdlParams['Test_Label_uniqueCV'].append(np.array(mdlParams['Test_Label_unique']))
        mdlParams['Train_Label_uniqueCV'].append(np.array(mdlParams['Train_Label_unique']))

        mdlParams['ValInd_ID_uniqueCV'].append(mdlParams['ValInd_ID_unique'])
        mdlParams['TestInd_ID_uniqueCV'].append(mdlParams['TestInd_ID_unique'])
        mdlParams['TrainInd_ID_uniqueCV'].append(mdlParams['TrainInd_ID_unique'])

        cv_test_dict['cv_'+str(cv)+'_test_pat_id'] = list(mdlParams['TestInd_ID_unique'])

        cv_test_dict['cv_'+str(cv)+'_test_label'] = list(mdlParams['Test_Label_unique'])

        cv_train_dict['cv_'+str(cv)+'_train_pat_id'] = list(mdlParams['TrainInd_ID_unique'])
        cv_train_dict['cv_'+str(cv)+'_train_label'] = list(mdlParams['Train_Label_unique'])

        #cv_dict['cv_'+str(cv)+'_test_label'] = mdlParams['TestInd_ID_unique']

    #print(cv_dict)
    #new_df_test = pd.DataFrame(cv_test_dict)
    #new_df_train = pd.DataFrame(cv_train_dict)
    #new_df_test.to_csv('histo_3class_test.csv', index= True)
    #new_df_train.to_csv('histo_3class_train.csv', index= True)


    # Ind properties
    '''
    print("Train")
    for i in range(len(mdlParams['trainIndCV'])):
        print(mdlParams['trainIndCV'][i].shape)
        print(np.sum(mdlParams['labels_array'][mdlParams['trainIndCV'][i],:],0))
    print("Val")
    for i in range(len(mdlParams['valIndCV'])):
        print(mdlParams['valIndCV'][i].shape)
        print(np.sum(mdlParams['labels_array'][mdlParams['valIndCV'][i],:],0))
        #print("Intersect",np.intersect1d(mdlParams['trainIndCV'][i],mdlParams['valIndCV'][i]))
    print("Test")
    for i in range(len(mdlParams['testIndCV'])):
        print(mdlParams['testIndCV'][i].shape)
        print(np.sum(mdlParams['labels_array'][mdlParams['testIndCV'][i],:],0))
        #print("Intersect",np.intersect1d(mdlParams['trainIndCV'][i],mdlParams['testIndCV'][i]))
    '''
    return mdlParams
