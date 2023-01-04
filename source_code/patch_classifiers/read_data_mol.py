import pickle
import sys
import numpy as np
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import os
import h5py
import re
import csv

from glob import glob
import scipy
import pickle

import pandas as pd
import re
import random

from collections import Counter

#def load_image_path(path_init='/home/i3m4/Bengs/data/Bockmayr/mb_anno_300919.xlsx', data_Dir= '/home/i3m4/Bengs/data/',mdlParams={}):
def load_image_path(path_init='/home/Mukherjee/ProjectFiles/MasterProject/source_code/mb_anno_070721.xlsx', data_Dir= '/home/Mukherjee/MBlst/new_data_sets',mdlParams={}):

    loadParams = {}
    #print('the size ' ,mdlParams['input_file_dimension'])

    #print('coe')
    mdlParams['input_file_dimension']  = '4000'
    mdlParams['class_names'] = ['Klassisch', 'DesmoNodlaer']
    mdlParams['numClasses'] = len(mdlParams['class_names'])
    mdlParams['numOut'] = mdlParams['numClasses']
    mdlParams['Task'] = 'Histologischer Subtyp'





    # Load Excel Sheet for ground truth
    path = path_init
    loadParams['dataDir'] = data_Dir
    #print(loadParams['dataDir'])
    df = pd.read_excel(path)

    # Get the data path to image data
    loadParams['data_paths'] = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(loadParams['dataDir']):
        for file in f:
            if '.png' in file:
                    loadParams['data_paths'].append(os.path.join(r, file))

    # filter png images (get all the images that are cropped, done)
    loadParams['data_paths'] = [path for path in loadParams['data_paths'] if 'done' in path]
    #print("Number of png Images - done",len(loadParams['data_paths']))

    # Match labels with images
    Label_Column = mdlParams['Task'] # Define the Task / Labels that should be loaded from the excel sheet
    loadParams['data_paths_filtered'] = [] # Only Keep those with a match in excel sheet
    loadParams['data_paths_labels'] = []  # Only Keep those with a match in excel sheet
    loadParams['patient_ID'] = [] # row in excel sheet (used to match multiple images to a patient)
    loadParams['location'] = []
    loadParams['status'] = []
    all_image_names = [] # All images that have a match in the excel sheet
    image_no_match = []  # All images that have NO match in the excel sheet
    count = 0
    # Loop over images.png (done) and match description from excel sheet
    for i in range(len(loadParams['data_paths'])):

        s = loadParams['data_paths'][i] #
        #print(s)
        img_name = []
        img_name = re.search('done(.*)_t', s) # get the image name of the current path
        img_name =  img_name.group(1)
        img_name = img_name[1::]
        #print(img_name)
        Sanity_Check = 0
        # Check if present in one of the columns
        if df['WSI3'].str.contains(img_name).any():
            loadParams['data_paths_labels'].append(df.loc[df['WSI3'].str.contains(img_name)==True, Label_Column].iloc[0])
            loadParams['patient_ID'].append(df.loc[df['WSI3'].str.contains(img_name)==True, 'PatientID'].iloc[0])
            loadParams['data_paths_filtered'].append(s)
            loadParams['location'].append(df.loc[df['WSI3'].str.contains(img_name)==True, 'OP-Ort/ Diagnose-Ort'].iloc[0])
            loadParams['status'].append(df.loc[df['WSI3'].str.contains(img_name)==True, 'klin. Status'].iloc[0])
            all_image_names.append(img_name)
            #print(df.loc[df['WSI3'].str.contains(img_name)==True, 'OP-Ort/ Diagnose-Ort'].iloc[0])
            #print(df.loc[df['WSI3'].str.contains(img_name)==True, 'klin. Status'].iloc[0])
            #print(i, df.loc['klin. Status'])
            Sanity_Check = Sanity_Check +1

        elif df['WSI2'].str.contains(img_name).any():
            loadParams['data_paths_labels'].append(df.loc[df['WSI2'].str.contains(img_name)==True, Label_Column].iloc[0])
            loadParams['patient_ID'].append(df.loc[df['WSI2'].str.contains(img_name)==True, 'PatientID'].iloc[0])
            loadParams['data_paths_filtered'].append(s)
            loadParams['location'].append(df.loc[df['WSI2'].str.contains(img_name)==True, 'OP-Ort/ Diagnose-Ort'].iloc[0])
            loadParams['status'].append(df.loc[df['WSI2'].str.contains(img_name)==True, 'klin. Status'].iloc[0])
            all_image_names.append(img_name)
            Sanity_Check = Sanity_Check +1

        elif df['WSI1'].str.contains(img_name).any():
            loadParams['data_paths_labels'].append(df.loc[df['WSI1'].str.contains(img_name)==True, Label_Column].iloc[0])
            loadParams['patient_ID'].append(df.loc[df['WSI1'].str.contains(img_name)==True, 'PatientID'].iloc[0])
            loadParams['location'].append(df.loc[df['WSI1'].str.contains(img_name)==True, 'OP-Ort/ Diagnose-Ort'].iloc[0])
            loadParams['status'].append(df.loc[df['WSI1'].str.contains(img_name)==True, 'klin. Status'].iloc[0])
            loadParams['data_paths_filtered'].append(s)
            all_image_names.append(img_name)
            Sanity_Check = Sanity_Check +1

        else:
            #print(img_name)
            count = count +1
            image_no_match.append(img_name)


    # filter data (remove nan values, set same classes)
    #print('Patches with no match in excel sheet', count)
    #print(list(set(image_no_match)))
    #print(image_no_match)

    # throw out nan values (labels)
    label_filters = []
    path_filtered = []
    all_image_names_filtered = []
    patient_ID_filtered = []
    locations = []
    status = []
    #print(len(loadParams['data_paths_labels']))
    for i in range(len(loadParams['data_paths_labels'])): # class

        l = loadParams['data_paths_labels'][i]

        if  l == l: # throw out nan values (labels)
            label_filters.append(loadParams['data_paths_labels'][i])
            path_filtered.append(loadParams['data_paths_filtered'][i])
            all_image_names_filtered.append(all_image_names[i])
            patient_ID_filtered.append(loadParams['patient_ID'][i])
            locations.append(loadParams['location'][i])
            status.append(loadParams['status'][i])
    #print(len(locations), len(status))
    loadParams['all_image_names_filtered'] = all_image_names_filtered # name of .png data
    loadParams['data_paths_filtered'] = path_filtered # path to png data
    loadParams['data_paths_labels'] = label_filters # class name of png data
    loadParams['patient_ID'] = patient_ID_filtered # patient ID number
    loadParams['location'] = locations # location

    if mdlParams.get('location_filter',False):
        location_filter =  ['München', 'Hamburg', 'Göttingen', 'Bremen', 'Münster']  #mdlParams['locations']
    else:
        location_filter = loadParams['location']

    #print('location filter', location_filter)

    # set up label vectors
    num_clases = mdlParams['numClasses']

    labels_final = []
    data_path_final = []
    patient_ID_final = []
    image_names_final = []
    image_names_klassisch = []
    data_path_final_klassisch =[]
    image_names_others = []
    data_path_final_others =[]
    locations = []

    # Set up the learning task
    # Set up the label vector for the defined classes, e.g. binary
    # Filter out classes that are not relevant
    extra_label_counter = 0
    sap = 0
    #print('filtered data length ', len(loadParams['data_paths_filtered']))
    for i in range(len(loadParams['data_paths_filtered'])):

            l = loadParams['data_paths_labels'][i] # gives the class name for path
            current_location = loadParams['location'][i]
            #print(l)

            # only keep specific locations

            if current_location in location_filter:

                if Label_Column == 'Histologischer Subtyp':
                    # Klassisch vs rest (nicht typisierbar ist raus)
                    if num_clases ==2:
                        if l == 'Klassisch' or l =='klassisch':
                            labels_final.append([1, 0])
                            data_path_final.append(loadParams['data_paths_filtered'][i])
                            patient_ID_final.append(loadParams['patient_ID'][i])
                            locations.append(loadParams['location'][i])

                        elif  l == 'Desmoplastisch' or  l == 'Extensiv nodulär':
                            labels_final.append([0, 1])
                            data_path_final.append(loadParams['data_paths_filtered'][i])
                            patient_ID_final.append(loadParams['patient_ID'][i])
                            locations.append(loadParams['location'][i])

                        # elif l == 'Anaplastisch' or l == 'Desmoplastisch' or  l == 'Extensiv nodulär' or l == 'Desmoplastisch*'  or l == 'Großzellig/anapl.'or  l== 'Medullomyoblastom' or l == 'Desmoplastisch/Anaplastisch':

                        #     labels_final.append([0, 1])
                        #     data_path_final.append(loadParams['data_paths_filtered'][i])
                        #     patient_ID_final.append(loadParams['patient_ID'][i])
                        #     #image_names_final.append(loadParams['all_image_names_filtered'][i])
                        #     #image_names_others.append(loadParams['all_image_names_filtered'][i])
                        #     #data_path_final_others.append(loadParams['data_paths_filtered'][i])

                    if num_clases ==3:

                        if l == 'Klassisch' or l =='klassisch':

                            labels_final.append([1, 0, 0])
                            data_path_final.append(loadParams['data_paths_filtered'][i])
                            patient_ID_final.append(loadParams['patient_ID'][i])
                            #image_names_final.append(loadParams['all_image_names_filtered'][i])
                            #image_names_klassisch.append(loadParams['all_image_names_filtered'][i])
                            #data_path_final_klassisch.append(loadParams['data_paths_filtered'][i])
                            locations.append(loadParams['location'][i])

                        elif l == 'Desmoplastisch' or  l == 'Extensiv nodulär':

                            labels_final.append([0, 1, 0])
                            data_path_final.append(loadParams['data_paths_filtered'][i])
                            patient_ID_final.append(loadParams['patient_ID'][i])
                            #image_names_final.append(loadParams['all_image_names_filtered'][i])
                            #image_names_others.append(loadParams['all_image_names_filtered'][i])
                            #data_path_final_others.append(loadParams['data_paths_filtered'][i])
                            locations.append(loadParams['location'][i])


                        elif l == 'Anaplastisch'  or l == 'Großzellig/anapl.'   or l == 'Desmoplastisch/Anaplastisch' or l == 'Großzellig':

                            labels_final.append([0, 0, 1])
                            data_path_final.append(loadParams['data_paths_filtered'][i])
                            patient_ID_final.append(loadParams['patient_ID'][i])
                            #image_names_final.append(loadParams['all_image_names_filtered'][i])
                            #image_names_others.append(loadParams['all_image_names_filtered'][i])
                            #data_path_final_others.append(loadParams['data_paths_filtered'][i])
                            locations.append(loadParams['location'][i])



                elif Label_Column == 'Molekularer Subtyp (Method of detection)':
                    #print('Molekularer Subtyp (Method of detection)')
                    l = str(l)
                    if num_clases ==2:

                        if 'SHH' in l:
                            labels_final.append([1, 0])
                            data_path_final.append(loadParams['data_paths_filtered'][i])
                            patient_ID_final.append(loadParams['patient_ID'][i])
                            locations.append(loadParams['location'][i])
                            #image_names_final.append(loadParams['all_image_names_filtered'][i])
                            #image_names_klassisch.append(loadParams['all_image_names_filtered'][i])
                            #data_path_final_klassisch.append(loadParams['data_paths_filtered'][i])
                        elif  '3' in l or '4' in l:
                            if  'WNT' not in l and 'SSH' not in l:
                                #print(l)
                                labels_final.append([0, 1])
                                data_path_final.append(loadParams['data_paths_filtered'][i])
                                patient_ID_final.append(loadParams['patient_ID'][i])
                                locations.append(loadParams['location'][i])
                                #image_names_final.append(loadParams['all_image_names_filtered'][i])
                                #image_names_others.append(loadParams['all_image_names_filtered'][i])
                                #data_path_final_others.append(loadParams['data_paths_filtered'][i])

                        # if  'WNT' in l or 'SHH' in l:
                        #     labels_final.append([1, 0])
                        #     data_path_final.append(loadParams['data_paths_filtered'][i])
                        #     patient_ID_final.append(loadParams['patient_ID'][i])
                        #     #image_names_final.append(loadParams['all_image_names_filtered'][i])
                        #     #image_names_klassisch.append(loadParams['all_image_names_filtered'][i])
                        #     #data_path_final_klassisch.append(loadParams['data_paths_filtered'][i])
                        # elif  '3' in l or '4' in l:
                        #     if  'WNT' not in l or 'SSH' not in l:
                        #         #print(l)
                        #         labels_final.append([0, 1])
                        #         data_path_final.append(loadParams['data_paths_filtered'][i])
                        #         patient_ID_final.append(loadParams['patient_ID'][i])
                        #         #image_names_final.append(loadParams['all_image_names_filtered'][i])
                        #         #image_names_others.append(loadParams['all_image_names_filtered'][i])
                        #         #data_path_final_others.append(loadParams['data_paths_filtered'][i])
                    if num_clases ==3:


                        if  'WNT' in l:
                            labels_final.append([1, 0, 0])
                            data_path_final.append(loadParams['data_paths_filtered'][i])
                            patient_ID_final.append(loadParams['patient_ID'][i])
                            locations.append(loadParams['location'][i])
                            #image_names_final.append(loadParams['all_image_names_filtered'][i])
                            #image_names_klassisch.append(loadParams['all_image_names_filtered'][i])
                            #data_path_final_klassisch.append(loadParams['data_paths_filtered'][i])
                        elif  'SHH' in l :

                                #print(l)
                                #print(l)
                                labels_final.append([0, 1, 0])
                                data_path_final.append(loadParams['data_paths_filtered'][i])
                                patient_ID_final.append(loadParams['patient_ID'][i])
                                locations.append(loadParams['location'][i])
                                #image_names_final.append(loadParams['all_image_names_filtered'][i])
                                #image_names_others.append(loadParams['all_image_names_filtered'][i])
                                #data_path_final_others.append(loadParams['data_paths_filtered'][i])
                        elif  '4' in l or '3' in l:
                            if  'WNT' not in l and 'SHH' not in l:

                                #print(l)
                                #print(l)
                                labels_final.append([0, 0, 1])
                                data_path_final.append(loadParams['data_paths_filtered'][i])
                                patient_ID_final.append(loadParams['patient_ID'][i])
                                locations.append(loadParams['location'][i])
                                #image_names_final.append(loadParams['all_image_names_filtered'][i])
                                #image_names_others.append(loadParams['all_image_names_filtered'][i])
                                #data_path_final_others.append(loadParams['data_paths_filtered'][i])
                        else:
                            #print(l)
                            extra_label_counter += 1
                            #print('OUT OF 4', extra_label_counter)

                    if num_clases ==0:


                        if  'SHH' in l:
                            labels_final.append([1, 0, 0])
                            data_path_final.append(loadParams['data_paths_filtered'][i])
                            patient_ID_final.append(loadParams['patient_ID'][i])
                            locations.append(loadParams['location'][i])
                            #image_names_final.append(loadParams['all_image_names_filtered'][i])
                            #image_names_klassisch.append(loadParams['all_image_names_filtered'][i])
                            #data_path_final_klassisch.append(loadParams['data_paths_filtered'][i])
                        elif  '3' in l :

                                #print(l)
                                #print(l)
                                labels_final.append([0, 1, 0])
                                data_path_final.append(loadParams['data_paths_filtered'][i])
                                patient_ID_final.append(loadParams['patient_ID'][i])
                                locations.append(loadParams['location'][i])
                                #image_names_final.append(loadParams['all_image_names_filtered'][i])
                                #image_names_others.append(loadParams['all_image_names_filtered'][i])
                                #data_path_final_others.append(loadParams['data_paths_filtered'][i])
                        elif  '4' in l:
                            if '3/4' not in l:

                                #print(l)
                                #print(l)
                                labels_final.append([0, 0, 1])
                                data_path_final.append(loadParams['data_paths_filtered'][i])
                                patient_ID_final.append(loadParams['patient_ID'][i])
                                locations.append(loadParams['location'][i])
                                #image_names_final.append(loadParams['all_image_names_filtered'][i])
                                #image_names_others.append(loadParams['all_image_names_filtered'][i])
                                #data_path_final_others.append(loadParams['data_paths_filtered'][i])
                        else:
                            #print(l)
                            extra_label_counter += 1
                            #print('OUT OF 4', extra_label_counter)


                    if num_clases ==4:

                        if  'WNT' in l:
                            labels_final.append([1, 0, 0,0])
                            data_path_final.append(loadParams['data_paths_filtered'][i])
                            patient_ID_final.append(loadParams['patient_ID'][i])
                            locations.append(loadParams['location'][i])
                            #image_names_final.append(loadParams['all_image_names_filtered'][i])
                            #image_names_klassisch.append(loadParams['all_image_names_filtered'][i])
                            #data_path_final_klassisch.append(loadParams['data_paths_filtered'][i])

                        elif  'SHH' in l:
                            labels_final.append([0, 1, 0,0])
                            data_path_final.append(loadParams['data_paths_filtered'][i])
                            patient_ID_final.append(loadParams['patient_ID'][i])
                            locations.append(loadParams['location'][i])
                            #image_names_final.append(loadParams['all_image_names_filtered'][i])
                            #image_names_klassisch.append(loadParams['all_image_names_filtered'][i])
                            #data_path_final_klassisch.append(loadParams['data_paths_filtered'][i])
                        elif   '4' in l[:3]  :
                            #if '3' in l:# and '3/4' not in l:
                                #print(l)
                            #if  '3/4' not  in l:
                            #if  'WNT' not in l or 'SSH' not in l:
                                #print(l)
                                labels_final.append([0, 0,0, 1])
                                data_path_final.append(loadParams['data_paths_filtered'][i])
                                patient_ID_final.append(loadParams['patient_ID'][i])
                                locations.append(loadParams['location'][i])
                                #image_names_final.append(loadParams['all_image_names_filtered'][i])
                                #image_names_others.append(loadParams['all_image_names_filtered'][i])
                                #data_path_final_others.append(loadParams['data_paths_filtered'][i])


                        elif  '3' in l:

                            #if  '3/4' not in l:
                                #print(l)
                                #print(l)
                                labels_final.append([0, 0, 1,0])
                                data_path_final.append(loadParams['data_paths_filtered'][i])
                                patient_ID_final.append(loadParams['patient_ID'][i])
                                locations.append(loadParams['location'][i])
                                #image_names_final.append(loadParams['all_image_names_filtered'][i])
                                #image_names_others.append(loadParams['all_image_names_filtered'][i])
                                #data_path_final_others.append(loadParams['data_paths_filtered'][i])



                        else:
                            #print(l)
                            extra_label_counter += 1
                            #print('OUT OF 4', extra_label_counter)

                elif Label_Column == 'Follow-up [months]':
                    #print(type(l))

                    if status[i] == 'DOD' :#or 1:

                        l =  float(l)
                        if isinstance(l, (int,float)) :

                            if float(l) <= 60.0:
                                #print(l)
                                labels_final.append([1, 0])
                                data_path_final.append(loadParams['data_paths_filtered'][i])
                                patient_ID_final.append(loadParams['patient_ID'][i])
                                locations.append(loadParams['location'][i])
                                #pass
                            #else:
                            elif 0:
                                #print('Cant detect')
                                labels_final.append([0, 1])
                                data_path_final.append(loadParams['data_paths_filtered'][i])
                                patient_ID_final.append(loadParams['patient_ID'][i])
                                locations.append(loadParams['location'][i])

                        else:
                                print('removed DOD of', l)
                    else:


                        try:

                            l = float(l)
                        except:
                            #print('canot conver floar:', l)
                            if 'ca.' in l:
                                l = float(l[4:])
                            #print(l)
                        if  l > 60:
                            labels_final.append([0, 1])
                            data_path_final.append(loadParams['data_paths_filtered'][i])
                            patient_ID_final.append(loadParams['patient_ID'][i])
                            locations.append(loadParams['location'][i])
    num_ones =[]
    num_zeros =[]
    class_weight =[]

    #print('Total dod',sap)
    for i in range(num_clases):
       #print(i, 'class', end = '  ')
       result1 = len(list(filter(lambda x: x[i]== 1, labels_final)))
       num_ones.append(result1)
       result2 = len(labels_final)-result1
       num_zeros.append(result2)

       #class_weight.append(result2/result1)
    #print(num_ones, num_zeros,extra_label_counter)#
    #
    #  extra_label_counter)
    #print(len(labels_final))
    # only save / keep relevant classes
    loadParams['labels'] = labels_final
    loadParams['data_paths_filtered'] = data_path_final
    loadParams['patient_ID'] = patient_ID_final
    loadParams['patient_ID_unique'] = list(set(loadParams['patient_ID']))
    loadParams['location'] = locations
    #print('Length of locations: ', locations[:5], len(locations))
    # count the number of patches per patient, and the class frequency for patients
    num_patches_ID = np.zeros(len(loadParams['patient_ID_unique']))
    # Get unique image names & corresponding label
    image_names = []
    image_names_labels = []
    location_patient_based = []
    #print('Num uniue ', len(loadParams['patient_ID']), loadParams['patient_ID_unique'])

    for j in range(len(loadParams['patient_ID_unique'])):
        count = 0
        #i  = 0
        for i in range(len(loadParams['patient_ID'])):
            if loadParams['patient_ID'][i] == loadParams['patient_ID_unique'][j]:
                count +=1
                num_patches_ID[j] = count
                #print(i, j, loadParams['patient_ID'][i] ,image_names)
                if loadParams['patient_ID'][i] not in image_names:
                    #print(i, loadParams['patient_ID'][i] ,loadParams['labels'][i] )
                    image_names.append(loadParams['patient_ID'][i] ) # patient IDs for filterting
                    image_names_labels.append(loadParams['labels'][i])  # only save one label per patient
                    location_patient_based.append(loadParams['location'][i])  # only save one location per patient

    #print('Number of Patients from the different Sides')
    #print(Counter(location_patient_based))


    # filter based on locations
    # only keep locations with more than x patients



    loadParams['num_patches_ID_unique']  = num_patches_ID
    # Describes the number of extracted patches per patient and the corresponding label
    loadParams['patient_ID_unique_summary'] = [loadParams['patient_ID_unique'], loadParams['num_patches_ID_unique'], image_names_labels]

    # Get unique classes in labels
    output = []
    for x in loadParams['data_paths_labels']: # names of different classes
        if x not in output:
            output.append(x)

    print('------------ Classes -------------')
    print('Number of Unique Patients', len(loadParams['patient_ID_unique']))
    #print(output)
    print('Number of patches all', len(loadParams['data_paths_filtered']))
    print('Label Frequency of patches: ', np.sum(np.array(loadParams['labels']), axis=0))
    print('Label Frequency of images given patients: ', np.sum(np.array(image_names_labels), axis=0))

    return np.array(loadParams['labels']), loadParams['data_paths_filtered'], loadParams['patient_ID'], loadParams['patient_ID_unique_summary']


#labels, datapaths, patient_id, patient_id_summary = load_image_path(data_Dir = '/home/Mukherjee/MBlst/new_data_sets/2000/1979/done/')
#print('Labels', labels[0:5])
#print('Data Paths', datapaths[0:5])
#print('patient_id', patient_id[0:5])
#print('patient_id_summary', patient_id_summary[0:5])


#def load_image_path
