import os
import sys
import h5py
import re
import csv
import numpy as np

from glob import glob
import scipy
import pickle

import read_data_mol as read_data


def init(mdlParams_):
    #print('I was gherher')
    mdlParams = {}


    #######################################
    #change config as required
    ### Model Selection ###
    ##select one of
    ##1 : 'efficientnet' 
    ##2 : 'xcit'
    ##3 : "vit'

    mdlParams['multiple_mag'] = True
    mdlParams['model_type'] =  'efficient'
    ## num of crops for evaluation
    mdlParams['multiCropEval'] = 1
    
    #### model input dimension [224,224,3] for efficient-netB0,  [384,384,3] for XCiT and ViT
    

    mdlParams['input_size_load'] = [224,224,3] # Intermediate Resolution value #direct change for test




    ###################################
    if 'efficient' in  mdlParams['model_type']:
        mdlParams['model_type'] =  'efficientnet_b0'
        mdlParams['input_size'] = [224,224,3] 
    elif 'xcit' in  mdlParams['model_type']:
        mdlParams['model_type'] =  'xcit_medium_24_p16_384_dist'
        mdlParams['input_size'] = [384,384,3] 
    elif 'vit' in  mdlParams['model_type']:
        mdlParams['model_type'] =  'vit_base_r50_s16_384'
        mdlParams['input_size'] = [384,384,3] 
    else:
        assert 0

    # Save summaries and model here
    mdlParams['saveDir'] = mdlParams_['pathBase']+'/models/'
    # Data is loaded from here
    #mdlParams['dataDir'] = mdlParams_['pathBase']+'/data/'
    
    # Number of GPUs on device
    mdlParams['numGPUs'] = [1]
    
    #mdlParams['model_type'] =  'efficientnet_b0' #'xcit_medium_24_p16_384_dist' # "vit_base_r50_s16_384" #'vit_base_r50_s16_384'# "efficientnet_b0"
    #mdlParams['model_type'] = 'vit_tiny_r_s16_p8_384'#'vit_base_r50_s16_384'#'efficientnet_b0'#'xcit_medium_24_p16_384_dist'#'efficientnet_b0'#'xcit_medium_24_p16_384_dist'#'efficientnet_b0'#'xcit_medium_24_p16_384_dist'#"efficientnet_b0"

    ## chanfged fron 4
    

  
    mdlParams['orderedCrop'] = True

    mdlParams['same_sized_crops'] = True
    mdlParams['voting_scheme'] = 'average'
    mdlParams['average_for_patients'] = False
    mdlParams['Freeze_Layer'] = False
    mdlParams['Train_Layer'] = 9 # only when Freeze = True
    mdlParams['Child_Counter'] = 1 # req. for selecting the layers (dont change)
    # A type of class balancing. Available:
    # 0: No balancing
    # 1: Use inverse class frequency
    # 2: Balanced batch sampling
    # 3: Balanced batch sampling (repeat underrepresented examples)
    # 4: Diagnosis weighting + inverse class frequency
    mdlParams['balance_classes'] = 1
    mdlParams['extra_fac'] = 1.0
    mdlParams['trainSetState'] = 'train'

    mdlParams['use_and_mix_all'] = False
    mdlParams['use_and_mix_all_split'] = 0.20

    mdlParams['setMean'] = np.array([0,0,0])
 
    ### Training Parameters ###
    # Batch size
    mdlParams['patient_based_batching'] = True
    mdlParams['full_color_distort'] = True

    mdlParams['batchSize'] = 20#*len(mdlParams['numGPUs']) #changed batch size from 40
    # Initial learning rate
    mdlParams['learning_rate'] = 0.0001#*len(mdlParams['numGPUs'])
    # Lower learning rate after no improvement over 100 epochs
    mdlParams['lowerLRAfter'] = 200
    # If there is no validation set, start lowering the LR after X steps
    mdlParams['lowerLRat'] = 200
    mdlParams['LRstep'] = 5 #changed from 2 to 10 because of poor convergence during traning
    # Divide learning rate by this value
    # Maximum number of training iterations
    mdlParams['training_steps'] = 300#300
    # Display error every X steps
    mdlParams['display_step'] = 400
    # Peak at test error during training? (generally, dont do this!)
    mdlParams['peak_at_testerr'] = False
    # Print trainerr
    mdlParams['print_trainerr'] = False
    # Subtract trainset mean?
    mdlParams['subtract_set_mean'] = False

    ### Data ###
    mdlParams['preload'] = False
    
    mdlParams['image_resize'] = True
    # Save all im paths here
    mdlParams['im_paths'] = []
    mdlParams['labels_list'] = []

    # First: get all paths into dict
    # Create labels array
    
    # Use this for ordered multi crops
    if mdlParams['orderedCrop']:
        # Crop positions, always choose multiCropEval to be 4, 9, 16, 25, etc.
        if mdlParams['multiCropEval'] ==1:
            mdlParams['cropPositions'] = np.zeros([mdlParams['multiCropEval'],2],dtype=np.int64)
            ind = 0
            mdlParams['cropPositions'][ind,0] = mdlParams['input_size'][0]/2
            mdlParams['cropPositions'][ind,1] = mdlParams['input_size'][1]/2

        else:
            mdlParams['cropPositions'] = np.zeros([mdlParams['multiCropEval'],2],dtype=np.int64)
            ind = 0
            for i in range(np.int32(np.sqrt(mdlParams['multiCropEval']))):
                for j in range(np.int32(np.sqrt(mdlParams['multiCropEval']))):
                    mdlParams['cropPositions'][ind,0] = mdlParams['input_size'][0]/2+i*((mdlParams['input_size_load'][0]-mdlParams['input_size'][0])/(np.sqrt(mdlParams['multiCropEval'])-1))
                    mdlParams['cropPositions'][ind,1] = mdlParams['input_size'][1]/2+j*((mdlParams['input_size_load'][1]-mdlParams['input_size'][1])/(np.sqrt(mdlParams['multiCropEval'])-1))
                    ind += 1

        # Sanity checks
        #print("Positions",mdlParams['cropPositions'])
        # Test image sizes
        test_im = np.zeros(mdlParams['input_size_load'])
        height = mdlParams['input_size'][0]
        width = mdlParams['input_size'][1]
        for i in range(mdlParams['multiCropEval']):
            im_crop = test_im[np.int32(mdlParams['cropPositions'][i,0]-height/2):np.int32(mdlParams['cropPositions'][i,0]-height/2)+height,np.int32(mdlParams['cropPositions'][i,1]-width/2):np.int32(mdlParams['cropPositions'][i,1]-width/2)+width,:]
            #print("Shape",i+1,im_crop.shape)

    return mdlParams
