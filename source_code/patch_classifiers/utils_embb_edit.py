#print(1/0)
import os
import torch
#import pandas as pd
from skimage import io, transform
import scipy
import numpy as np
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, utils
from sklearn.metrics import confusion_matrix, auc, roc_curve, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
import math
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import types
import models
from torchvision import models as tvmodels
from visdom import Visdom
from PIL import ImageFile
import importlib
ImageFile.LOAD_TRUNCATED_IMAGES = True

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def image2grayscale(image):
    rgb_weights = [0.2989, 0.5870, 0.1140]
    grayscale_image = np.dot(image[...,:3], rgb_weights)
    grayscale_image = np.expand_dims(grayscale_image, axis=2)
    grayscale_image = np.repeat(grayscale_image, 3, 2)
    return grayscale_image

# Define ISIC Dataset Class
class Bockmayr_DataSet(Dataset):
    """Digital Pathology Data Set"""

    def __init__(self, mdlParams, indSet):
        """
        Args:
            mdlParams (dict): Configuration for loading
            indSet (string): Indicates train, val, test
        """
        # Number of classes
        self.numClasses = mdlParams['numClasses']
        # Model input size
        self.input_size = (np.int32(mdlParams['input_size'][0]),np.int32(mdlParams['input_size'][1]))
        # Load size/downsampled size, reversed order for loading
        self.input_size_load = (np.int32(mdlParams['input_size_load'][1]),np.int32(mdlParams['input_size_load'][0]))
        # Whether or not to use ordered cropping
        self.orderedCrop = mdlParams['orderedCrop']
        # Number of crops for multi crop eval
        self.multiCropEval = mdlParams['multiCropEval']
        # Whether during training same-sized crops should be used
        self.same_sized_crop = mdlParams['same_sized_crops']
        # Potential class balancing option
        self.balancing = mdlParams['balance_classes']
        # Whether data should be preloaded
        self.preload = mdlParams['preload']
        # Potentially subtract a mean
        self.subtract_set_mean = mdlParams['subtract_set_mean']
        # Potential switch for evaluation on the training set
        self.train_eval_state = mdlParams['trainSetState']
        # Potential setMean to deduce from channels
        self.setMean = mdlParams['setMean'].astype(np.float32)
        # Current indSet = 'trainInd'/'valInd'/'testInd'
        self.indices = mdlParams[indSet]
        self.indSet = indSet
        # Image_Resize
        self.resize_image  = mdlParams.get('image_resize',False)
        # Patient based Batchting
        self.patient_based_batching = mdlParams.get('patient_based_batching',False)
        self.gray_scale = mdlParams.get('gray_scale',False)
        self.croplocation = mdlParams['cropPositions']
        # Cropping during training?
        if mdlParams['input_size'][0] == mdlParams['input_size_load'][0]:
            self.same_sized_input = True
        else:
            self.same_sized_input = False
        # Multimag is true or not
        if mdlParams['multiple_mag'] == True:
            self.multiple_mag = True
        else:
            self.multiple_mag = False
           

        #Idx to match others
        

        if self.orderedCrop and (indSet == 'valInd' or self.train_eval_state  == 'eval' or indSet == 'testInd'):
            # Complete labels array, only for current indSet, repeat for multiordercrop
            inds_rep = np.repeat(mdlParams[indSet], mdlParams['multiCropEval'])
            self.labels = mdlParams['labels_array'][inds_rep,:]
            # Path to images for loading, only for current indSet, repeat for multiordercrop
            self.im_paths = np.array(mdlParams['im_paths'])[inds_rep].tolist()
            # Set up crop positions for every sample
            self.cropPositions = np.tile(mdlParams['cropPositions'], (mdlParams[indSet].shape[0],1))
            #print("CP Example",self.cropPositions[0:len(mdlParams['cropPositions']),:])
            # Set up transforms
            self.norm =  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.trans = transforms.ToTensor()
            self.resize =  transforms.Resize(mdlParams['input_size_load'][1])
            self.gray_scale_trans =  transforms.Grayscale(num_output_channels=3)

            # Potentially prel oad
            if self.preload:
                self.im_list = []
                for i in range(len(self.im_paths)):
                    temp = Image.open(self.im_paths[i])
                    keep = temp.copy()
                    temp.close()
                    # Resize
                    if self.resize_image == True:
                        keep = self.resize(keep)
                    self.im_list.append(keep)
                    #print('yeta mathi ko')
                    printProgressBar(i,len(self.im_paths), prefix = 'Pre-Load Eval Data:', suffix = 'Complete', length = 50)

        elif indSet == 'valInd' or indSet == 'testInd':
            if self.multiCropEval == 0:
                #print('multi ===0=====')
                self.cropping = transforms.Compose([transforms.CenterCrop(np.int32(self.input_size[0]*1.5)),transforms.Resize(self.input_size)])
                # Complete labels array, only for current indSet
                self.labels = mdlParams['labels_array'][mdlParams[indSet],:]
                # Path to images for loading, only for current indSet
                self.im_paths = np.array(mdlParams['im_paths'])[mdlParams[indSet]].tolist()
            else:
                self.cropping = transforms.RandomResizedCrop(self.input_size[0])
                # Complete labels array, only for current indSet, repeat for multiordercrop
                inds_rep = np.repeat(mdlParams[indSet], mdlParams['multiCropEval'])
                self.labels = mdlParams['labels_array'][inds_rep,:]
                # Path to images for loading, only for current indSet, repeat for multiordercrop
                self.im_paths = np.array(mdlParams['im_paths'])[inds_rep].tolist()
            # Set up transforms
            #self.norm = transforms.Normalize(torch.from_numpy(self.setMean).float(),torch.from_numpy(np.array([1.,1.,1.])).float())
            self.gray_scale_trans = transforms.Grayscale(num_output_channels=3)
            self.norm =  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.trans = transforms.ToTensor()
            self.resize =  transforms.Resize(mdlParams['input_size_load'][1])

            # Potentially preload
            if self.preload:
                self.im_list = []
                for i in range(len(self.im_paths)):
                    temp = Image.open(self.im_paths[i])
                    keep = temp.copy()
                    temp.close()
                    # Resize
                    if self.resize_image == True:
                        keep = self.resize(keep)
                    self.im_list.append(keep)
                    printProgressBar(i,len(self.im_paths), prefix = 'Pre-Load Eval Data:', suffix = 'Complete', length = 50)

        else: ####### Normal train proc #######

            
            if mdlParams.get('full_color_distort',False):
                color_distort = transforms.ColorJitter(brightness= 32. / 255,saturation=0.5,  contrast=0.5 , hue=0.5)
            else:
                color_distort = transforms.ColorJitter(brightness= 32. / 255,saturation=0.5,  contrast=0.5)

            self.resize =  transforms.Resize(mdlParams['input_size_load'][1])

            # All transforms
            if self.gray_scale == False:
                self.composed = transforms.Compose([                        
                        color_distort,
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        ])

           

            else:
                gray_scale = transforms.Grayscale(num_output_channels=3)
                self.composed = transforms.Compose([                        
                        gray_scale,
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        ])

            # Complete labels array, only for current indSet
            self.norm =  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.trans = transforms.ToTensor()
            self.labels = mdlParams['labels_array'][mdlParams[indSet],:]
            # Path to images for loading, only for current indSet
            self.im_paths = np.array(mdlParams['im_paths'])[mdlParams[indSet]].tolist()
            #print(mdlParams['im_paths'])

            # Patient based Batching
            if self.patient_based_batching:
                self.patient_paths = {}
                self.patient_labels = {}
                self.patient_images = {}

                self.TrainInd_ID_unique = mdlParams['TrainInd_ID_unique']

                num_patches = mdlParams['Train_numPatches_unique'] # array with patch numbers, e.g. [20,31, 7 , 8...]
                n_samples = num_patches.shape[0] # number of patients

                p_num_patch = 0
                c_num_patch = 0

                for p in range(n_samples): # loop over the different patients
                    c_num_patch = int(c_num_patch + num_patches[p])
                    patient_path = self.im_paths[p_num_patch:c_num_patch]
                    patient_label = self.labels[p_num_patch:c_num_patch]
                    p_num_patch = int(p_num_patch + num_patches[p])

                    self.patient_paths[mdlParams['TrainInd_ID_unique'][p]] = patient_path
                    self.patient_labels[mdlParams['TrainInd_ID_unique'][p]] = patient_label
                    #print(patient_label.
                    # Potentially preload for training
                    if self.preload:
                        patient_images = [] # all images for one patient
                        for path in patient_path:
                            temp = Image.open(path)
                            keep = temp.copy()
                            temp.close()
                            # Resize
                            if self.resize_image == True:
                                keep = self.resize(keep)
                            patient_images.append(keep)
                        self.patient_images[mdlParams['TrainInd_ID_unique'][p]] = patient_images

                    printProgressBar(p,n_samples, prefix = 'Pre-Load Train Data:', suffix = 'Complete', length = 50)

                self.labels = np.zeros([n_samples,mdlParams['numClasses']])

            # non-patient based batching and preloading
            else:
                # Potentially preload
                if self.preload:
                    self.im_list = []
                    for i in range(len(self.im_paths)):
                        temp = Image.open(self.im_paths[i])
                        keep = temp.copy()
                        temp.close()
                        # Resize
                        if self.resize_image == True:
                            keep = self.resize(keep)
                        self.im_list.append(keep)





    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx,def_train_idx = None):
        if not isinstance(idx, int):
            def_train_idx = idx[1]
            idx = idx[0]
        # Transform data based on whether train or not train. If train, also check if its train train or train inference
        if self.orderedCrop and (self.indSet == 'valInd' or self.indSet == 'testInd' or self.train_eval_state == 'eval'):

            # Load image
            if self.preload:
                x = self.im_list[idx]
            else:
                x = Image.open(self.im_paths[idx])
                # Resize Image
                if self.resize_image == True:
                    x = self.resize(x)
            # Get label
            y = self.labels[idx,:]

            if self.gray_scale == True:
                x = self.gray_scale_trans(x)

            # Apply ordered cropping to validation or test set
            # First, to pytorch tensor (0.0-1.0)
            # Get current crop position
            x = self.trans(x)

            x_loc = self.cropPositions[idx,0]
            y_loc = self.cropPositions[idx,1]
            # Then, apply current crop
            #print("Before",x.size(),"xloc",x_loc,"y_loc",y_loc)
            #print((x_loc-np.int32(self.input_size[0]/2.)),(x_loc-np.int32(self.input_size[0]/2.))+self.input_size[0],(y_loc-np.int32(self.input_size[1]/2.)),(y_loc-np.int32(self.input_size[1]/2.))+self.input_size[1])
            x = x[:,(x_loc-np.int32(self.input_size[0]/2.)):(x_loc-np.int32(self.input_size[0]/2.))+self.input_size[0],(y_loc-np.int32(self.input_size[1]/2.)):(y_loc-np.int32(self.input_size[1]/2.))+self.input_size[1]]

            #print("After",x.size())
            #print(x)

            # x -= x.min() # bring the lower range to 0
            # x /= x.max() # bring the upper range to 1
            x = self.norm(x)
            if self.multiple_mag == True:
                x = x.unsqueeze(0)

        elif self.indSet == 'valInd' or self.indSet == 'testInd':

            # Load image
            if self.preload:
                x = self.im_list[idx]
            else:
                x = Image.open(self.im_paths[idx])
                # Resize Image
                if self.resize_image == True:
                    x = self.resize(x)
            # Get label
            y = self.labels[idx,:]
            # Normalize
            x = self.cropping(x)

            if self.gray_scale == True:
                x = self.gray_scale_trans(x)

            # x = np.array(x)
            x = self.trans(x)
            # x -= x.min() # bring the lower range to 0
            # x /= x.max() # bring the upper range to 1
            # First, to pytorch tensor (0.0-1.0)

            x = self.norm(x)

        else:

            if self.patient_based_batching == False:
                # Load image
                if self.preload:
                    x = self.im_list[idx]
                else:
                    x = Image.open(self.im_paths[idx])
                    # Resize Image
                    if self.resize_image == True:
                        x = self.resize(x)
                
                # Get label
             
                y = self.labels[idx,:]
                print('X', self.im_paths[idx])
                print('Y', len(y))

            # training with patient based batchting
            else:
                if self.preload:
                    patient_images = self.patient_images[self.TrainInd_ID_unique[idx]] # all images for one patient
                    patient_labels = self.patient_labels[self.TrainInd_ID_unique[idx]] # all labels for one patient
                    # choose random image for patient
                    
                    train_idx = np.random.randint(0, int(patient_labels.shape[0]), 1) #get random index for training
                    
                    x = patient_images[int(train_idx)]
                    patient_labels = patient_labels[int(train_idx)]
                    y = patient_labels
                else:
                    #print('idx',idx)
                    patient_path = self.patient_paths[self.TrainInd_ID_unique[idx]] # all path for one patient
                    patient_labels = self.patient_labels[self.TrainInd_ID_unique[idx]] # all labels for one patient
                    # choose random image for patient
                    if def_train_idx == None:
                        train_idx = np.random.randint(0, int(patient_labels.shape[0]), 1) #get random index for training
                    
                    else:
                        train_idx = def_train_idx
                    #print('Number of patches',int(patient_labels.shape[0]))
                    #print('Patch Selected',train_idx)
                    #print('Train_idx', train_idx)
                    patient_path = patient_path[int(train_idx)]
                    patient_labels = patient_labels[int(train_idx)]
                    # open image for patient
                    x = Image.open(patient_path)
                    
                    # Resize Image
                    if self.resize_image == True:
                        x = self.resize(x)
                    y = patient_labels
                    #print('Before Y',y)

                temp_new = torch.empty(self.multiCropEval, 3,self.input_size[0], self.input_size[1]) ##harcoded
               
                
                #temp_new_img = x.ToTensor()
                x = self.composed(x)
                
                x = self.trans(x)
                #temp_new_img = x
                x = self.norm(x)
                #temp_new_img = x
                for i in range(self.multiCropEval):
                        #print(i)

                        x_loc = self.croplocation[i,0]
                        y_loc = self.croplocation[i,1]
                        #print(x_loc, y_loc)
                        x_new = x[:,(x_loc-np.int32(self.input_size[0]/2.)):(x_loc-np.int32(self.input_size[0]/2.))+self.input_size[0],(y_loc-np.int32(self.input_size[1]/2.)):(y_loc-np.int32(self.input_size[1]/2.))+self.input_size[1]]
                        
                        # Transform y

                        temp_new[i] = x_new
                        
                       
                
                x = temp_new
                

        #print(x.shape)
        # Transform y
        y = np.argmax(y)
        y = np.int64(y)
        
        #print('After y',y)
        if self.indSet == 'trainInd':            
            
            if self.multiple_mag == True:
                #print('HereinMag')
                return x, y, int(train_idx)
            else:
                return x, y, idx
        return x, y, idx



#Combiner for 3 datasets
class ConcatMag(Dataset):
    def __init__(self, data2000, data4000, data8000 ):
        self.data2000 = data2000        
        self.data4000 = data4000
        self.data8000 = data8000
    def __len__(self):
        return len(self.data2000)
    def __getitem__(self,idx):
        #print(idx)
        data2000_ = self.data2000[idx]
        train_idx = self.data2000[idx][2]
        data4000_ = self.data4000[idx,train_idx]
        data8000_ = self.data8000[idx,train_idx]
        #print('IDX', train_idx, data4000_[2],data8000_[2])
        x = torch.cat((data2000_[0],data4000_[0],data8000_[0]),dim = 0)
        if self.data2000[idx][1] == self.data4000[idx][1] and self.data2000[idx][1] == self.data8000[idx][1]:
            y = self.data2000[idx][1]
        return x, y, idx
    

# Sampler for balanced sampling
class StratifiedSampler(torch.utils.data.sampler.Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, mdlParams):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.dataset_len = len(mdlParams['trainInd'])
        self.numClasses = mdlParams['numClasses']
        self.trainInd = mdlParams['trainInd']
        # Sample classes equally for each batch
        # First, split set by classes
        not_one_hot = np.argmax(mdlParams['labels_array'][mdlParams['trainInd'],:],1)
        self.class_indices = []
        for i in range(mdlParams['numClasses']):
            self.class_indices.append(np.where(not_one_hot==i)[0])
        self.current_class_ind = 0
        self.current_in_class_ind = np.zeros([mdlParams['numClasses']],dtype=int)

    def gen_sample_array(self):
        # Shuffle all classes first
        for i in range(self.numClasses):
            np.random.shuffle(self.class_indices[i])
        # Construct indset
        indices = np.zeros([self.dataset_len])
        ind = 0
        while(ind < self.dataset_len):
            indices[ind] = self.class_indices[self.current_class_ind][self.current_in_class_ind[self.current_class_ind]]
            # Take care of in-class index
            if self.current_in_class_ind[self.current_class_ind] == len(self.class_indices[self.current_class_ind])-1:
                self.current_in_class_ind[self.current_class_ind] = 0
                # Shuffle
                np.random.shuffle(self.class_indices[self.current_class_ind])
            else:
                self.current_in_class_ind[self.current_class_ind] += 1
            # Take care of overall class ind
            if self.current_class_ind == self.numClasses-1:
                self.current_class_ind = 0
            else:
                self.current_class_ind += 1
            ind += 1
        return indices

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return self.dataset_len

def getErrClassification(step_outputs,mdlParams, indices):
    """Helper function to return the error of a set
    Args:
      mdlParams: dictionary, configuration file
      indices: string, either "trainInd", "valInd" or "testInd"
      modelVars: dictionary, contains model vars
    Returns:
      loss: float, avg loss
      acc: float, accuracy
      sensitivity: float, sensitivity
      spec: float, specificity
      conf: float matrix, confusion matrix
      f1: float, F1-score
      roc_auc: float array, roc values
      wacc: float, same as sensitivity
      predictions: float, predictions that were used for metric calculation
      targets: float,according targets
    """
    # Set up sizes
    if indices == 'trainInd':
        numBatches = int(math.floor(len(mdlParams[indices])/mdlParams['batchSize']/len(mdlParams['numGPUs'])))
    else:
        numBatches = int(math.ceil(len(mdlParams[indices])/mdlParams['batchSize']/len(mdlParams['numGPUs'])))
        print('Batches' ,numBatches)
    # Consider multi-crop case
    if 'multiCropEval' in mdlParams and mdlParams['multiCropEval'] > 0 and mdlParams.get('model_type_cnn') is None:
        loss_all = np.zeros([numBatches])
        allInds = np.zeros([len(mdlParams[indices])])
        predictions = np.zeros([len(mdlParams[indices]),mdlParams['numClasses']])
        targets = np.zeros([len(mdlParams[indices]),mdlParams['numClasses']])
        loss_mc = np.zeros([len(mdlParams[indices])])
        predictions_mc = np.zeros([len(mdlParams[indices]),mdlParams['numClasses'],mdlParams['multiCropEval']])
        targets_mc = np.zeros([len(mdlParams[indices]),mdlParams['numClasses'],mdlParams['multiCropEval']])

        for i, (loss,preds,tar) in enumerate(step_outputs):
          
            # Write into proper arrays
            loss_mc[i] = np.mean(loss.cpu().numpy())          

            predictions_mc[i,:,:] = np.transpose(preds.cpu().numpy())           
           
            targets_mc[i,:,:] = np.transpose(tar)
          
        # Targets stay the same
        loss_all = loss_mc
        targets = targets_mc[:,:,0]
        print(targets)
        if mdlParams['voting_scheme'] == 'vote':
            # Vote for correct prediction
            predictions_mc = np.argmax(predictions_mc,1)
            for j in range(predictions_mc.shape[0]):
                predictions[j,:] = np.bincount(predictions_mc[j,:],minlength=mdlParams['numClasses'])
        elif mdlParams['voting_scheme'] == 'average':
            predictions = np.mean(predictions_mc,2)
    else:
        for i, (inputs, labels, indices) in enumerate(modelVars['dataloader_'+indices]):
            # Get data
            inputs = inputs.to(modelVars['device'])
            labels = labels.to(modelVars['device'])
            # Not sure if thats necessary
            #modelVars['optimizer'].zero_grad()
            with torch.set_grad_enabled(False):
                # Get outputs
                outputs = model(inputs)
                #print("in",inputs.shape,"out",outputs.shape)
                preds = modelVars['softmax'](outputs)
                # Loss
                loss = criterion(outputs, labels)

            # Write into proper arrays
            if i==0:
                loss_all = np.array([loss.cpu().numpy()])
                predictions = preds
                tar_not_one_hot = labels.data.cpu().numpy()
                tar = np.zeros((tar_not_one_hot.shape[0], mdlParams['numClasses']))
                tar[np.arange(tar_not_one_hot.shape[0]),tar_not_one_hot] = 1
                targets = tar
                #print("Loss",loss_all)
            else:
                loss_all = np.concatenate((loss_all,np.array([loss.cpu().numpy()])),0)
                predictions = np.concatenate((predictions,preds),0)
                tar_not_one_hot = labels.data.cpu().numpy()
                tar = np.zeros((tar_not_one_hot.shape[0], mdlParams['numClasses']))
                tar[np.arange(tar_not_one_hot.shape[0]),tar_not_one_hot] = 1
                targets = np.concatenate((targets,tar),0)
                #allInds[(i*len(mdlParams['numGPUs'])+k)*bSize:(i*len(mdlParams['numGPUs'])+k+1)*bSize] = res_tuple[3][k]
        predictions_mc = predictions
    #print("Check Inds",np.setdiff1d(allInds,mdlParams[indices]))

    # Average predictions for patients
    if mdlParams['average_for_patients'] == True:

        # get the number of patches for each patient
        if indices == 'trainInd':
            num_patches = mdlParams['Train_numPatches_unique'] # array with patch numbers, e.g. [20,31, 7 , 8...]
            patient_ID = mdlParams['TrainInd_ID_unique'] # array with the patients IDs
            print('Train Eval')

        elif indices == 'testInd':
            num_patches = mdlParams['Test_numPatches_unique']
            patient_ID = mdlParams['TestInd_ID_unique']
            print('Test Eval')

        elif indices == 'valInd':
            num_patches = mdlParams['Val_numPatches_unique']
            patient_ID = mdlParams['ValInd_ID_unique']
            print('Val Eval')


        #print(num_patches)
        n_samples = num_patches.shape[0] # number of patients
        print('Number of Patients:', n_samples)
        print('Number of Patients:', patient_ID.shape[0])

        p_num_patch = 0
        c_num_patch = 0
        predictions_p_avg = []
        targets_p_avg = []

        for p in range(n_samples):
            c_num_patch = int(c_num_patch + num_patches[p])
            #print('c_num_patch',c_num_patch )
            #print('p_num_patch',p_num_patch )

            pred = np.mean(predictions[p_num_patch:c_num_patch] ,0)
            tar = np.mean(targets[p_num_patch:c_num_patch] ,0)
            #print(tar)

            if np.equal(np.argmax(pred,0),np.argmax(tar,0)) == 0:
                print('False Patient with ID', patient_ID[p])
                print('Num Patches:', num_patches[p])
                print('--------------------')

            p_num_patch = int(p_num_patch + num_patches[p])

            predictions_p_avg.append(pred)
            targets_p_avg.append(tar)

        predictions = np.array(predictions_p_avg)
        targets = np.array(targets_p_avg)

    # Calculate metrics
    # Accuarcy
    acc = np.mean(np.equal(np.argmax(predictions,1),np.argmax(targets,1)))
    # Confusion matrix
    conf = confusion_matrix(np.argmax(targets,1),np.argmax(predictions,1))
    if conf.shape[0] < mdlParams['numClasses']:
        conf = np.ones([mdlParams['numClasses'],mdlParams['numClasses']])
    # Class weighted accuracy
    wacc = conf.diagonal()/conf.sum(axis=1)
    # Sensitivity / Specificity
    sensitivity = np.zeros([mdlParams['numClasses']])
    specificity = np.zeros([mdlParams['numClasses']])
    for k in range(mdlParams['numClasses']):
            sensitivity[k] = conf[k,k]/(np.sum(conf[k,:]))
            true_negative = np.delete(conf,[k],0)
            true_negative = np.delete(true_negative,[k],1)
            true_negative = np.sum(true_negative)
            false_positive = np.delete(conf,[k],0)
            false_positive = np.sum(false_positive[:,k])
            specificity[k] = true_negative/(true_negative+false_positive)
            # F1 score
            f1 = f1_score(np.argmax(predictions,1),np.argmax(targets,1),average='weighted')
    # AUC
    fpr = {}
    tpr = {}
    # Calculate some per example metrics
    per_example_metrics = {}
    if 'valIndCV_association' in mdlParams:
        num_examples = len(np.unique(mdlParams['valInd_association']))
        per_example_metrics['Acc'] = np.zeros([num_examples])
        per_example_metrics['WAcc'] = np.zeros([num_examples,mdlParams['numClasses']])
        per_example_metrics['F1'] = np.zeros([num_examples])
        per_example_metrics['Conf'] = [None]*num_examples
        for i in range(num_examples):
            example_indices = np.where(mdlParams['valInd_association'] == i)[0]
            #print("Example indices",example_indices.shape)
            per_example_metrics['Acc'][i] = np.mean(np.equal(np.argmax(predictions[example_indices,:],1),np.argmax(targets[example_indices,:],1)))
            per_example_metrics['F1'][i] = f1_score(np.argmax(predictions[example_indices,:],1),np.argmax(targets[example_indices,:],1),average='weighted')
            per_example_metrics['Conf'][i] = confusion_matrix(np.argmax(targets[example_indices,:],1),np.argmax(predictions[example_indices,:],1))
            per_example_metrics['WAcc'][i,:] = per_example_metrics['Conf'][i].diagonal()/per_example_metrics['Conf'][i].sum(axis=1)
    roc_auc = np.zeros([mdlParams['numClasses']])
    for i in range(mdlParams['numClasses']):
        fpr[i], tpr[i], _ = roc_curve(targets[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        #print(roc_auc[i], roc_auc_score(targets[:, i], predictions[:, i]))

    return np.mean(loss_all), acc, sensitivity, specificity, conf, f1, roc_auc, wacc, predictions, targets, per_example_metrics

def get_metrics(Targets,Predictions):
        acc = np.mean(np.equal(np.argmax(Predictions,1),np.argmax(Targets,1)))
        conf = confusion_matrix(np.argmax(Targets,1),np.argmax(Predictions,1))
        sensitivity = conf[1,1]/(np.sum(conf[1,:]))
        true_negative = np.delete(conf,[1],0)
        true_negative = np.delete(true_negative,[1],1)
        true_negative = np.sum(true_negative)
        false_positive = np.delete(conf,[1],0)
        false_positive = np.sum(false_positive[:,1])
        specificity = true_negative/(true_negative+false_positive)
        f1 = f1_score(np.argmax(Targets,1),np.argmax(Predictions,1),average='weighted')
        roc_auc = roc_auc_score(Targets[:, 1].astype(int),Predictions[:, 1])

        print('Performance Metrics')
        wacc = conf.diagonal()/conf.sum(axis=1)
        print("Weighted Accuracy",np.mean(wacc))
        print('F1-Score:', f1)
        print('Sensitivy:', sensitivity)
        print('Specifity:', specificity)
        print('AUC:', roc_auc_score(Targets[:, 1].astype(int),Predictions[:, 1]))

        return acc, sensitivity, specificity, conf, f1, roc_auc, wacc



