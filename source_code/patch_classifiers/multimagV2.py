import torch
from IPython.core.display import display
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.nn import functional as F

from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models as tv_models
from torch.utils.data import DataLoader
from torchsummary import summary
import numpy as np
from scipy import io
import threading
import pickle
from pathlib import Path
import math
import os
import sys
from glob import glob
import re
import gc
import importlib
import time
import sklearn.preprocessing
from sklearn.utils import class_weight
import models
import utils_embb_edit as utilsMIL
import psutil
import wandb
#import pytorch_warmup as warmup
from copy import deepcopy
from sklearn.metrics import confusion_matrix, auc, roc_curve, f1_score, roc_auc_score
import timm
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.utils import save_image
from efficientnet_pytorch import EfficientNet
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torchmetrics



## Defining command line input parameters

mdlParams = {}
modelVars = {}
number_epochs = 50 #int(sys.argv[2])

# Can choose between: task0, task1, task2

#if 'task' in sys.argv[3]:
mdlParams['task_num'] = 0  #int(sys.argv[3][-1:])  

# Can make use of one GU or multiple based on availability


# Set visible devices
gpu_option = 'gpu0' #sys.argv[1]

print('Checking for GPU')
if 'gpu' in gpu_option:
    temp_len = len(gpu_option)
    gpu_option = gpu_option[-(temp_len-3):]
    gpu_option = [int(char) for char in gpu_option]
    gpu_option = list( dict.fromkeys(gpu_option) )

    cuda_str = ""
    for i in range(len(gpu_option)):
        cuda_str = cuda_str + str(gpu_option[i])
        if i is not len(gpu_option)-1:
            cuda_str = cuda_str + ","

    #print("Devices to use:",cuda_str)
    print(torch.cuda.is_available())
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_str
    mdlParams['numGPUs'] = gpu_option

    print(' USING GPUs', mdlParams['numGPUs'])
 





# Define and Initiate WandB
average_for_patients = False
#wandb.init()
#util_name = 'AllDatas(Proper)_thresh0.25_Mag_ATTN_EffiNet_Frozen_Relu_NoMulticrop_ReducedLRNoScheduler_YesDropout_Yescolor_NoEarlyStopping_L2Regularization_YesCV10_NoSweep_2000_v1'
util_name = 'Test'
#wandb.init(project="_Digital_Phatology", name = str(sys.argv[1]) + '_epochs_'+ str(number_epochs)+util_name)
wandb_logger = WandbLogger(project="_Digital_Phatology_Magnif", name = str(mdlParams['numGPUs'] ) + '_epochs_'+ str(number_epochs) +util_name)

input_file_size = 2000
print('Start Train Script')


# Import machine config
pc_cfg = importlib.import_module('pc_cfgs.'+'path_configs')
mdlParams.update(pc_cfg.mdlParams)
mdlParams['input_file_dimension'] = input_file_size
# Import model config
#print('s2a', mdlParams['model_type'])
model_cfg = importlib.import_module('cfgs.'+'model_config')
mdlParams_model = model_cfg.init(mdlParams)
mdlParams.update(mdlParams_model)

# Import Data parameters
data_cfg = importlib.import_module('cfgs.'+ '10CV')
mdlParams_data = data_cfg.init(mdlParams)
mdlParams.update(mdlParams_data)

mdlParams['saveDirBase'] = mdlParams['saveDir'] +util_name


 


# Dicts for paths
mdlParams['im_list'] = {}
mdlParams['tar_list'] = {}

# Check if there is a validation set, if not, evaluate train error instead
if 'valIndCV' in mdlParams or 'valInd' in mdlParams:
    eval_set = 'valInd'
    print("Evaluating on validation set during training.")
else:
    eval_set = 'trainInd'
    print("No validation set, evaluating on training set during training.")

# Check if there were previous ones that have already been learned
prevFile = Path(mdlParams['saveDirBase'] + '/CV.pkl')
#print(prevFile)
if prevFile.exists():
    print("Part of CV already done")
    with open(mdlParams['saveDirBase'] + '/CV.pkl', 'rb') as f:
        allData = pickle.load(f)



mdlParams['training_steps'] = number_epochs
mdlParams['average_for_patients'] = average_for_patients
#mdlParams['display_step'] = 300

# Prepare stripped cfg


mdlParams_wandb = deepcopy(mdlParams)
mdlParams_wandb.pop('labels_array',None)
mdlParams_wandb.pop('im_paths',None)
mdlParams_wandb.pop('trainIndCV',None)
mdlParams_wandb.pop('valIndCV',None)

mdlParams_wandb.pop('Val_numPatches_uniqueCV',None)
mdlParams_wandb.pop('Test_numPatches_uniqueCV',None)
mdlParams_wandb.pop('Train_numPatches_uniqueCV',None)

mdlParams_wandb.pop('Val_Label_uniqueCV',None)
mdlParams_wandb.pop('Test_Label_uniqueCV',None)
mdlParams_wandb.pop('Train_Label_uniqueCV',None)

mdlParams_wandb.pop('ValInd_ID_uniqueCV',None)
mdlParams_wandb.pop('TestInd_ID_uniqueCV',None)
mdlParams_wandb.pop('TrainInd_ID_uniqueCV',None)
wandb.config.update(mdlParams_wandb,allow_val_change = True)




# Class for Ligthning Module

class Multimag_Classifier(LightningModule):  ## Change name to multimag
    '''
    model = Model that will be used from Training, Test and Validation
    criterion = Loss function 

    '''
    def __init__(self,model,criterion,cv):
        super().__init__()       
             
        self.model = model

        #Define Criterion
        #self.class_weights = 1.0/np.mean(mdlParams['Train_Label_unique'],axis=0)
        #print(mdlParams['Train_Label_unique'])
        self.criterion = criterion   
        self.cv = cv
        
        #Freeze layers
        #self.model.network_1.freeze()
        #for param in self.model.network_1.parameters():
        #    param.requires_grad = False
        #for param in self.model.network_2.parameters():
        
        #    param.requires_grad = False
        for name, para in model.named_parameters():
            print("-"*20)
            print(f"name: {name}")
            print("values: ")
            print(para)
        sys.exit()
        #Metrics
        self.train_acc = torchmetrics.classification.BinaryAccuracy(threshold = 0.25)
        self.val_acc = torchmetrics.classification.BinaryAccuracy(threshold = 0.25)
        self.test_acc = torchmetrics.classification.BinaryAccuracy(threshold = 0.25)
        self.aucroc = torchmetrics.classification.BinaryAUROC(threshold = 0.25)
        self.confmat = torchmetrics.classification.BinaryConfusionMatrix(threshold = 0.25)
        self.f1 = torchmetrics.classification.BinaryF1Score(threshold = 0.25)
        self.Tprecision = torchmetrics.classification.BinaryPrecision(threshold = 0.25)
        self.recall = torchmetrics.classification.BinaryRecall(threshold = 0.25)
        self.specificity = torchmetrics.classification.BinarySpecificity(threshold = 0.25)
        self.prc = torchmetrics.classification.BinaryPrecisionRecallCurve(task="binary",thresholds = 5)
       
        wandb.watch(self.model,log = 'all',log_freq = 10)

       
    # Training_step defines the train loop.
    def training_step(self, batch, batch_idx):        
        
        x,x_thumb, labels,idx = batch    
        #print('X',x.shape)
        #print('Lablel',labels,labels.shape)
        #sys.exit()

        #if self.current_epoch+1 > int(20):
        #    #print('Paramters not trained')
            
        x = x.cpu()
        outputs = self.model(x,x_thumb,'train')      
   
        loss = self.criterion(outputs, labels)

        preds = modelVars['softmax'](outputs)
        tar = F.one_hot(labels, num_classes=2)
        #print('Tar',tar,tar.shape)

        train_acc = self.train_acc(preds,tar)
        #print(train_acc)
        
        
      
        step = self.current_epoch+1
        #wandb.log({"loss training":loss.cpu().item(),'train_epoch':step+self.cv*number_epochs})
        #wandb.log({"acc training":train_acc,'epoch':step+self.cv*number_epochs})
        self.log('train_loss',loss,on_epoch = True)
        self.log("train_loss_CV"+str(self.cv),loss,on_epoch = True)
        self.log("train_acc_CV"+str(self.cv),train_acc,on_epoch = True)
        #wandb.log

        return loss


    def validation_step(self,batch,batch_idx):
        #print('New')
        # Get data
        x,x_thumb, labels,idx = batch
        #print('Lablel Validation',labels,labels[0])
        # Get outputs
        if mdlParams['multiple_mag'] == True:
            outputs = self.model(x,x_thumb)#.unsqueeze(0)
        else:
            outputs = self.model(x,x_thumb) # due to change
            #outputs = self.model(inputs.unsqueeze(0))#.unsqueeze(0)
        #print(outputs)       

        # Loss 
        #print('Outputs',outputs)              
        loss = self.criterion(outputs, labels)#[0].unsqueeze(0))

        #Metrics
        preds = modelVars['softmax'](outputs)
        tar = F.one_hot(labels, num_classes=2)
        #print('Predictions and Target',preds,tar)
        val_acc = self.val_acc(preds,tar)
        aucroc = self.aucroc(preds,tar)
        #print(aucroc)
        confmat = self.confmat(preds,tar)
        #print('Confusion Matrix',confmat)
        f1 = self.f1(preds,tar)
        #print(f1)
        precision = self.Tprecision(preds,tar)
        recall = self.recall(preds,tar)
        precision_list, recall_list, thresholds_list = self.prc(preds,tar)
        
        step = self.current_epoch+1
        #wandb.log({'loss_val_CV'+str(self.cv): loss})
        #wandb.log({"acc val":val_acc,'epoch':step+self.cv*number_epochs})
        #wandb.log({'auc val': aucroc,'epoch':step+self.cv*number_epochs})
        self.log('val_loss_CV'+str(self.cv), loss,on_epoch = True)          
        self.log('val_auc_CV'+str(self.cv), aucroc,on_epoch = True)     
        self.log('val_f1_CV'+str(self.cv), f1,on_epoch = True)
        self.log("val_acc_CV"+str(self.cv),val_acc,on_epoch = True)
        #wandb.log({'val_PRC': wandb.plots.precision_recall(tar.cpu().argmax(), preds.cpu().argmax())})

        #self.log("val_precision_CV"+str(self.cv),precision,on_epoch = True)
        #self.log("val_recall_CV"+str(self.cv),recall,on_epoch = True)
        #self.log("val_threshold_CV"+str(self.cv),thresholds_list,on_epoch = True)
        for idx,thresh in enumerate(thresholds_list):
            self.log('Thresholds',thresh,on_epoch = True)
            self.log("val_precision_CV"+str(self.cv)+"_threshold"+str(thresh),precision_list[idx],on_epoch = True)
            self.log("val_recall_CV"+str(self.cv)+"_threshold"+str(thresh),recall_list[idx],on_epoch = True)


        #self.log('loss_val',loss)
        #print('Tar',tar)
        #print('Pred',preds)
        return {'val_loss':loss,'preds':preds,'targets':tar}
    

    def test_step(self,batch,batch_idx):
        # Get data
        x,x_thumb, labels,idx = batch
        #print('Lablel Validation',labels,labels[0])
        # Get outputs
        if mdlParams['multiple_mag'] == True:
            outputs = self.model(x,x_thumb)#.unsqueeze(0)
        else:
            outputs = self.model(x,x_thumb) # due to change
            #outputs = self.model(inputs.unsqueeze(0))#.unsqueeze(0)
        #print(outputs)       

        # Loss 
        #print('Outputs',outputs)              
        loss = self.criterion(outputs, labels)#[0].unsqueeze(0))

        #Metrics
        preds = modelVars['softmax'](outputs)
        tar = F.one_hot(labels, num_classes=2)
        #print('Predictions and Target',preds,tar)
        test_acc = self.test_acc(preds,tar)
        #print('Test_acc',test_acc)
        aucroc = self.aucroc(preds,tar)
        #print('Aucroc',aucroc)
        confmat = self.confmat(preds,tar)
        #print('Confusion Matrix',confmat)
        f1 = self.f1(preds,tar)
        #print('F1',f1)
        precision = self.Tprecision(preds,tar)
        recall = self.recall(preds,tar)
        specificity = self.specificity(preds,tar)
        precision_list, recall_list, thresholds_list = self.prc(preds,tar)
        
        step = self.current_epoch+1
        #wandb.log({'Cross Validation fold':self.cv+1})
        #wandb.log({'loss test': loss} )
        #wandb.log({"acc test":test_acc})
        #wandb.log({'auc test': aucroc})       
        #wandb.log({'f1 test': f1})
        self.log('Cross Validation fold',self.cv+1)
        self.log('test_loss_CV'+str(self.cv), loss)
        self.log('test_auc_CV'+str(self.cv), aucroc,on_epoch = True)     
        self.log('test_f1_CV'+str(self.cv), f1,on_epoch = True)
        self.log("test_acc_CV"+str(self.cv),test_acc,on_epoch = True)
        self.log("test_precision_CV"+str(self.cv),precision,on_epoch = True)
        self.log("test_recall_CV"+str(self.cv),recall,on_epoch = True)
        #self.log('test_confmat_CV'+str(self.cv),confmat,on_epoch = True)
        self.log("test_specificity_CV"+str(self.cv),specificity,on_epoch = True)
        #self.log('test_PRC',wandb.plots.precision_recall(tar.cpu(), preds.cpu()),on_epoch = True)
        #self.log("test_threshold_CV"+str(self.cv),thresholds_list,on_epoch = True)
        for idx,thresh in enumerate(thresholds_list):
            self.log('Thresholds',thresh,on_epoch = True)
            self.log("test_precision_CV"+str(self.cv)+"_threshold"+str(thresh),precision_list[idx],on_epoch = True)
            self.log("test_recall_CV"+str(self.cv)+"_threshold"+str(thresh),recall_list[idx],on_epoch = True)

        #self.log('loss_val',loss)
        #print('Tar',tar)
        #print('Pred',preds)
        return {'test_loss':loss,'preds':preds,'targets':tar,'acc_val':test_acc,'aucroc':aucroc,'f1':f1,'confmat':confmat}
    

    

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=mdlParams['learning_rate'])#,weight_decay=1e-5)
        #scheduler = lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)
        
        return [optimizer]#,[scheduler]


       
class Effinet(nn.Module):
    '''
    model = The base model that will be used fro Transfer Learning
    batch_size = Size of the Training, Validation  and Test Batches
    mul_crop = The number of crops from the image used in training
    out_class = The number of output classes
    model_ip = The input size of the Transfer learning Model
    num_gpu = Number of GPUs
    
    '''
    def __init__(self,model, batch_size=16, mul_crop=4, out_class=2, model_ip= [456,456,3], num_gpu=1):
        super(Effinet, self).__init__()
        
        #Model Parameters
        self.D = 512
        self.K = 1
        self.p_drop = mdlParams['p_drop']
        self.batchsize = batch_size
        self.multi_cropval = mul_crop
        self.out_class = out_class
        self.printdata = False   
        self.network = model
        self.num_gpus = num_gpu
        self.model_ip = model_ip
        self.per_partition= int(self.batchsize/self.num_gpus)
        # Flag to check for Multiple Magnification Evaluation  
        if mdlParams['multiple_mag'] == True:
            self.num_mag = 3  ###ADD THIS TO mdlPARAMS
        else:
            self.num_mag = 1        
        

        #Number of outputs from the feedforward layer
        self.L = self.network._fc.in_features
        self.network._fc = torch.nn.Identity() #nn.Linear(num_ftrs, mdlParams['numClasses']) ###Changed this
        self.network._swish  = torch.nn.Identity()

        
        
        self.classifier = nn.Sequential(
            nn.Dropout(self.p_drop),
            
            nn.Linear(self.L*self.K*self.multi_cropval, self.out_class), ## Changed outclass to hidden
            #nn.Sigmoid()
            #nn.ReLU()          
        )
        

    def forward(self, x,y='eval'):
        if self.printdata == True:
            print('Input')
            print('Type',y)
            print('X0 ', x.shape) #[20,3,244,244]
            print('Reshape', self.per_partition*self.multi_cropval, self.model_ip[2],self.model_ip[0],self.model_ip[1])
        if y == 'eval':
            x = x.squeeze(0)
        else:
            x = x.reshape(self.per_partition*self.multi_cropval, self.model_ip[2],self.model_ip[0],self.model_ip[1])

        if self.printdata == True:
            print('X1 ', x.shape) #[80,3,244,244]
       
        E = self.network(x) #[80,1280]
        if self.printdata:
            print('E2', E.shape)
       
        # [20,4,1280]
        if y == 'eval':
           E = E.reshape(1,-1) 
        else: 
           E = E.reshape(self.per_partition,-1)
        if self.printdata:
            print('E3', E.shape)
        

        C = self.classifier(E)
        return C    


class Multimag_effinet(nn.Module):
    '''
    model = The base model that will be used fro Transfer Learning
    batch_size = Size of the Training, Validation  and Test Batches
    mul_crop = The number of crops from the image used in training
    out_class = The number of output classes
    model_ip = The input size of the Transfer learning Model
    num_gpu = Number of GPUs
    
    '''
    def __init__(self,model, batch_size=16, mul_crop=4, out_class=2, model_ip= [456,456,3], num_gpu=1):
        super(Multimag_effinet, self).__init__()
        
        #Model Parameters
        self.D = 512
        self.K = 1
        self.p_drop = mdlParams['p_drop']
        self.batchsize = batch_size
        self.multi_cropval = mul_crop
        self.out_class = out_class
        self.printdata = True
        self.network = model
        self.num_gpus = num_gpu
        self.model_ip = model_ip
        self.per_partition= int(self.batchsize/self.num_gpus)
        # Flag to check for Multiple Magnification Evaluation  
        if mdlParams['multiple_mag'] == True:
            self.num_mag = 3  ###ADD THIS TO mdlPARAMS
        else:
            self.num_mag = 1        
        

        #Number of outputs from the feedforward layer
        self.L = self.network._fc.in_features
        self.network._fc = torch.nn.Identity() #nn.Linear(num_ftrs, mdlParams['numClasses']) ###Changed this
        self.network._swish  = torch.nn.Identity()

        
        
        self.classifier = nn.Sequential(
            nn.Dropout(self.p_drop),
            
            nn.Linear(self.L*self.K*self.multi_cropval*3, self.out_class), ## Changed outclass to hidden
            #nn.Sigmoid()
            #nn.ReLU()          
        )
        

    def forward(self, x,y='eval'):
        if self.printdata == True:
            print('Input',x.shape)
        #Getting the 3 Values seperated
        if y == 'eval':
            x_2000 = x[0:1*self.multi_cropval,:,:,:]   
            x_4000 = x[1*self.multi_cropval:2*self.multi_cropval,:,:,:] 
            x_8000 = x[2*self.multi_cropval:3*self.multi_cropval,:,:,:]
        else:
            x_2000 = x[:, 0:1*self.multi_cropval,:,:,:]   
            x_4000 = x[:, 1*self.multi_cropval:2*self.multi_cropval,:,:,:] 
            x_8000 = x[:, 2*self.multi_cropval:3*self.multi_cropval,:,:,:] 

        #x_2000 = torch.squeeze(x_2000)
        #x_4000 = torch.squeeze(x_4000)
        #x_8000 = torch.squeeze(x_8000)

        if self.printdata == True:
            print('Input')
            print('X0 ', x_2000.shape) 
            print('Reshape', self.per_partition*self.multi_cropval, self.model_ip[2],self.model_ip[0],self.model_ip[1])
        if y == 'eval':
            x_2000 = x_2000
            x_4000 = x_4000
            x_8000 = x_8000
        else:
            #sys.exit()
            x_2000 = x_2000.reshape(self.per_partition*self.multi_cropval, self.model_ip[2],self.model_ip[0],self.model_ip[1])
            x_4000 = x_4000.reshape(self.per_partition*self.multi_cropval, self.model_ip[2],self.model_ip[0],self.model_ip[1])
            x_8000 = x_8000.reshape(self.per_partition*self.multi_cropval, self.model_ip[2],self.model_ip[0],self.model_ip[1])

        if self.printdata == True:
            print('X1 ', x_2000.shape)
       
        E1 = self.network(x_2000) 
        E2 = self.network(x_4000)
        E3 = self.network(x_8000)
        if self.printdata:
            print('E2', E1.shape)


    
       
        # [20,4,1280]
        if y == 'eval':
           E1 = E1.reshape(1,-1)
           E2 = E2.reshape(1,-1)
           E3 = E3.reshape(1,-1) 
        else: 
           E1 = E1.reshape(self.per_partition,-1)
           E2 = E2.reshape(self.per_partition,-1)
           E3 = E3.reshape(self.per_partition,-1)
        if self.printdata:
            print('E3', E1.shape)
        
        #concat
        E = torch.cat((E1,E2,E3),1)
        if self.printdata:
            print('E4', E.shape)



        C = self.classifier(E)
        return C    

   
class Multimag_effi_Atten(nn.Module):
    '''
    model = The base model that will be used fro Transfer Learning
    batch_size = Size of the Training, Validation  and Test Batches
    mul_crop = The number of crops from the image used in training
    out_class = The number of output classes
    model_ip = The input size of the Transfer learning Model
    num_gpu = Number of GPUs
    
    '''
    
    def __init__(self,model1,model2, batch_size=16, mul_crop=4, out_class=2, model_ip= [456,456,3], num_gpu=1):
        super(Multimag_effi_Atten, self).__init__()
        self.D = 512
        self.K = 1
        self.p_drop = mdlParams['p_drop']
        self.batchsize = batch_size
        self.multi_cropval = mul_crop
        self.out_class = out_class
        self.printdata = True
        self.network_1 = model1
        self.network_2 = model2
        self.num_gpus = num_gpu
        self.model_ip = model_ip
        self.feature_model_ip = [28,28,3]
        self.per_partition= int(self.batchsize/self.num_gpus)
        self.percent_patches = 40
        self.P = int(self.multi_cropval*self.percent_patches/100)


        self.L_1 = self.network_1._fc.in_features
        

        self.network_1._fc = torch.nn.Identity() #nn.Linear(num_ftrs, mdlParams['numClasses']) ###Changed this
        self.network_1._swish  = torch.nn.Identity()

        self.L_2 = self.network_2._fc.in_features
        self.network_2._fc = torch.nn.Identity() #nn.Linear(num_ftrs, mdlParams['numClasses']) ###Changed this
        self.network_2._swish  = torch.nn.Identity()



        self.classifier = nn.Sequential(
            nn.Dropout(self.p_drop),
            
            nn.Linear(self.L_1*self.K, self.out_class), ## Changed outclass to hidden
            #nn.Sigmoid()
            #nn.ReLU()          
        )


        self.attention = nn.Sequential(

            nn.Linear(self.L_1, self.D),
            nn.Dropout(self.p_drop),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        

    def forward(self, x,x_thumb,y='eval'):
        #There are 2 inputs, the resized image and the full image
        
        if self.printdata:
            print('Shapes')
            print('X0',x.shape)
            print('X0_thumb',x_thumb.shape)
        #Reshaping

        if y == 'eval':
            train = False
            x_thumb = x_thumb.reshape(self.multi_cropval, self.model_ip[2],self.model_ip[0],self.model_ip[1])
            x = x.reshape(x.shape[0],x.shape[1],-1)
            if self.printdata == True:
                print('Eval')
                print('X0',x.shape)
                print('X0_thumb',x_thumb.shape)

        else:    
            train = True
            x_thumb = x_thumb.reshape(self.per_partition*self.multi_cropval, self.model_ip[2],self.model_ip[0],self.model_ip[1])        
            x = x.reshape(x.shape[0],x.shape[1],-1)
            if self.printdata:
                print('X1_thumb',x_thumb.shape)
                print('X1',x.shape)
            

        #First feature extractor
        E = self.network_1(x_thumb)
        if self.printdata:
            print('Shape E1',E.shape) ##10x1280

        #Attention
        A = self.attention(E)
        A = torch.transpose(A, 1, 0)
        A = A.reshape(-1,self.multi_cropval)
        A = F.softmax(A, dim=1)
        A_max,A_max_patches = torch.topk(A,self.P)
        if self.printdata:
            print('Shape A',A.shape)
            print('A_max',A_max)
            print('Indices',A_max_patches)

        #Attention patches    
        x_out = torch.empty(x.shape[0], A_max_patches.shape[1],x.shape[2]).cuda()
        if self.printdata:
            
            print('X',x_out.shape)
        for i in range(x.shape[0]):
            x_out[i] = x[i][A_max_patches[i]][:]
        #x = torch.gather(x,dim = 1,index = A_max_patches)
        if self.printdata:            
            print('X',x_out.shape)


        #Reshaping to (X, 3,244,244)
        if train:
            x = x_out.reshape(self.per_partition*self.P, self.model_ip[2],self.model_ip[0],self.model_ip[1])
        else:
            x = x_out.reshape(self.P, self.model_ip[2],self.model_ip[0],self.model_ip[1])

        if self.printdata:
            
            print('X',x.shape)


        #Second Feature extractor
        x = x.type_as(x_thumb)
        E2 = self.network_2(x)

        if self.printdata:
            
            print('E2',E2.shape)
            
        if train:
            E2 = E2.reshape(self.per_partition,self.P,-1)
        else:
            #E2 = E2.reshape(1,-1)
            
            M = torch.mm(A_max, E2)
            C = self.classifier(M)
            #print(Y_prob, Y_prob.shape)
            return C
        if self.printdata:
            
            print('E2',E2.shape)
        
        for i in range (self.per_partition):
            #print('In Loop',i, A[i].unsqueeze(0).shape,  H[i].shape)
            M = torch.mm(A_max[i].unsqueeze(0), E2[i])  # KxL   #[1, 4] x  [4, 1000]-> [1, 1000]
            if self.printdata == True:
               print('10 ', M.shape)
            C = self.classifier(M)
            if self.printdata == True:
                print('11 ', C, C.shape)
            if i==0:
                op = C
            else:
                op= torch.cat((op, C))

        C = op
        if self.printdata:
            
            print('C',C.shape)        

        return C





class Eff_attention(nn.Module):
    '''
    model = The base model that will be used fro Transfer Learning
    batch_size = Size of the Training, Validation  and Test Batches
    mul_crop = The number of crops from the image used in training
    out_class = The number of output classes
    model_ip = The input size of the Transfer learning Model
    num_gpu = Number of GPUs
    
    '''
    def __init__(self,model, batch_size=16, mul_crop=4, out_class=2, model_ip= [456,456,3], num_gpu=1):
        super(Eff_attention, self).__init__()

        
        #Model Parameters
        self.D = 512
        self.K = 1
        self.p_drop = mdlParams['p_drop']
        self.batchsize = batch_size
        self.multi_cropval = mul_crop
        self.out_class = out_class
        self.num_gpus = num_gpu
        self.model_ip = model_ip

        # Flag to check for Multiple Magnification Evaluation       
        if mdlParams['multiple_mag'] == True:
            self.num_mag = 3  
        else:
            self.num_mag = 1

        self.printdata = False

        self.per_partition= int(self.batchsize/self.num_gpus)

        self.network = model 
        #Number of outputs from the feedforward layer
        self.L = self.network._fc.in_features       
        
        self.network._fc = torch.nn.Identity()         
        self.network._swish  = torch.nn.Identity()

       
        self.attention = nn.Sequential(

            nn.Linear(self.L, self.D),
            nn.Dropout(self.p_drop),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(self.p_drop),            
            nn.Linear(self.L*self.K, self.out_class),
            #nn.Sigmoid()
            #nn.ReLU()
        )

    def forward(self, x,y='eval'):

        if self.printdata == True:
            print('X1 ', x.shape)

        
        #Check for Eval or train
        if y=='eval':
            
            train = False          
            if self.printdata == True:
                print('Eval')
                print('X2',x.shape)
        else:

            train = True
            if self.printdata == True:
                print('Train')
                print('X2',x.shape)
            
            x = x.reshape(self.per_partition*self.multi_cropval, self.model_ip[2],self.model_ip[0],self.model_ip[1])#squeeze(0)#
            


       #Get output from CNN
        H = self.network(x)   
        if self.printdata == True:
            print('H4 ', H.shape)   ##(120,1280)
     

        #Get output from Attention Net
        A = self.attention(H)  # NxK##[60, 1]
        if self.printdata == True:
            print('A5 ', A.shape)        
       
        A = torch.transpose(A, 1, 0)  # KxN##[1, 60]   ###Ask if transpose affects the pipeline 
        if self.printdata == True:
            print('A6 ', A.shape)
        
        if train:           
            A = A.reshape(-1,self.multi_cropval)  ##[15, 4]
            if self.printdata == True:
                print('A7 ', A.shape)

        #Getting Ratios of the Attention for each image
        A = F.softmax(A, dim=1)  # softmax over N  ##[15, 4]
        if self.printdata == True:
            print('A8 ', A.shape)
       
        if train:
            H = H.reshape(self.per_partition,self.multi_cropval,-1)   ##[60, 1000] -> [15,4, 1000]
            if self.printdata == True:
                print('H9', H.shape)
        if not train:
            M = torch.mm(A, H)
            Y_prob = self.classifier(M)
            #print(Y_prob, Y_prob.shape)
            return Y_prob

        for i in range (self.per_partition):
            #print('In Loop',i, A[i].unsqueeze(0).shape,  H[i].shape)
            M = torch.mm(A[i].unsqueeze(0), H[i])  # KxL   #[1, 4] x  [4, 1000]-> [1, 1000]
            if self.printdata == True:
               print('10 ', M.shape)
            Y_prob = self.classifier(M)
            if self.printdata == True:
                print('11 ', Y_prob, Y_prob.shape)
            if i==0:
                op = Y_prob
            else:
                op= torch.cat((op, Y_prob))
        
        return op



#torch.backends.cudnn.benchmark = True

def cv_set(mdlParams,mdlParams_,cv):
    mdlParams['trainInd'] = mdlParams_['trainIndCV'][cv] # get the length of the full data set
    #print(mdlParams['trainInd'])
    mdlParams['trainInd_eval'] = mdlParams_['trainInd'] # use full data set for training eval.

    if 'valIndCV' in mdlParams:
        mdlParams['valInd'] = mdlParams_['valIndCV'][cv] # get the length of the full validation set
        mdlParams['saveDir'] = mdlParams_['saveDirBase'] + '/CVSet' + str(cv) #restore save direction for fold
    else:
        mdlParams['saveDir'] = mdlParams_['saveDirBase']
    if 'valIndCV_association' in mdlParams:
        mdlParams['valInd_association'] = mdlParams_['valIndCV_association'][cv]
        mdlParams['valInd_association_name'] = mdlParams_['valIndCV_association_name'][cv]

    mdlParams['testInd'] = mdlParams_['testIndCV'][cv]

    mdlParams['Val_numPatches_unique'] = mdlParams_['Val_numPatches_uniqueCV'][cv]
    mdlParams['Test_numPatches_unique'] = mdlParams['Test_numPatches_uniqueCV'][cv]
    mdlParams['Train_numPatches_unique'] = mdlParams['Train_numPatches_uniqueCV'][cv]

    mdlParams['Val_Label_unique'] = mdlParams_['Val_Label_uniqueCV'][cv]
    mdlParams['Test_Label_unique'] = mdlParams_['Test_Label_uniqueCV'][cv]
    mdlParams['Train_Label_unique'] = mdlParams_['Train_Label_uniqueCV'][cv]

    mdlParams['ValInd_ID_unique'] = mdlParams_['ValInd_ID_uniqueCV'][cv]
    mdlParams['TestInd_ID_unique'] = mdlParams_['TestInd_ID_uniqueCV'][cv]
    mdlParams['TrainInd_ID_unique'] = mdlParams_['TrainInd_ID_uniqueCV'][cv]
    
    return mdlParams

    


   



# Take care of CV
def training(mdlParams):
    
    mdlParams_2000 = {}
    mdlParams_4000 = {}
    mdlParams_8000 = {}
    
    file = open ('/home/Mukherjee/ProjectFiles/MasterProject/source_code/patch_classifiers/time.txt', 'a') 
    for cv in range(mdlParams['numCV']):

       
        # Check if this fold was already trained

        # Reset model graph
        importlib.reload(models)
        # Collect model variables
        
        modelVars['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache() 
        print('print the devices', modelVars['device'])
        print(("cuda:0" if torch.cuda.is_available() else "cpu"))
        # Def current CV set
        mdlParams = cv_set(mdlParams,mdlParams,cv)

        #config time
        
        #print(wandb.config)
        mdlParams['batchSize'] = wandb.config.batchSize
        mdlParams['learning_rate'] = wandb.config.learning_rate
        mdlParams['p_drop'] = wandb.config.p_drop

        print('BS,LR,PD', mdlParams['batchSize'],mdlParams['learning_rate'],mdlParams['p_drop'])
       





        

        # Create basepath if it doesnt exist yet
        if not os.path.isdir(mdlParams['saveDirBase']):
            #print(mdlParams['saveDirBase'])
            os.mkdir(mdlParams['saveDirBase']) ## create folder for entire Model and all folds
        # Check if there is something to load
        load_old = 0
        if os.path.isdir(mdlParams['saveDir']): #check if there exists a folder for the CvSet
            # Check if a checkpoint is in there
            if len([name for name in os.listdir(mdlParams['saveDir'])]) > 0: #check if something is in the folder
                load_old = 1
                print("Loading old model") #an old model exists / nothing is loaded yet
            else:
                # Delete whatever is in there (nothing happens)
                filelist = [os.remove(mdlParams['saveDir'] +'/'+f) for f in os.listdir(mdlParams['saveDir'])]
        else:
            os.mkdir(mdlParams['saveDir'])
        print('Load old', load_old)
        # Save training progress in here
        


    

        # Set up dataloaders
        # For a normal model
        # For train
        dataset_train = utilsMIL.Bockmayr_DataSet(mdlParams, 'trainInd') # loader for traningset (String indicates train or val set)
        print('Dataset Train label',dataset_train)
        # For val
        dataset_val = utilsMIL.Bockmayr_DataSet(mdlParams, 'valInd') # loader for val set
        
        #print('Dataset Val',len(dataset_val))

        num_workers = psutil.cpu_count(logical=False)
    	#cnanged bachsize from mdlParams['multiCropEval'] to 1
        modelVars['dataloader_valInd'] = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=False)
        #print('Val Labels',modelVars['dataloader_valInd'][0][1])
        # For test
        dataset_test = utilsMIL.Bockmayr_DataSet(mdlParams, 'testInd') # loader for val set
        modelVars['dataloader_testInd'] = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=False)
        print('**** Batch Size ;;:',mdlParams['batchSize'] , mdlParams['numGPUs'])




        if mdlParams['balance_classes'] == 2:
            #print(np.argmax(mdlParams['labels_array'][mdlParams['trainInd'],:],1).size(0))
            strat_sampler = utilsMIL.StratifiedSampler(mdlParams)
            modelVars['dataloader_trainInd'] = DataLoader(dataset_train, batch_size=mdlParams['batchSize'], sampler=strat_sampler, num_workers=num_workers, pin_memory=True, drop_last=True)
        else:
            modelVars['dataloader_trainInd'] = DataLoader(dataset_train, batch_size=mdlParams['batchSize'], shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)


        #For Magnified concatenated Data
        #Need to put the output of CV outisde and only define it for the first CV loop and then make use of it for the rest?
        
        if mdlParams['multiple_mag'] == True:
            print('-----Combining magnified data------')
            if cv == 0:
                mdlParams_2000 = mdlParams
                mdlParams_2000['data_dir'] = '/home/Mukherjee/MBlst/new_data_sets/2000'
                mdlParams_data = data_cfg.init(mdlParams_2000)
                mdlParams_2000.update(mdlParams_data)        
            mdlParams_2000 = cv_set(mdlParams_2000,mdlParams_2000,cv)
            #mdlParams['trainInd'] = mdlParams['trainIndCV']
            print('trainind',mdlParams_2000['trainInd'][0])
            data2000_train = utilsMIL.Bockmayr_DataSet(mdlParams_2000,'trainInd')
            data2000_val = utilsMIL.Bockmayr_DataSet(mdlParams_2000, 'valInd')
            data2000_test = utilsMIL.Bockmayr_DataSet(mdlParams_2000, 'testInd')
            idx = data2000_train[0][2]
            print('idx',idx)
            print('Train 2000,', data2000_train[0][0].shape)
            print('Val 2000,', data2000_val[0][0].shape)
            print('Train 2000 ID', data2000_train[0][2]) 
            save_image(data2000_train[0][0], '/home/Mukherjee/ProjectFiles/MasterProject/source_code/Images/img1.png')

            if cv == 0:
                mdlParams_4000 = mdlParams
                mdlParams_4000['data_dir'] = '/home/Mukherjee/MBlst/new_data_sets/4000'
                mdlParams_data = data_cfg.init(mdlParams_4000)
                mdlParams_4000.update(mdlParams_data)        
            mdlParams_4000 = cv_set(mdlParams_4000,mdlParams_4000,cv)
            #mdlParams['trainInd'] = mdlParams['trainIndCV']
            print('trainind',mdlParams_4000['trainInd'][0])
            data4000_train = utilsMIL.Bockmayr_DataSet(mdlParams_4000,'trainInd')
            data4000_val = utilsMIL.Bockmayr_DataSet(mdlParams_4000, 'valInd')
            data4000_test = utilsMIL.Bockmayr_DataSet(mdlParams_4000, 'testInd')
            print('Train 4000,', data4000_train[0][0].shape)
            print('Val 4000,', data4000_val[0][0].shape)
            print('Train 4000 ID', data4000_train[0][2])  
            save_image(data4000_train[0][0], '/home/Mukherjee/ProjectFiles/MasterProject/source_code/Images/img2.png')

            if cv == 0:
                mdlParams_8000 = mdlParams
                mdlParams_8000['data_dir'] = '/home/Mukherjee/MBlst/new_data_sets/8000'
                mdlParams_data = data_cfg.init(mdlParams_8000)
                mdlParams_8000.update(mdlParams_data)        
            mdlParams_8000 = cv_set(mdlParams_8000,mdlParams_8000,cv)
            #mdlParams['trainInd'] = mdlParams['trainIndCV']
            print('trainind',mdlParams_8000['trainInd'][0])
            data8000_train = utilsMIL.Bockmayr_DataSet(mdlParams_8000,'trainInd')
            data8000_val = utilsMIL.Bockmayr_DataSet(mdlParams_8000, 'valInd')
            data8000_test = utilsMIL.Bockmayr_DataSet(mdlParams_8000, 'testInd')
            print('Train 8000,', data8000_train[0][0].shape) 
            print('Val 8000,', data8000_val[0][0].shape)
            print('Train 8000 ID', data8000_train[0][2])
            save_image(data8000_train[0][0], '/home/Mukherjee/ProjectFiles/MasterProject/source_code/Images/img3.png')

            #print('Length 2', len(data2000))
            #print('idx', data2000[2][1])

            #For Train, Val and Test datasets
            print('-------Concatanation--------')
            dataset_train = utilsMIL.ConcatMag(data2000_train,data4000_train,data8000_train)
            dataset_val = utilsMIL.ConcatMag(data2000_val,data4000_val,data8000_val)
            dataset_test = utilsMIL.ConcatMag(data2000_test ,data4000_test ,data8000_test)
            print('Train Mag,', dataset_train[0][0].shape,len(dataset_train))   
            print('Val Mag,', dataset_val[0][0].shape,len(dataset_val))  
            
            #train_img = dataset_train[0][0]
            #save_image(train_img, '/home/Mukherjee/ProjectFiles/MasterProject/source_code/Images/imgcomb.png')
            
            modelVars['dataloader_valInd'] = DataLoader(dataset_val, batch_size=mdlParams['multiCropEval'], shuffle=False, num_workers=num_workers, pin_memory=False)
            modelVars['dataloader_testInd'] = DataLoader(dataset_test, batch_size=mdlParams['multiCropEval'], shuffle=False, num_workers=num_workers, pin_memory=False)

            if mdlParams['balance_classes'] == 2:
                #print(np.argmax(mdlParams['labels_array'][mdlParams['trainInd'],:],1).size(0))
                strat_sampler = utilsMIL.StratifiedSampler(mdlParams)
                modelVars['dataloader_trainInd'] = DataLoader(dataset_train, batch_size=mdlParams['batchSize'], sampler=strat_sampler, num_workers=num_workers, pin_memory=True, drop_last=True)
            else:
                modelVars['dataloader_trainInd'] = DataLoader(dataset_train, batch_size=mdlParams['batchSize'], shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
            
            #for i in modelVars['dataloader_valInd']:
            #    print('Val shape',i[0].shape)
            #    sys.exit()  
            
        # Define softmax
        modelVars['softmax'] = nn.Softmax(dim=1)
    

        
        #Defining Model
        print('Model Type', mdlParams['model_type'])
        if mdlParams['model_type'] == 'efficientnet_b0':
            model1 = models.getModel(mdlParams['model_type'])()
            model2 = models.getModel(mdlParams['model_type'])()
            if mdlParams['multiple_mag'] == True:    
                model = Multimag_effinet(model1, mdlParams['batchSize'], mdlParams['multiCropEval'], mdlParams['numClasses'], mdlParams['input_size'], len(mdlParams['numGPUs']))     
            else:
                model = Multimag_effi_Atten(model1,model2, mdlParams['batchSize'], mdlParams['multiCropEval'], mdlParams['numClasses'], mdlParams['input_size'], len(mdlParams['numGPUs']))
        elif mdlParams['model_type'] == 'efficient_notAtten':
            model1 = models.getModel(mdlParams['model_type'])()  
            model = Effinet(model1, mdlParams['batchSize'], mdlParams['multiCropEval'], mdlParams['numClasses'], mdlParams['input_size'], len(mdlParams['numGPUs']))

        #print('Model', model)
        

    
        #Define Criterion
        #class_weights = 1.0/np.mean(mdlParams['Train_Label_unique'],axis=0) 
        print('Mean',np.mean(mdlParams['Train_Label_unique'],axis=0))
        class_weights = 1.0/np.mean(mdlParams['labels_array'][mdlParams['trainIndCV'][cv],:],axis=0)

        if mdlParams['balance_classes'] == 3 or mdlParams['balance_classes'] == 2 or mdlParams['balance_classes'] == 0:
            modelVars['criterion'] = nn.CrossEntropyLoss()
        elif mdlParams['balance_classes'] == 4:
            modelVars['criterion'] = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor(class_weights.astype(np.float32)),reduce=False)
        else:
            modelVars['criterion'] = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor(class_weights.astype(np.float32)))
    
        #Define Optimizer   
        #modelVars['optimizer'] = optim.Adam(modelVars['model'].parameters(), lr=mdlParams['learning_rate'])

        
        classify_model = Multimag_Classifier(model,modelVars['criterion'],cv)

        # Checkpoint
        #checkpoint_earlystopping = EarlyStopping(monitor = 'loss_val', patience = 10, mode = 'min')
        checkpoint_callback_all = ModelCheckpoint(auto_insert_metric_name = True, dirpath = mdlParams['saveDir'], filename='checkpoint-{epoch:02d}')
        checpoint_callback_best = ModelCheckpoint(monitor="val_acc_CV"+str(cv), mode = 'max',auto_insert_metric_name = True, dirpath = mdlParams['saveDir'], filename='checkpoint_best')#+trainer.logger.version)
        #checkpoint_callback_ensemble = ModelCheckpoint(monitor="ensemble_count", auto_insert_metric_name = True, dirpath = mdlParams['saveDir'], filename='checkpoint-ensemble{epoch:02d}')
        checkpoint_callback_last = ModelCheckpoint(save_last = True, auto_insert_metric_name = True, dirpath = mdlParams['saveDir'], filename='checkpoint-{epoch:02d}')
    
        
        


        #Removed Ensemble add it later
        trainer = Trainer(callbacks =[checkpoint_callback_all,checkpoint_callback_last,checpoint_callback_best],max_epochs=mdlParams['training_steps'],accelerator='gpu', devices=1,log_every_n_steps = 1,check_val_every_n_epoch=1,benchmark=True,num_sanity_val_steps=0,logger=wandb_logger) ##Migh improve speed with the addition fo benchmark
        #trainer.validate(model = classify_model, dataloaders = modelVars['dataloader_valInd'])
        #sys.exit()
    

        # loading from checkpoint
        load_old = 0
        if load_old:
            files = glob(mdlParams['saveDir']+'/*')
            global_steps = np.zeros([len(files)])
            for i in range(len(files)):
                # Use meta files to find the highest index
                if 'best' in files[i]:
                    continue
                if 'checkpoint-' not in files[i]:
                    continue
                # Extract global step
                nums = [int(s) for s in re.findall(r'\d+',files[i])]
                global_steps[i] = nums[-1]
            # Create path with maximum global step found
            steps = int(np.max(global_steps))
            chkPath = mdlParams['saveDir'] + '/checkpoint-epoch=' + f'{steps:02}'+'.ckpt'  #str(int(np.max(global_steps))) + '.pt'
            mdlParams['lastBestInd'] = int(np.max(global_steps))
            trainer.fit(model = classify_model, ckpt_path=chkPath,train_dataloaders = modelVars['dataloader_trainInd'], val_dataloaders = modelVars['dataloader_valInd'])

        else:
            start_epoch = 1
            mdlParams['lastBestInd'] = -1
            trainer.fit(model = classify_model,train_dataloaders = modelVars['dataloader_trainInd'], val_dataloaders = modelVars['dataloader_valInd'])
            #trainer.validate(model = classify_model,dataloaders = modelVars['dataloader_valInd'])
            #print('Test_path',mdlParams['saveDir']+'/checkpoint_best.ckpt')
            #op = trainer.test(dataloaders = modelVars['dataloader_testInd'],ckpt_path = mdlParams['saveDir']+'/checkpoint_best.ckpt')
            #print(op)
            #print('Layer 0 weights', modelVars['model'].classifier[1].weight)

            

        print('Next Fold......')

        modelVars.clear()
 
    
if __name__ == '__main__':
    sweep = False
    if sweep == True:
        print('--------Hparameter optimization---------')
        sweep_configuration = {
    'program':'source_code/patch_classifiers/multimagV1.py',        
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 
        'name': 'loss val'},
    'parameters': {'batchSize': {'values': [15,20,25]},
        'p_drop': {'values': [0.5, 0.6, 0.7, 0.8, 0.9]},
        'learning_rate': {'max': 0.001, 'min': 0.00001}}}
        wandb.init(config = sweep_configuration)
        
    
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="_Digital_Phatology")
        wandb.agent(sweep_id=sweep_id,count = 3)

    else:
        print('---------Training and Validation----------')
        training(mdlParams)
