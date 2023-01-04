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



## Defining command line input parameters

mdlParams = {}
number_epochs = int(sys.argv[2])

# Can choose between: task0, task1, task2

if 'task' in sys.argv[3]:
    mdlParams['task_num'] = int(sys.argv[3][-1:])  

# Can make use of one GU or multiple based on availability

if len(sys.argv) > 1:
    # Set visible devices
    gpu_option = sys.argv[1]
    
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

average_for_patients = True
#util_name = '_Multimag_Effiatten_RunwithWatch_NoMulticrop_3sidebyside_ReducedLRAddScheduler_YesDropout_Yes	Color_'
util_name = 'Test'
wandb.init(project="_Digital_Phatology", name = str(sys.argv[1]) + '_epochs_'+ str(number_epochs) + '_avg_' +str(average_for_patients)+util_name)

input_file_size = 4000
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
else:
    # best Model
    allData = {}
    allData['f1Best'] = {}
    allData['sensBest'] = {}
    allData['specBest'] = {}
    allData['accBest'] = {}
    allData['waccBest'] = {}
    allData['aucBest'] = {}
    allData['convergeTime'] = {}
    allData['bestPred'] = {}
    allData['targets'] = {}

    # current Model
    allData['f1Last'] = {}
    allData['sensLast'] = {}
    allData['specLast'] = {}
    allData['accLast'] = {}
    allData['waccLast'] = {}
    allData['aucLast'] = {}
    allData['convergeTime_last'] = {}
    allData['bestPred_last'] = {}
    allData['targets_Last'] = {}

    allData['f1Last_Test']= {}
    allData['sensLast_Test']= {}
    allData['specLast_Test']= {}
    allData['accLast_Test']= {}
    allData['waccLast_Test']= {}
    allData['aucLast_Test']= {}
    allData['predictions_Last_Test']= {}
    allData['targets_Last_Test']= {}

    allData['f1Last_Test_Patch']= {}
    allData['sensLast_Test_Patch']= {}
    allData['specLast_Test_Patch']= {}
    allData['accLast_Test_Patch']= {}
    allData['waccLast_Test_Patch']= {}
    allData['aucLast_Test_Patch']= {}
    allData['predictions_Last_Test_Patch']= {}
    allData['targets_Last_Test_Patch']= {}


    allData['f1Best_Test']= {}
    allData['sensBest_Test']= {}
    allData['specBest_Test']= {}
    allData['accBest_Test']= {}
    allData['waccBest_Test']= {}
    allData['aucBest_Test']= {}
    allData['predictions_Best_Test']= {}
    allData['targets_Best_Test']= {}
    allData['Best_Loss_Test'] ={}

    allData['accuracy_tr'] = {}
    allData['loss_tr'] = {}

if mdlParams['input_size_load'][0] >=1000:
    mdlParams['preload'] = False
elif mdlParams['model_type'] != 'adad':
    mdlParams['preload'] = False
else:
    mdlParams['preload'] = False


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
wandb.config.update(mdlParams_wandb)



# Class for Ligthning Module

class Multimag_Classifier(LightningModule):  ## Change name to multimag
    '''
    model = Model that will be used from Training, Test and Validation
    criterion = Loss function 

    '''
    def __init__(self,model,criterion):
        super().__init__()       
             
        self.model = model

        #Define Criterion
        self.class_weights = 1.0/np.mean(mdlParams['Train_Label_unique'],axis=0)
        self.criterion = criterion   
     
       
        wandb.watch(self.model,log = 'all',log_freq = 10)

       
    # Training_step defines the train loop.
    def training_step(self, batch, batch_idx):        
        
        x, labels,idx = batch    
        
        outputs = self.model(x,'train')      
   
        loss = self.criterion(outputs, labels)
      
        step = self.current_epoch+1
        wandb.log({"loss training":loss.cpu().item()},step = step + cv*(mdlParams['training_steps']+50))


        return loss


    def validation_step(self,batch,batch_idx):
        #print('i',i)
        # Get data
        inputs, labels,idx = batch
        # Get outputs
        if mdlParams['multiple_mag'] == True:
            outputs = model(inputs.squeeze(0))#.unsqueeze(0)
        else:
            outputs = model(inputs.unsqueeze(0))#.unsqueeze(0)
        preds = modelVars['softmax'](outputs)
        

        # Loss               
        loss = self.criterion(outputs, labels[0].unsqueeze(0))

        tar_not_one_hot = labels.data.cpu().numpy()
        tar = np.zeros((tar_not_one_hot.shape[0], mdlParams['numClasses']))
        tar[np.arange(tar_not_one_hot.shape[0]),tar_not_one_hot] = 1
        

        
        return loss,preds,tar

        
        

    def validation_epoch_end(self,validation_step_outputs):
        
        loss, accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy, predictions, targets, per_example_metrics = utilsMIL.getErrClassification(validation_step_outputs,mdlParams, eval_set)

        eval_metric = -np.mean(waccuracy)
        
        self.log('eval_metric',eval_metric)

        step = self.current_epoch+1
        wandb.log({'loss val': loss},step = step + cv*(mdlParams['training_steps']))
        wandb.log({'accuracy val': accuracy},step = step + cv*(mdlParams['training_steps']))
        wandb.log({'waccuracy val': eval_metric},step = step + cv*(mdlParams['training_steps']))
        wandb.log({'auc val': auc[1]},step = step + cv*(mdlParams['training_steps']) )
        wandb.log({'sensitivity val': sensitivity[1]},step = step + cv*(mdlParams['training_steps']) )
        wandb.log({'specificity val': specificity[1]},step = step + cv*(mdlParams['training_steps']) )
        wandb.log({'f1 val': f1},step = step + cv*(mdlParams['training_steps']) )


    

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=mdlParams['learning_rate'])
        scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        
        return [optimizer],[scheduler]


       
class Effinet(nn.Module):
    def __init__(self,model, batch_size=16, mul_crop=4, out_class=2, model_ip= [456,456,3], num_gpu=1):
        super(Effinet, self).__init__()
        
        self.D = 512
        self.K = 1
        self.p_drop = 0.2
        self.batchsize = batch_size
        self.multi_cropval = mul_crop
        self.out_class = out_class
        multimag = False
        if mdlParams['multiple_mag'] == True:
            self.num_mag = 3  ###ADD THIS TO mdlPARAMS
        else:
            self.num_mag = 1

        self.printdata = False
        self.network = model
        self.num_gpus = num_gpu
        self.model_ip = model_ip
        self.L = self.network._fc.in_features
        #print('L',self.L)
        self.per_partition= int(self.batchsize/self.num_gpus)
        self.network._fc = torch.nn.Identity() #nn.Linear(num_ftrs, mdlParams['numClasses']) ###Changed this
        #self.network._fc = nn.Linear(self.L*self.K, self.out_class)
        #self.network._swish  = torch.nn.Identity()

        #for param in self.network.parameters():
        #    param.requires_grad = False
        
        self.classifier = nn.Sequential(
            #nn.Dropout(self.p_drop),
            
            nn.Linear(self.L*self.K, self.out_class), ## Changed outclass to hidden
            #nn.Sigmoid()          
        )
        

    def forward(self, x,y='eval'):
        if self.printdata == True:
            print('X0 ', x.shape) #[20,4,3,244,244]

        x = x.reshape(self.per_partition*self.multi_cropval*self.num_mag, self.model_ip[2],self.model_ip[0],self.model_ip[1])

        if self.printdata == True:
            print('X1 ', x.shape) #[80,3,244,244]
       
        E = self.network(x) #[80,1280]
        if self.printdata:
            print('E2', E.shape)
       
        # [20,4,1280] 
        #   -> [4,1280] -> [1,1280] 
        E = E.reshape(self.per_partition,self.multi_cropval*self.num_mag,-1)
        if self.printdata:
            print('E3', E.shape)
        for i in range (self.per_partition):
            A = torch.ones(1, self.multi_cropval).to("cuda:0")
            
            E_ = torch.mm(A, E[i]) 
            if self.printdata:
                print('E4', E_.shape)

            C_ = self.classifier(E_)
            if i==0:
                op = C_
            else:
                op= torch.cat((op, C_))
 
        C = op
        #C = self.classifier(E) #[80,2]
        if self.printdata:
            print('C3', C.shape)  
        return C      


class Resnet_test(nn.Module):
    def __init__(self,model, batch_size=16, mul_crop=4, out_class=2, model_ip= [456,456,3], num_gpu=1):
        super(Resnet_test, self).__init__()
        
        self.D = 512
        self.K = 1
        self.p_drop = 0.2
        self.batchsize = batch_size
        self.multi_cropval = mul_crop
        self.out_class = out_class
        multimag = False
        if multimag == True:
            self.num_mag = 3  ###ADD THIS TO mdlPARAMS
        else:
            self.num_mag = 1
        self.printdata = False
        self.network = model
        self.num_gpus = num_gpu
        self.model_ip = model_ip
        self.L = self.network.fc.in_features
        #print('L',self.L)
        self.per_partition= int(self.batchsize/self.num_gpus)
        self.network.fc = torch.nn.Identity() #nn.Linear(num_ftrs, mdlParams['numClasses']) ###Changed this
        #self.network._fc = nn.Linear(self.L*self.K, self.out_class)
        

        #for param in self.network.parameters():
        #    param.requires_grad = False
        
        self.classifier = nn.Sequential(
            #nn.Dropout(self.p_drop),
            
            nn.Linear(self.L*self.K, self.out_class), ## Changed outclass to hidden
            #nn.Sigmoid()          
        )
        

    def forward(self, x):
        if self.printdata == True:
            print('X0 ', x.shape) #[20,4,3,244,244]


        x = x.reshape(self.per_partition*self.multi_cropval*self.num_mag, self.model_ip[2],self.model_ip[0],self.model_ip[1])

        if self.printdata == True:
            print('X1 ', x.shape) #[80,3,244,244]
       
        E = self.network(x) #[80,1280]
        if self.printdata:
            print('E2', E.shape)
       
        # [20,4,1280] 
        #   -> [4,1280] -> [1,1280] 
        E = E.reshape(self.per_partition,self.multi_cropval*self.num_mag,-1)
        if self.printdata:
            print('E3', E.shape)

        for i in range (self.per_partition):
            A = torch.ones(1, self.multi_cropval).to("cuda:0")
            
            E_ = torch.mm(A, E[i]) 
            if self.printdata:
                print('E4', E_.shape)

            C_ = self.classifier(E_)
            if i==0:
                op = C_
            else:
                op= torch.cat((op, C_))


 
        C = op
        #C = self.classifier(E) #[80,2]
        if self.printdata:
            print('C3', C.shape)  
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
        self.p_drop = 0.2
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

        self.printdata = True

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
            nn.ReLU()
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
            
            x = x.reshape(self.per_partition*self.multi_cropval*self.num_mag, self.model_ip[2],self.model_ip[0],self.model_ip[1])#squeeze(0)#
            


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
            A = A.reshape(-1,self.multi_cropval*self.num_mag)  ##[15, 4]
            if self.printdata == True:
                print('A7 ', A.shape)

        #Getting Ratios of the Attention for each image
        A = F.softmax(A, dim=1)  # softmax over N  ##[15, 4]
        if self.printdata == True:
            print('A8 ', A.shape)
       
        if train:
            H = H.reshape(self.per_partition,self.multi_cropval*self.num_mag,-1)   ##[60, 1000] -> [15,4, 1000]
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

def cv_set(mdlParams,mdlParams_):
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
for cv in range(mdlParams['numCV']):

    # Check if this fold was already trained
    if cv in allData['f1Best']:
        print('Fold ' + str(cv) + ' already trained.')
        continue
    # Reset model graph
    importlib.reload(models)
    # Collect model variables
    modelVars = {}
    modelVars['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache() 
    print('print the devices', modelVars['device'])
    print(("cuda:0" if torch.cuda.is_available() else "cpu"))
    # Def current CV set
    mdlParams = cv_set(mdlParams,mdlParams)
    

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
    save_dict = {}
    save_dict['acc'] = []
    save_dict['loss'] = []
    save_dict['wacc'] = []
    save_dict['auc'] = []
    save_dict['sens'] = []
    save_dict['spec'] = []
    save_dict['f1'] = []
    save_dict['step_num'] = []
    if mdlParams['print_trainerr']:
        save_dict_train = {}
        save_dict_train['acc'] = []
        save_dict_train['loss'] = []
        save_dict_train['wacc'] = []
        save_dict_train['auc'] = []
        save_dict_train['sens'] = []
        save_dict_train['spec'] = []
        save_dict_train['f1'] = []
        save_dict_train['step_num'] = []
    # Potentially calculate setMean to subtract
    if mdlParams['subtract_set_mean'] == 1:
        mdlParams['setMean'] = np.mean(mdlParams['images_means'][mdlParams['trainInd'],:],(0))
        print("Set Mean",mdlParams['setMean'])


  

    # Set up dataloaders
    # For a normal model
    # For train
    dataset_train = utilsMIL.Bockmayr_DataSet(mdlParams, 'trainInd') # loader for traningset (String indicates train or val set)
    print('Dataset Train label',dataset_train)
    # For val
    dataset_val = utilsMIL.Bockmayr_DataSet(mdlParams, 'valInd') # loader for val set
    
    #print('Dataset Val',len(dataset_val))

    num_workers = psutil.cpu_count(logical=False)

    modelVars['dataloader_valInd'] = DataLoader(dataset_val, batch_size=mdlParams['multiCropEval'], shuffle=False, num_workers=num_workers, pin_memory=False)
    #print('Val Labels',modelVars['dataloader_valInd'][0][1])
    # For test
    dataset_test = utilsMIL.Bockmayr_DataSet(mdlParams, 'testInd') # loader for val set
    modelVars['dataloader_testInd'] = DataLoader(dataset_test, batch_size=mdlParams['multiCropEval'], shuffle=False, num_workers=num_workers, pin_memory=False)
    print('**** Batch Size ;;:',mdlParams['batchSize'] , mdlParams['numGPUs'])
    if mdlParams['balance_classes'] == 2:
        #print(np.argmax(mdlParams['labels_array'][mdlParams['trainInd'],:],1).size(0))
        strat_sampler = utilsMIL.StratifiedSampler(mdlParams)
        modelVars['dataloader_trainInd'] = DataLoader(dataset_train, batch_size=mdlParams['batchSize'], sampler=strat_sampler, num_workers=num_workers, pin_memory=True, drop_last=True)
    else:
        modelVars['dataloader_trainInd'] = DataLoader(dataset_train, batch_size=mdlParams['batchSize'], shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)


    #For Magnified concatenated Data
    multimag = False
    if mdlParams['multiple_mag'] == True:
        print('-----Combining magnified data------')
        mdlParams_ = mdlParams
        mdlParams_['data_dir'] = '/home/Mukherjee/MBlst/new_data_sets/2000'
        mdlParams_data = data_cfg.init(mdlParams_)
        mdlParams_.update(mdlParams_data)        
        mdlParams_2000 = cv_set(mdlParams,mdlParams_)
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
    
        mdlParams_ = mdlParams
        mdlParams_['data_dir'] = '/home/Mukherjee/MBlst/new_data_sets/4000'
        mdlParams_data = data_cfg.init(mdlParams_)
        mdlParams_.update(mdlParams_data)        
        mdlParams_4000 = cv_set(mdlParams,mdlParams_)
        #mdlParams['trainInd'] = mdlParams['trainIndCV']
        print('trainind',mdlParams_4000['trainInd'][0])
        data4000_train = utilsMIL.Bockmayr_DataSet(mdlParams_4000,'trainInd')
        data4000_val = utilsMIL.Bockmayr_DataSet(mdlParams_4000, 'valInd')
        data4000_test = utilsMIL.Bockmayr_DataSet(mdlParams_4000, 'testInd')
        print('Train 4000,', data4000_train[0][0].shape)
        print('Val 4000,', data4000_val[0][0].shape)
        print('Train 4000 ID', data4000_train[0][2])  
        save_image(data4000_train[0][0], '/home/Mukherjee/ProjectFiles/MasterProject/source_code/Images/img2.png')


        mdlParams_ = mdlParams
        mdlParams_['data_dir'] = '/home/Mukherjee/MBlst/new_data_sets/8000'
        mdlParams_data = data_cfg.init(mdlParams_)
        mdlParams_.update(mdlParams_data)        
        mdlParams_8000 = cv_set(mdlParams,mdlParams_)
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
          
        train_img = dataset_train[0][0]
        save_image(train_img, '/home/Mukherjee/ProjectFiles/MasterProject/source_code/Images/imgcomb.png')
        
        modelVars['dataloader_valInd'] = DataLoader(dataset_val, batch_size=mdlParams['multiCropEval'], shuffle=False, num_workers=num_workers, pin_memory=False)
        modelVars['dataloader_testInd'] = DataLoader(dataset_test, batch_size=mdlParams['multiCropEval'], shuffle=False, num_workers=num_workers, pin_memory=False)

        if mdlParams['balance_classes'] == 2:
            #print(np.argmax(mdlParams['labels_array'][mdlParams['trainInd'],:],1).size(0))
            strat_sampler = utilsMIL.StratifiedSampler(mdlParams)
            modelVars['dataloader_trainInd'] = DataLoader(dataset_train, batch_size=mdlParams['batchSize'], sampler=strat_sampler, num_workers=num_workers, pin_memory=True, drop_last=True)
        else:
            modelVars['dataloader_trainInd'] = DataLoader(dataset_train, batch_size=mdlParams['batchSize'], shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
            

    # Define softmax
    modelVars['softmax'] = nn.Softmax(dim=1)
  

     
    #Defining Model
    print('Model Type', mdlParams['model_type'])
    if mdlParams['model_type'] == 'efficientnet_b0':
        model1 = models.getModel(mdlParams['model_type'])()          
        model = Eff_attention(model1, mdlParams['batchSize'], mdlParams['multiCropEval'], mdlParams['numClasses'], mdlParams['input_size'], len(mdlParams['numGPUs']))
    elif mdlParams['model_type'] == 'efficient_notAtten':
        model1 = models.getModel(mdlParams['model_type'])()  
        model = Effinet(model1, mdlParams['batchSize'], mdlParams['multiCropEval'], mdlParams['numClasses'], mdlParams['input_size'], len(mdlParams['numGPUs']))

    #print('Model', model)
    

 
    #Define Criterion
    class_weights = 1.0/np.mean(mdlParams['Train_Label_unique'],axis=0) 

    if mdlParams['balance_classes'] == 3 or mdlParams['balance_classes'] == 2 or mdlParams['balance_classes'] == 0:
        modelVars['criterion'] = nn.CrossEntropyLoss()
    elif mdlParams['balance_classes'] == 4:
        modelVars['criterion'] = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor(class_weights.astype(np.float32)),reduce=False)
    else:
        modelVars['criterion'] = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor(class_weights.astype(np.float32)))
 
    #Define Optimizer   
    #modelVars['optimizer'] = optim.Adam(modelVars['model'].parameters(), lr=mdlParams['learning_rate'])

    
    classify_model = Multimag_Classifier(model,modelVars['criterion'])

    # Checkpoint
 
    checkpoint_callback_all = ModelCheckpoint(auto_insert_metric_name = True, dirpath = mdlParams['saveDir'], filename='checkpoint-{epoch:02d}')
    checpoint_callback_best = ModelCheckpoint(monitor="eval_metric", auto_insert_metric_name = True, dirpath = mdlParams['saveDir'], filename='checkpoint_best-{epoch:02d}')
    #checkpoint_callback_ensemble = ModelCheckpoint(monitor="ensemble_count", auto_insert_metric_name = True, dirpath = mdlParams['saveDir'], filename='checkpoint-ensemble{epoch:02d}')
    checkpoint_callback_last = ModelCheckpoint(save_last = True, auto_insert_metric_name = True, dirpath = mdlParams['saveDir'], filename='checkpoint-{epoch:02d}')
   
    #Removed Ensemble add it later
    trainer = Trainer(callbacks =[checkpoint_callback_all,checpoint_callback_best,checkpoint_callback_last],max_epochs=mdlParams['training_steps'],accelerator='gpu', devices=1,log_every_n_steps = 1,check_val_every_n_epoch=1,benchmark=True,num_sanity_val_steps=0) ##Migh improve speed with the addition fo benchmark
    
  

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
        #print('Layer 0 weights', modelVars['model'].classifier[1].weight)

        

    print('Next Fold......')

    modelVars.clear()
 
    