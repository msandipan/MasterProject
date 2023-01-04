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
import utils_embb as utilsMIL
import psutil
import wandb
#import pytorch_warmup as warmup
from copy import deepcopy
from sklearn.metrics import confusion_matrix, auc, roc_curve, f1_score, roc_auc_score
import timm
from pytorch_lightning.callbacks import ModelCheckpoint

mdlParams = {}
number_epochs = int(sys.argv[2])
average_for_patients = True
util_name = '_Experiment_Name_s2new'
wandb.init(project="_Digital_Phatology", name = str(sys.argv[1]) + '_epochs_'+ str(number_epochs) + '_avg_' +str(average_for_patients)+util_name)

input_file_size = 4000
#print('The patch is of ', input_file_size)
print('Number of epochs to train ', number_epochs, sys.argv[1])
#assert 0
# Data Split config)

if 'task' in sys.argv[3]:
    mdlParams['task_num'] = int(sys.argv[3][-1:])
#torch.backends.cudnn.benchmark = True

# add configuration file
# Dictionary for model configuration
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

# Data Split config
data_cfg = importlib.import_module('cfgs.'+ '10CV')
mdlParams_data = data_cfg.init(mdlParams)
mdlParams.update(mdlParams_data)

print('After Split Config')
multimag = False
#mdlParams['trainInd'] = mdlParams['trainIndCV']	
if multimag == True:
    mdlParams['data_dir'] = '/home/Mukherjee/MBlst/new_data_sets/2000'
    
    mdlParams.update(mdlParams_data)
    mdlParams['trainInd'] = mdlParams['trainIndCV']
    #print(mdlParams['trainInd'])
    data2000 = utilsMIL.Bockmayr_DataSet(mdlParams,'trainInd')
    #print('Length 2', len(data2000))
    #print('idx', data2000[2][1])
    
    mdlParams['data_dir'] = '/home/Mukherjee/MBlst/new_data_sets/4000'
    mdlParams_data = data_cfg.init(mdlParams)
    mdlParams.update(mdlParams_data)
    mdlParams['trainInd'] = mdlParams['trainIndCV']
    #print(mdlParams['trainInd'])
    data4000 = utilsMIL.Bockmayr_DataSet(mdlParams,'trainInd')
    #print('Length 4', len(data4000[0][0]))
    #print('idx', data4000[2][1])


    mdlParams['data_dir'] = '/home/Mukherjee/MBlst/new_data_sets/8000'
    mdlParams_data = data_cfg.init(mdlParams)
    mdlParams.update(mdlParams_data)
    mdlParams['trainInd'] = mdlParams['trainIndCV']
    #print(mdlParams['trainInd'])
    data8000 = utilsMIL.Bockmayr_DataSet(mdlParams,'trainInd')
    #print('idx', data8000[0][0].shape)
    #print('Length 8', len(data8000))

#mdlParams['trainInd'] = mdlParams['trainIndCV'][0]
#datamag = utilsMIL.ConcatMag(data2000,data4000,data8000)
#print(datamag[0][0].shape)

#print(data[0][0].shape)


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
        #os.environ["CUDA_VISIBLE_DEVICES"] = cuda_str
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#mdlParams['numGPUs'] = [0]
# Path name from filename
mdlParams['saveDirBase'] = mdlParams['saveDir'] +util_name

#print(mdlParams['saveDir'])

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


class SelfSupervised_Classifier(LightningModule):  ## Chnage name to multimag
    '''
    backbone_model = Model that is used to trin the pretext task
    body_model = Model that is used to train the classifier

    '''
    def __init__(self):
        super().__init__()

                
        print('model_type', mdlParams['model_type'])
        #print('Batchsize', mdlParams['batchSize'])

    # training_step defines the train loop.
    def training_step(self, batch, batch_idx):
        #print('Batch Train',len(batch))

        x, labels,idx = batch        
        
        outputs = modelVars['model'](x, 'train')
        print('Layer 0 weights', modelVars['model'].classifier[1].weight)
        #print('Output',outputs.shape)
        #print('Labels',labels.shape)

        #outputs = self.body(features)
        loss = modelVars['criterion'](outputs, labels)
        if mdlParams['balance_classes'] == 4:
                    #loss = loss.cpu()
                indices = idx.cpu().numpy()
                loss = loss*torch.cuda.FloatTensor(mdlParams['extra_fac'][indices].astype(np.float32))
                loss = torch.mean(loss)
        #get scheduler

        step = self.current_epoch+1

        ensemble_count = 0
        mdlParams['horizontal_voting_checkpoints'] = [mdlParams['training_steps']-1-50, mdlParams['training_steps']-1-25, mdlParams['training_steps']-1]
        for epoch in mdlParams['horizontal_voting_checkpoints']:
            if step == epoch:
                ensemble_count +=1


        wandb.log({"loss training":loss.cpu().item()},step = step + cv*(mdlParams['training_steps']+50))
        self.log('ensemble_count',ensemble_count)


        return loss

    def validation_step(self, batch, batch_idx):
        #print('Batch Val',len(batch))
        #print('Batch Size', mdlParams['batchSize'])
        loss, accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy, predictions, targets, per_example_metrics = utilsMIL.getErrClassification(mdlParams, eval_set, modelVars)
        #loss_Validation = loss

        # SAVE STEP VALUE
        step = self.current_epoch+1

        # logging Data
        save_dict['loss'].append(loss)
        save_dict['acc'].append(accuracy)
        save_dict['wacc'].append(waccuracy)
        save_dict['auc'].append(auc)
        save_dict['sens'].append(sensitivity)
        save_dict['spec'].append(specificity)
        save_dict['f1'].append(f1)
        save_dict['step_num'].append(step)
        if os.path.isfile(mdlParams['saveDir'] + '/progression_'+eval_set+'.mat'):
            os.remove(mdlParams['saveDir'] + '/progression_'+eval_set+'.mat')
        io.savemat(mdlParams['saveDir'] + '/progression_'+eval_set+'.mat',save_dict)
        eval_metric = -np.mean(waccuracy)
        #IF WANDB DOESNT WORK THEN WE NEED self.log
        #Attempt to make this logging seperate from wandb logger
        self.log('eval_metric',eval_metric)

        wandb.log({'loss val': loss},step = step + cv*(mdlParams['training_steps']))
        wandb.log({'accuracy val': accuracy},step = step + cv*(mdlParams['training_steps']))
        wandb.log({'waccuracy val': waccuracy},step = step + cv*(mdlParams['training_steps']))
        wandb.log({'auc val': auc[1]},step = step + cv*(mdlParams['training_steps']) )
        wandb.log({'sensitivity val': sensitivity[1]},step = step + cv*(mdlParams['training_steps']) )
        wandb.log({'specificity val': specificity[1]},step = step + cv*(mdlParams['training_steps']) )
        wandb.log({'f1 val': f1},step = step + cv*(mdlParams['training_steps']) )


        # Check if we have a new best value
        if eval_metric <= mdlParams['valBest']:
            mdlParams['valBest'] = eval_metric
            allData['f1Best'][cv] = f1
            allData['sensBest'][cv] = sensitivity
            allData['specBest'][cv] = specificity
            allData['accBest'][cv] = accuracy
            allData['waccBest'][cv] = waccuracy
            allData['aucBest'][cv] = auc
            oldBestInd = mdlParams['lastBestInd']
            mdlParams['lastBestInd'] = step
            allData['convergeTime'][cv] = step #tranings_steps = number of epochs, step = epoch
            # Save best predictions
            allData['bestPred'][cv] = predictions
            allData['targets'][cv] = targets

            wandb.log({'b loss val': loss},step = step + cv*(mdlParams['training_steps']) )
            wandb.log({'b accuracy val': accuracy},step = step + cv*(mdlParams['training_steps']) )
            wandb.log({'b waccuracy val': np.mean(waccuracy)},step = step + cv*(mdlParams['training_steps']) )
            wandb.log({'b auc val': np.mean(auc)},step = step + cv*(mdlParams['training_steps']) )
            wandb.log({'b sensitivity val': sensitivity[1]},step = step + cv*(mdlParams['training_steps']) )
            wandb.log({'b specificity val': specificity[1]},step = step + cv*(mdlParams['training_steps']) )
            wandb.log({'b f1 val': f1},step = step + cv*(mdlParams['training_steps']) )



            if mdlParams['peak_at_testerr']:
                loss_t, accuracy_t, sensitivity_t, specificity_t, conf_matrix_t, f1_t, auc_t, waccuracy_t, predictions_t, targets_t, per_example_metrics_t = utilsMIL.getErrClassification(mdlParams, 'testInd', modelVars)
                allData['f1Best_Test'][cv] = f1_t
                allData['sensBest_Test'][cv] = sensitivity_t
                allData['specBest_Test'][cv] = specificity_t
                allData['accBest_Test'][cv] = accuracy_t
                allData['waccBest_Test'][cv] = waccuracy_t
                allData['aucBest_Test'][cv] = auc_t
                allData['predictions_Best_Test'][cv] = predictions_t
                allData['targets_Best_Test'][cv] = targets_t
                allData['Best_Loss_Test'][cv] = loss_t

                wandb.log({'b loss test': loss_t},step = step + cv*(mdlParams['training_steps']))
                wandb.log({'b accuracy test': accuracy_t},step = step + cv*(mdlParams['training_steps']))
                wandb.log({'b waccuracy test':  np.mean(waccuracy_t)},step = step + cv*(mdlParams['training_steps']))
                wandb.log({'b auc test': np.mean(auc_t)},step = step + cv*(mdlParams['training_steps']))
                wandb.log({'b sensitivity test': sensitivity_t[1]},step = step + cv*(mdlParams['training_steps']))
                wandb.log({'b specificity test': specificity_t[1]},step = step + cv*(mdlParams['training_steps']) )
                wandb.log({'b f1 test': f1_t},step = step + cv*(mdlParams['training_steps']) )

            # save current Model and Metrics

            allData['f1Last'][cv] = f1
            allData['sensLast'][cv] = sensitivity
            allData['specLast'][cv] = specificity
            allData['accLast'][cv] = accuracy
            allData['waccLast'][cv] = waccuracy
            allData['aucLast'][cv] = auc
            mdlParams['lastlastInd'] = step
            allData['convergeTime_last'][cv] = step
            # Save best predictions
            allData['bestPred_last'][cv] = predictions
            allData['targets_Last'][cv] = targets

            # Potentially print train err
            if mdlParams['print_trainerr'] and 'train' not in eval_set:
                loss, accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy, predictions, targets, per_example = utilsMIL.getErrClassification(mdlParams, 'trainInd', modelVars)
                    # Save in mat
                save_dict_train['loss'].append(loss)
                save_dict_train['acc'].append(accuracy)
                save_dict_train['wacc'].append(waccuracy)
                save_dict_train['auc'].append(auc)
                save_dict_train['sens'].append(sensitivity)
                save_dict_train['spec'].append(specificity)
                save_dict_train['f1'].append(f1)
                save_dict_train['step_num'].append(step)
                if os.path.isfile(mdlParams['saveDir'] + '/progression_trainInd.mat'):
                    os.remove(mdlParams['saveDir'] + '/progression_trainInd.mat')
                scipy.io.savemat(mdlParams['saveDir'] + '/progression_trainInd.mat',s)




        #return loss

    #Defines the test step
    def test_step(self, batch, batch_idx):

        #when horizontal ensampble is true
        if mdlParams.get('horizontal_voting_ensemble',False):
            print('---horizontal_voting_ensemble---')
            count = 0
            for save_point in mdlParams['horizontal_voting_checkpoints']:
                state = torch.load(mdlParams['saveDir'] + '/checkpoint-ensemble' + str(save_point) + '.pt')
                # Initialize model is done outside with the proper checkpoints
                #modelVars['model'].load_state_dict(state['state_dict'])
                #modelVars['model'].eval()
                loss, accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy, predictions_n, targets_n, per_example = utilsMIL.getErrClassification(mdlParams, 'testInd', modelVars)
                if count==0:
                    predictions = predictions_n
                    targets = targets_n
                    count+=1
                else:
                    predictions = predictions + predictions_n
                    targets = targets + targets_n
            predictions = predictions/len(mdlParams['horizontal_voting_checkpoints'])
            targets = targets/len(mdlParams['horizontal_voting_checkpoints'])
            accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy  = utilsMIL.get_metrics(targets,predictions)

        else:
            modelVars['model'].eval()
            loss, accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy, predictions, targets, per_example = utilsMIL.getErrClassification(mdlParams, 'testInd', modelVars)
        allData['f1Last_Test'][cv] = f1
        allData['sensLast_Test'][cv] = sensitivity
        allData['specLast_Test'][cv] = specificity
        allData['accLast_Test'][cv] = accuracy
        allData['waccLast_Test'][cv] = waccuracy
        allData['aucLast_Test'][cv] = auc
        allData['predictions_Last_Test'][cv] = predictions
        allData['targets_Last_Test'][cv] = targets

        wandb.log({'l loss test': loss})
        wandb.log({'l accuracy test': accuracy})
        wandb.log({'l waccuracy test': np.mean(waccuracy)})
        wandb.log({'l auc test': np.mean(auc)})
        wandb.log({'l sensitivity test': sensitivity})
        wandb.log({'l specificity test': specificity})
        wandb.log({'l f1 test': f1})

         ## Evaluate Patch Performance
        print('-------------------------')
        print('Patch Performance')
        mdlParams['average_for_patients'] = False

        if mdlParams.get('horizontal_voting_ensemble',False):
            count = 0
            for save_point in mdlParams['horizontal_voting_checkpoints']:
                # need to put this outside for the model
                state = torch.load(mdlParams['saveDir'] + '/checkpoint-ensemble' + str(save_point) + '.pt')
                # Initialize model
                #modelVars['model'].load_state_dict(state['state_dict'])
                #modelVars['model'].eval()
                loss, accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy, predictions_n, targets_n, per_example = utilsMIL.getErrClassification(mdlParams, 'testInd', modelVars)
                if count==0:
                    predictions = predictions_n
                    targets = targets_n
                    count+=1
                else:
                    predictions = predictions + predictions_n
                    targets = targets + targets_n
            predictions = predictions/len(mdlParams['horizontal_voting_checkpoints'])
            targets = targets/len(mdlParams['horizontal_voting_checkpoints'])
            accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy  = utilsMIL.get_metrics(targets,predictions)

        else:
            loss, accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy, predictions, targets, per_example = utilsMIL.getErrClassification(mdlParams, 'testInd', modelVars)


        allData['f1Last_Test_Patch'][cv] = f1
        allData['sensLast_Test_Patch'][cv] = sensitivity
        allData['specLast_Test_Patch'][cv] = specificity
        allData['accLast_Test_Patch'][cv] = accuracy
        allData['waccLast_Test_Patch'][cv] = waccuracy
        allData['aucLast_Test_Patch'][cv] = auc
        allData['predictions_Last_Test_Patch'][cv] = predictions
        allData['targets_Last_Test_Patch'][cv] = targets

        wandb.log({'l loss test_Patch': loss})
        wandb.log({'l accuracy test_Patch': accuracy})
        wandb.log({'l waccuracy test_Patch': np.mean(waccuracy)})
        wandb.log({'l auc test_Patch': np.mean(auc)})
        wandb.log({'l sensitivity test_Patch': sensitivity})
        wandb.log({'l specificity test_Patch': specificity})
        wandb.log({'l f1 test_Patch': f1})

        mdlParams['average_for_patients'] = average_for_patients

        #return loss

    def configure_optimizers(self):
        optimizer = modelVars['optimizer']
        #print(optimizer)
        scheduler = modelVars['scheduler']
       
        return [optimizer] ,[scheduler]

class Attn_Net_Gated(nn.Module):
    def __init__(self,model, batch_size=16, mul_crop=4, out_class=2, model_ip= [456,456,3], num_gpu=1):
        super(Attn_Net_Gated, self).__init__()
   
        self.batchsize = batch_size
        self.multi_cropval = mul_crop
        self.out_class = out_class
        self.network = model
        self.num_gpus = num_gpu
        self.model_ip = model_ip
        self.L = self.network._fc.in_features
        self.per_partition= int(self.batchsize/self.num_gpus)
        self.network._fc = torch.nn.Identity() #nn.Linear(num_ftrs, mdlParams['numClasses'])
        self.network._swish  = torch.nn.Identity()

        #rel = self.network._swish
        #self.network._fc = torch.nn.Identity() #nn.Linear(num_ftrs, mdlParams['numClasses'])
        #self.network._swish  = torch.nn.Identity()
        self.dropout = True
        n_classes = 1
        #self.L = 1024
        self.D = 512
        self.K = 1
        
        self.attention_a = [
            nn.Linear(self.L, self.D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(self.L, self.D),
                            nn.Sigmoid()]
        if self.dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(self.D, n_classes)
    
        self.classifier = nn.Sequential(
                nn.Linear(self.L*self.K, self.out_class),
                nn.Sigmoid()
            )

    def forward(self, x,y='eval'):
        if y=='eval':
            #print('test')
            #print(x = x)
            train = False
            x = x.squeeze(0)
        else:
            #print('0',x.shape)
            train = True
            x = x.reshape(self.per_partition*self.multi_cropval, self.model_ip[2],self.model_ip[0],self.model_ip[1])#squeeze(0)#
       
        H = self.network(x)

        #print(H.shape)
        a = self.attention_a(H)
        b = self.attention_b(H)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        
        A = torch.transpose(A, 1, 0)  # KxN##[1, 60]
        #print('7 ', A.shape)
        #print(A)
        if train:
            #print('7 ', A.shape)
            A = A.reshape(-1,self.multi_cropval)  ##[15, 4]
        #print('7A',A,A.squeeze(0).shape)
        A = F.softmax(A, dim=1)  # softmax over N  ##[15, 4]
        #print('8 ', A.shape)
        #print(A)
        #print(0, A[0].shape)
        if train:
            H = H.reshape(self.per_partition,self.multi_cropval,-1)   ##[60, 1000] -> [15,4, 1000]
        #print('H', H, H.shape)
        if not train:
            M = torch.mm(A, H)
            Y_prob = self.classifier(M)
            #print(Y_prob, Y_prob.shape)
            return Y_prob
        
        for i in range (self.per_partition):
            #print(i, A[i].unsqueeze(0).shape,  H[i].shape)
            M = torch.mm(A[i].unsqueeze(0), H[i])  # KxL   #[1, 4] x  [4, 1000]-> [1, 1000]
            #print('9 ',i, M.shape)
            Y_prob = self.classifier(M)
            #print('10 ', Y_prob, Y_prob.shape)
            if i==0:
                op = Y_prob
            else:
                op= torch.cat((op, Y_prob))
        #Y_hat = torch.ge(Y_prob, 0.5).float()
        #print('11 ', Y_hat.shape)

        #print(op)
        return op#, Y_hat, A       




#assert 0
###changed
class modified_net_eff(torch.nn.Module):
    def __init__(self, model):
        super(modified_net_eff, self).__init__()
        print('MODIFIED CLASS FOR N/W EFFICIENTET')
        self.network = model
        self.relu = self.network._swish
        self.network._fc = torch.nn.Identity()
        #self.network._fc = torch.nn.Linear(1280, 2)
        self.network._swish  = torch.nn.Identity()

        #self.classifiers = torch.nn.Linear(1280, 2)
        #self.relu = rel
        self.classifiers = nn.Sequential(
            nn.Linear(1280, 2),
            self.relu
        )
    def forward(self, x):
        #print('1',x.shape)
        x  = self.network(x)
        #print('2',x,x.shape)
        #print(x.shape[0])
        if x.shape[0] >4:
            x = x.reshape(15,4,1280)
            #print('3',x,x.shape)
            x = torch.mean(x, 1)#.values
        else:
            #print('here', x, x.shape)
            x = torch.mean(x, 0).unsqueeze(0)
            #print('after', x, x.shape)

        #print('4',x,x.shape)
        x  = self.classifiers(x)
        #print('5',x.shape)
        #x = self.relu(x)
        #print('5',x.shape)
        #print(1/0)
        return x


class Eff_attention(nn.Module):
    def __init__(self,model, batch_size=16, mul_crop=4, out_class=2, model_ip= [456,456,3], num_gpu=1):
        super(Eff_attention, self).__init__()

        self.D = 512
        self.K = 1
        self.p_drop = 0.2
        self.batchsize = batch_size
        self.multi_cropval = mul_crop
        self.out_class = out_class
        if multimag == True:
            self.num_mag = 3  ###ADD THIS TO mdlPARAMS
        else:
            self.num_mag = 1
        self.network = model
        self.num_gpus = num_gpu
        self.model_ip = model_ip
        self.L = self.network._fc.in_features
        self.per_partition= int(self.batchsize/self.num_gpus)
        #rel = self.network._swish
        self.network._fc = torch.nn.Identity() #nn.Linear(num_ftrs, mdlParams['numClasses'])
        self.network._swish  = torch.nn.Identity()

        #self.classifiers = torch.nn.Linear(1280, 2)
        #self.relu = rel



        self.attention = nn.Sequential(

            nn.Linear(self.L, self.D),
            nn.Dropout(self.p_drop),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(self.p_drop),
            nn.Linear(self.L*self.K, self.out_class),
            nn.Sigmoid()
        )

    def forward(self, x,y='eval'):
        print('1 ', x.shape)
        #x = x.squeeze(0)

        #print('1 ', x.shape, y)
        if y=='eval':
            #print('test')
            #print(x = x)
            train = False
            x = x.squeeze(0)
        else:
            print('0',x.shape)
            train = True
            x = x.reshape(self.per_partition*self.multi_cropval*self.num_mag, self.model_ip[2],self.model_ip[0],self.model_ip[1])#squeeze(0)#
            #print('2',x.shape)

        H = self.network(x)   ##[60, 1000]
        
        #H = H.view(-1, 50 * 4 * 4)
        print('4 ', H.shape)
        #H = self.feature_extractor_part2(H)  # NxL
        #print('5 ', H,H.shape)
        A = self.attention(H)  # NxK##[60, 1]
        print('6 ', A.shape)
        #print(A)
        '''
        if A.shape[0] >self.multi_cropval*self.num_gpus:
            train = True
        else:
            #print('TRAIN FALSE')
            #print('6 ', A.shape)
            train = False
            #print('Train falase')
        '''
        A = torch.transpose(A, 1, 0)  # KxN##[1, 60]
        print('7 ', A.shape)
        #print(A)
        if train:
            print('7 ', A.shape)
            A = A.reshape(-1,self.multi_cropval)  ##[15, 4]
        #print('7A',A,A.squeeze(0).shape)
        A = F.softmax(A, dim=1)  # softmax over N  ##[15, 4]
        print('8 ', A.shape)
        #print(A)
        #print(0, A[0].shape)
        if train:
            H = H.reshape(self.per_partition,self.multi_cropval,-1)   ##[60, 1000] -> [15,4, 1000]
            print('9', H, H.shape)
        if not train:
            M = torch.mm(A, H)
            Y_prob = self.classifier(M)
            #print(Y_prob, Y_prob.shape)
            return Y_prob

        for i in range (self.per_partition):
            #print(i, A[i].unsqueeze(0).shape,  H[i].shape)
            M = torch.mm(A[i].unsqueeze(0), H[i])  # KxL   #[1, 4] x  [4, 1000]-> [1, 1000]
            print('10 ',i, M.shape)
            Y_prob = self.classifier(M)
            print('11 ', Y_prob, Y_prob.shape)
            if i==0:
                op = Y_prob
            else:
                op= torch.cat((op, Y_prob))
        #Y_hat = torch.ge(Y_prob, 0.5).float()
        #print('11 ', Y_hat.shape)

        #print(op)
        return op#, Y_hat, A


class all_attention(nn.Module):
    def __init__(self,model, batch_size=16, mul_crop=4, out_class=2, model_ip= [456,456,3], num_gpu=1):
        super(all_attention, self).__init__()

        self.D = 512
        self.K = 1
        self.batchsize = batch_size
        self.multi_cropval = mul_crop
        self.out_class = out_class
        self.network = model
        self.num_gpus = num_gpu
        self.model_ip = model_ip
        self.L = 1000
        self.per_partition= int(self.batchsize/self.num_gpus)
        #rel = self.network._swish
        self.network._fc = torch.nn.Identity() #nn.Linear(num_ftrs, mdlParams['numClasses'])
        self.network._swish  = torch.nn.Identity()

        #self.classifiers = torch.nn.Linear(1280, 2)
        #self.relu = rel



        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, self.out_class),
            nn.Sigmoid()
        )

    def forward(self, x,y='eval'):
        #print('1 ', x.shape)
        #x = x.squeeze(0)

        #print('1 ', x.shape, y)
        if y=='eval':
            #print('test')
            #print(x = x)
            train = False
            x = x.squeeze(0)
        else:
            #print('0',x.shape)
            train = True
            x = x.reshape(self.per_partition*self.multi_cropval, self.model_ip[2],self.model_ip[0],self.model_ip[1])#squeeze(0)#
            #print('2',x.shape)

        H = self.network(x)   ##[60, 1000]
        #print('3 ', H.shape)
        #H = H.view(-1, 50 * 4 * 4)
        #print('4 ', H.shape)
        #H = self.feature_extractor_part2(H)  # NxL
        #print('5 ', H,H.shape)
        A = self.attention(H)  # NxK##[60, 1]
        #print('6 ', A.shape)
        #print(A)
        '''
        if A.shape[0] >self.multi_cropval*self.num_gpus:
            train = True
        else:
            #print('TRAIN FALSE')
            #print('6 ', A.shape)
            train = False
            #print('Train falase')
        '''
        A = torch.transpose(A, 1, 0)  # KxN##[1, 60]
        #print('7 ', A.shape)
        #print(A)
        if train:
            #print('7 ', A.shape)
            A = A.reshape(-1,self.multi_cropval)  ##[15, 4]
        #print('7A',A,A.squeeze(0).shape)
        A = F.softmax(A, dim=1)  # softmax over N  ##[15, 4]
        #print('8 ', A.shape)
        #print(A)
        #print(0, A[0].shape)
        if train:
            H = H.reshape(self.per_partition,self.multi_cropval,-1)   ##[60, 1000] -> [15,4, 1000]
        #print('H', H, H.shape)
        if not train:
            M = torch.mm(A, H)
            Y_prob = self.classifier(M)
            #print(Y_prob, Y_prob.shape)
            return Y_prob

        for i in range (self.per_partition):
            #print(i, A[i].unsqueeze(0).shape,  H[i].shape)
            M = torch.mm(A[i].unsqueeze(0), H[i])  # KxL   #[1, 4] x  [4, 1000]-> [1, 1000]
            #print('9 ',i, M.shape)
            Y_prob = self.classifier(M)
            #print('10 ', Y_prob, Y_prob.shape)
            if i==0:
                op = Y_prob
            else:
                op= torch.cat((op, Y_prob))
        #Y_hat = torch.ge(Y_prob, 0.5).float()
        #print('11 ', Y_hat.shape)

        #print(op)
        return op

torch.backends.cudnn.benchmark = True

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

    # balance classes
    if mdlParams['balance_classes'] == 1 or mdlParams['balance_classes'] == 4:
        # Normal inverse class balancing
        class_weights = 1.0/np.mean(mdlParams['Train_Label_unique'],axis=0)

        print("Current class weights",class_weights)
        class_weights = np.power(class_weights,mdlParams['extra_fac'])
        print("Current class weights with extra",class_weights)

    elif mdlParams['balance_classes'] == 2 or mdlParams['balance_classes'] == 3:
        # Balanced sampling
        # Split training set by classes
        not_one_hot = np.argmax(mdlParams['labels_array'],1)
        mdlParams['class_indices'] = []
        for i in range(mdlParams['numClasses']):
            mdlParams['class_indices'].append(np.where(not_one_hot==i)[0])
            # Kick out non-trainind indices
            mdlParams['class_indices'][i] = np.setdiff1d(mdlParams['class_indices'][i],mdlParams['valInd'])
            #print("Class",i,mdlParams['class_indices'][i].shape,np.min(mdlParams['class_indices'][i]),np.max(mdlParams['class_indices'][i]),np.sum(mdlParams['labels_array'][np.int64(mdlParams['class_indices'][i]),:],0))


    # Set up dataloaders
    # For a normal model
    # For train
    dataset_train = utilsMIL.Bockmayr_DataSet(mdlParams, 'trainInd') # loader for traningset (String indicates train or val set)
    print('Dataset Train',len(dataset_train))
    # For val
    dataset_val = utilsMIL.Bockmayr_DataSet(mdlParams, 'valInd') # loader for val set
    print('Dataset Val',len(dataset_val))

    num_workers = psutil.cpu_count(logical=False)

    modelVars['dataloader_valInd'] = DataLoader(dataset_val, batch_size=mdlParams['multiCropEval'], shuffle=False, num_workers=num_workers, pin_memory=False)
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
    magconcat = True 
    if magconcat == True:
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
        #print('Length 2', len(data2000))
        #print('idx', data2000[2][1])
    
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
        #print('Length 2', len(data2000))
        #print('idx', data2000[2][1])


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
        #print('Length 2', len(data2000))
        #print('idx', data2000[2][1])

        #For Train, Val and Test datasets
        dataset_train = utilsMIL.ConcatMag(data2000_train,data4000_train,data8000_train)
        dataset_val = utilsMIL.ConcatMag(data2000_val,data4000_val,data8000_val)
        dataset_test = utilsMIL.ConcatMag(data2000_test ,data4000_test ,data8000_test)
        print('Train,Val,test', len(dataset_train),len(dataset_val),len(dataset_test))       

        modelVars['dataloader_valInd'] = DataLoader(dataset_val, batch_size=mdlParams['multiCropEval'], shuffle=False, num_workers=num_workers, pin_memory=False)
        modelVars['dataloader_testInd'] = DataLoader(dataset_test, batch_size=mdlParams['multiCropEval'], shuffle=False, num_workers=num_workers, pin_memory=False)

        if mdlParams['balance_classes'] == 2:
            #print(np.argmax(mdlParams['labels_array'][mdlParams['trainInd'],:],1).size(0))
            strat_sampler = utilsMIL.StratifiedSampler(mdlParams)
            modelVars['dataloader_trainInd'] = DataLoader(dataset_train, batch_size=mdlParams['batchSize'], sampler=strat_sampler, num_workers=num_workers, pin_memory=True, drop_last=True)
        else:
            modelVars['dataloader_trainInd'] = DataLoader(dataset_train, batch_size=mdlParams['batchSize'], shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
            

              
        

        

     

    # Define model
    if 'CNN_FC' in mdlParams['model_type']:
        modelVars['model'] = utilsMIL.CNN_FC(mdlParams)


        #modelVars['model'].num_classes = mdlParams['numClasses']
        #print('num c')
    else:
        ###changed
        ###changed
        if not 'vit' in mdlParams['model_type'] and not "xcit" in mdlParams['model_type']:
            modelVars['model1'] = models.getModel(mdlParams['model_type'])() #load Model
        #Select Layers for Training (Freeze Layer)
        if mdlParams['Freeze_Layer']:
            child_counter = 0
            for child in modelVars['model1'].children():
                print('Child counter',child_counter)
                print(" child", child_counter, "is -")
                print(child)
                children_of_child_counter = 0
                for children_of_child in child.children():
                    print('children_of_child_counter',children_of_child_counter)

                    print('child ', children_of_child_counter, 'is children of - ',child_counter, "is -")
                    print(children_of_child)
                    if children_of_child_counter < mdlParams['Train_Layer'] and  child_counter < mdlParams['Child_Counter'] : #control
                        for param in children_of_child.parameters():
                            param.requires_grad = False
                        print('child ', children_of_child_counter, 'of child',child_counter,' was frozen')
                    else:
                        print('child ', children_of_child_counter, 'of child',child_counter,' was not frozen')
                    children_of_child_counter += 1
                child_counter += 1

        #Change Output Layer of the Network
        if 'Dense' in mdlParams['model_type']:
            num_ftrs = modelVars['model'].classifier.in_features #get number of features of the layer before the output layer
            modelVars['model'].classifier = nn.Linear(num_ftrs, mdlParams['numClasses'])
        elif 'Squeezenet' in mdlParams['model_type']:
            modelVars['model'].classifier[1] = nn.Conv2d(512, mdlParams['numClasses'], kernel_size=(1,1), stride=(1,1))
            modelVars['model'].num_classes = mdlParams['numClasses']
        elif 'VGG' in mdlParams['model_type']:
            print('VGG-16')
            modelVars['model'].classifier[6] = nn.Linear(4096,mdlParams['numClasses'])
        elif 'AlexNet' in mdlParams['model_type']:
            modelVars['model'].classifier[6] = nn.Linear(4096,mdlParams['numClasses'])
        elif 'ResNet50' in mdlParams['model_type']:
            modelVars['model'].fc = nn.Linear(512, mdlParams['numClasses'])
        elif 'Inception' in mdlParams['model_type']:
            modelVars['model'].AuxLogits.fc = nn.Linear(768, mdlParams['numClasses'])
            modelVars['model'].fc = nn.Linear(2048, mdlParams['numClasses'])
        elif 'efficient' in mdlParams['model_type']:

            ###changed
            ###changed
            ###changed

            #print(modelVars['model1'])
            #modelVars['model'] = modified_net_eff(modelVars['model1'])
            modelVars['model'] = Eff_attention(modelVars['model1'], mdlParams['batchSize'], mdlParams['multiCropEval'], mdlParams['numClasses'], mdlParams['input_size'], len(mdlParams['numGPUs']))
            ###changed
            #(self,model, batch_size=16, mul_crop=4, out_class=2, model_ip= [3,456,456], num_gpu=1):
            #num_ftrs = modelVars['model']._fc.in_features
            #modelVars['model']._fc = nn.Linear(num_ftrs, mdlParams['numClasses'])
            #print(modelVars['model'])
        elif 'vit' in mdlParams['model_type'] or 'xcit' in mdlParams['model_type']:
            #print('here')

            #modelVars['model1'] = models.getModel('efficientnet_b0')() #load Model
            #kwargs =  {"num_classes" : mdlParams['numClasses'],
            #          }
            modelVars['model1'] = timm.create_model(mdlParams['model_type'], pretrained=True)#, **kwargs)
            modelVars['model'] = Eff_attention(modelVars['model1'], mdlParams['batchSize'], mdlParams['multiCropEval'], mdlParams['numClasses'], mdlParams['input_size'], len(mdlParams['numGPUs']))

        else:
            num_ftrs = modelVars['model'].last_linear.in_features
            modelVars['model'].last_linear = nn.Linear(num_ftrs, mdlParams['numClasses'])

    print('The Model is ', mdlParams['model_type'])
    # multi gpu support

    modelVars['model'] = modelVars['model'].to(modelVars['device'])

    print('******---*** MODEL LOADED TO GPU/CPU *****---***')
    # Loss
    if mdlParams['balance_classes'] == 3 or mdlParams['balance_classes'] == 2 or mdlParams['balance_classes'] == 0:
        modelVars['criterion'] = nn.CrossEntropyLoss()
    elif mdlParams['balance_classes'] == 4:
        modelVars['criterion'] = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor(class_weights.astype(np.float32)),reduce=False)
    else:
        modelVars['criterion'] = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor(class_weights.astype(np.float32)))

    # Setup optimizer
    if mdlParams.get('retrain_class_only') is not None and mdlParams['retrain_class_only']:
        print('Train only Classifier')
        if 'Dense' in mdlParams['model_type']:
            modelVars['optimizer'] = optim.Adam(modelVars['model'].classifier.parameters(), lr=mdlParams['learning_rate'])
        elif 'Squeezenet' in mdlParams['model_type']:
            modelVars['optimizer'] = optim.Adam(modelVars['model'].classifier.parameters(), lr=mdlParams['learning_rate'])
        elif 'VGG' in mdlParams['model_type']:
            modelVars['optimizer'] = optim.Adam(modelVars['model'].classifier.parameters(), lr=mdlParams['learning_rate'])
        elif 'Inception' in mdlParams['model_type']:
            modelVars['optimizer'] = optim.Adam(modelVars['model'].fc.parameters(), lr=mdlParams['learning_rate'])
        elif 'efficient' in mdlParams['model_type']:
            modelVars['optimizer'] = optim.Adam(modelVars['model']._fc.parameters(), lr=mdlParams['learning_rate'])
        else:
            modelVars['optimizer'] = optim.Adam(modelVars['model'].last_linear.parameters(), lr=mdlParams['learning_rate'])
    else:

        if mdlParams['Freeze_Layer']:
            print('Only Train Parts of the Network')
            modelVars['optimizer'] = optim.Adam(filter(lambda p: p.requires_grad, modelVars['model'].parameters()), lr=mdlParams['learning_rate']) # Train specific layers
        else:
            print('Train all Weights')
            modelVars['optimizer'] = optim.Adam(modelVars['model'].parameters(), lr=mdlParams['learning_rate'])

    # Decay LR by a factor of 0.1 every 7 epochs

    if mdlParams.get('CyclicLR',False):
        #modelVars['optimizer'] = optim.SGD(modelVars['model'].parameters(), lr=0.1, momentum=0.9)
        modelVars['scheduler'] =  lr_scheduler.CyclicLR(modelVars['optimizer'], base_lr=0.000025, max_lr=0.00025, step_size_up=500, cycle_momentum=False)
        modelVars['StepLR'] = False
    else:
        modelVars['scheduler'] = lr_scheduler.StepLR(modelVars['optimizer'], step_size=mdlParams['lowerLRAfter'], gamma=1/np.float32(mdlParams['LRstep']))
        #if mdlParams.get('lr_warm_up',False):
        #    warmup_scheduler = warmup.LinearWarmup(modelVars['optimizer'], warmup_period = mdlParams.get('warm_up_period',5))
        #warmup_scheduler = warmup.UntunedLinearWarmup(modelVars['optimizer'])



    # Define softmax
    modelVars['softmax'] = nn.Softmax(dim=1)
    pytorch_total_params = sum(p.numel() for p in modelVars['model'].parameters() if p.requires_grad)
    print('Number of training Parameters', pytorch_total_params)

    # Set up training  


    # Track metrics for saving best model
    mdlParams['valBest'] = 1000

    # Run training (Lightning addition)

    # Initializing WandB logger
    #wandb_logger = WandbLogger(log_model="all",project="_Digital_Phatology", name = str(sys.argv[1]) + '_epochs_'+ str(number_epochs) + '_avg_' +str(average_for_patients)+util_name)

    # Calling the Lighting modeule forthe training
    classify_model = SelfSupervised_Classifier()

    # Checkpoint
    # save path mdlParams['saveDir'] + '/checkpoint_best-' + str(step) + '.pt'
    #print(mdlParams['saveDir'])
    checkpoint_callback_all = ModelCheckpoint(auto_insert_metric_name = True, dirpath = mdlParams['saveDir'], filename='checkpoint-{epoch:02d}')

    checpoint_callback_best = ModelCheckpoint(monitor="eval_metric", auto_insert_metric_name = True, dirpath = mdlParams['saveDir'], filename='checkpoint_best-{epoch:02d}')

    checkpoint_callback_ensemble = ModelCheckpoint(monitor="ensemble_count", auto_insert_metric_name = True, dirpath = mdlParams['saveDir'], filename='checkpoint-ensemble{epoch:02d}')
    checkpoint_callback_last = ModelCheckpoint(save_last = True, auto_insert_metric_name = True, dirpath = mdlParams['saveDir'], filename='checkpoint-{epoch:02d}')
    #ADD CHECKPOINT FOR ENSEMBLE VOTING training step needs to be monitered for it
    # Define 2 checkpoint callbacks, one for every epoch and one ablove and add the wandb_logger
    # add validation after every n epochs equal to the mdlparam[display_step]
    # set max epochs to mdlparam['training_steps']
    trainer = Trainer(callbacks =[checkpoint_callback_all,checpoint_callback_best,checkpoint_callback_last,checkpoint_callback_ensemble],max_epochs=mdlParams['training_steps'],accelerator='gpu', devices=1,log_every_n_steps = 1,check_val_every_n_epoch=100,num_sanity_val_steps=0,detect_anomaly = True)
    
    # Num batches
    #numBatchesTrain = int(math.floor(len(mdlParams['trainInd'])/mdlParams['batchSize'])) #how many iterations for one epoch
    #print("Train batches",numBatchesTrain)

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
 
    # After CV Training: print CV results and save them
    print('----------------------------------------------')
    #print("Best F1:",allData['f1Best'][cv])
    #print("Best Sens:",allData['sensBest'][cv])
    #print("Best Spec:",allData['specBest'][cv])
    #print("Best Acc:",allData['accBest'][cv])
    #print("Best Per Class Accuracy:",allData['waccBest'][cv])
    #print("Best Weighted Acc:",np.mean(allData['waccBest'][cv]))
    #print("Best AUC:",allData['aucBest'][cv])
    #print("Best Mean AUC:",np.mean(allData['aucBest'][cv]))
    #print("Convergence Steps:",allData['convergeTime'][cv])
    print('----------------------------------------------')
    #print("Test Set F1:",allData['f1Best_Test'][cv])
    #print("Test Set Sens:",allData['sensBest_Test'][cv])
    #print("Test Set Spec:",allData['specBest_Test'][cv])
    #print("Test Set Acc:",allData['accBest_Test'][cv])
    #print("Test Set Per Class Accuracy:",allData['waccBest_Test'][cv])
    #print("Test Set Weighted Acc:",np.mean(allData['waccBest_Test'][cv]))
    #print("Test Set AUC:",allData['aucBest_Test'][cv])
    #print("Test Set Mean AUC:",np.mean(allData['aucBest_Test'][cv]))
    #print('----------------------------------------------')
    
    #with open(mdlParams['saveDirBase'] + '/CV.pkl', 'wb') as f:
    #    pickle.dump(allData, f, pickle.HIGHEST_PROTOCOL)

#numFolds = len(allData['f1Last_Test'])
#f1Last = np.zeros([numFolds])
#sensLast = np.zeros([numFolds,len(allData['sensLast_Test'][0])])
#specLast = np.zeros([numFolds,len(allData['specLast_Test'][0])])
#accLast = np.zeros([numFolds])
#waccLast = np.zeros([numFolds])
#allaccLast = np.zeros([numFolds,len(allData['aucLast_Test'][0])])
#aucLast = np.zeros([numFolds,len(allData['aucLast_Test'][0])])
#maucLast = np.zeros([numFolds])
#convergeTime = np.zeros([numFolds])
#np.set_printoptions(precision=4)
#predictions = []
#targets = []
#Train_acc = []

#for i in range(numFolds):

    #check if last or val Last shows Last performance
    #print("CV Set",i+1)
    #print("------------------<Take Last Model for Evaluation>------------------------------")
    #print("F1 Score",allData['f1Last_Test'][i])
    #f1Last[i] = allData['f1Last_Test'][i]
    #print("Sensitivity",allData['sensLast_Test'][i])
    #sensLast[i,:] = allData['sensLast_Test'][i]
    #print("Specificity",allData['specLast_Test'][i])
    #specLast[i,:] = allData['specLast_Test'][i]
    #print("Accuracy",allData['accLast_Test'][i])
    #accLast[i] = allData['accLast_Test'][i]
    #print("Training Accuracy",allData['accuracy_tr'])

    #if 'waccLast' in allData:
    #    print("Per Class Accuracy",allData['waccLast_Test'][i])
    #    print("Weighted Accuracy",np.mean(allData['waccLast_Test'][i]))
    #    allaccLast[i,:] = allData['waccLast_Test'][i]
    #    waccLast[i] = np.mean(allData['waccLast_Test'][i])
    #else:
        # Confusion matrix
        #print("tar",allData['targets'][i].shape,"pred",allData['LastPred'][i].shape)
    #    conf = confusion_matrix(np.argmax(allData['targets_Last_Test'][i],1),np.argmax(allData['predictions_Last_Test'][i],1))

        # Class weighted accuracy
        #wacc = conf.diagonal()/conf.sum(axis=1)
        #print("Per Class Accuracy", wacc)
        #print("Weighted Accuracy",np.mean(wacc))
        #allaccLast[i,:] = wacc
        #waccLast[i] = np.mean(wacc)
    #conf = confusion_matrix(np.argmax(allData['targets_Last_Test'][i],1),np.argmax(allData['predictions_Last_Test'][i],1))

    #print(conf)

    #print("AUC",allData['aucLast_Test'][i])
    #aucLast[i,:] = allData['aucLast_Test'][i]
    #print("Mean AUC",np.mean(allData['aucLast_Test'][i]))
    #maucLast[i] = np.mean(allData['aucLast_Test'][i])
    #if 'convergeTime' in allData:
    #    print("Convergence Steps",allData['convergeTime'][i])
    #    convergeTime[i] = allData['convergeTime'][i]


# Mean results over all folds
np.set_printoptions(precision=4)
print("-------------------------------------------------")
print("Mean over all Folds Last")
print("-------------------------------------------------")
#print("F1 Score",np.array(np.mean(f1Last)),"+-",np.array(np.std(f1Last)))
#print("Sensitivtiy",np.array(np.mean(sensLast,0)),"+-",np.array(np.std(sensLast,0)))
#print("Specificity",np.array(np.mean(specLast,0)),"+-",np.array(np.std(specLast,0)))
#print("Accuracy",np.array(np.mean(accLast)),"+-",np.array(np.std(accLast)))
#print("Per Class Accuracy",np.array(np.mean(allaccLast,0)),"+-",np.array(np.std(allaccLast,0)))
#print("Weighted Accuracy",np.array(np.mean(waccLast)),"+-",np.array(np.std(waccLast)))
#print("AUC",np.array(np.mean(aucLast,0)),"+-",np.array(np.std(aucLast,0)))
#print("Mean AUC",np.array(np.mean(maucLast)),"+-",np.array(np.std(maucLast)))
#if 'convergeTime' in allData:
#    print("Convergence Steps",np.array(np.mean(convergeTime)),"+-",np.array(np.std(convergeTime)))

# Final Logs for W&B
#wandb.log({'mean Acc - CV': np.array(np.mean(accLast))})
#wandb.log({'std Acc - CV': np.array(np.std(accLast))})

#wandb.log({'mean AUC - CV': np.array(np.mean(aucLast,0))})
#wandb.log({'std AUC - CV': np.array(np.std(aucLast,0))})

#wandb.log({'mean f1 - CV': np.array(np.mean(f1Last))})
#wandb.log({'std f1 - CV': np.array(np.std(f1Last))})

#wandb.log({'mean Sens - CV': np.array(np.mean(sensLast,0))[1] })
#wandb.log({'std Sens - CV': np.array(np.std(sensLast,0))[1]  })

#wandb.log({'mean Spec - CV': np.array(np.mean(specLast,0))[1] })
#wandb.log({'std Spec - CV': np.array(np.std(specLast,0))[1]  })

#wandb.log({'mean Wacc - CV': np.array(np.mean(waccLast))})
#wandb.log({'std Wacc - CV': np.array(np.std(waccLast))})

