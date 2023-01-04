import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_SB, CLAM_MB
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import collections
def initiate_model(args, ckpt_path):
    print('Init Model')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}

    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})

    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type =='clam_mb':
        model = CLAM_MB(**model_dict)
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    print_network(model)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    model.relocate()
    model.eval()
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)

    print('Init Loaders')
    loader = get_simple_loader(dataset)
    patient_results, test_error, auc, df, _, acc_pat,auc_score_pat = summary(model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model, patient_results, test_error, auc, df, acc_pat,auc_score_pat

def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    #print(slide_ids)
    case_ids = loader.dataset.slide_data['case_id']
    #print(case_ids)

    data_dict = collections.defaultdict(list)
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, results_dict = model(data)

        acc_logger.log(Y_hat, label)

        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        #print(batch_idx,probs, Y_hat.item())

        #patient_results.update({{ 'slide_id': case_ids[batch_idx], 'prob': probs, 'label': label.item()}})
        data_dict[case_ids[batch_idx]].append({ 'slide_id': slide_id, 'prob': probs, 'y_hat':Y_hat.item(), 'label': label.item()})
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else:
        if args.n_classes == 2:
            #print(all_labels, all_probs)
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
            #print('AUC_score',auc_score)
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)



    #print(data_dict)

    patient_pred = np.zeros((len(data_dict.keys()), args.n_classes))
    patient_labels = np.zeros(len(data_dict.keys()))
    slide_list_pat = []
    count = 0
    for key in data_dict:
        #print('key:', key)
        #print(data_dict[key])
        #print(len(data_dict[key]))
        num_slides = len(data_dict[key])
        comb_prob = np.zeros(args.n_classes)
        patient_label = np.zeros(1)
        slide_list= []
        for i in range(num_slides):
            #print(data_dict[key][i]['prob'].shape)
            comb_prob = comb_prob + np.array(data_dict[key][i]['prob'][0])
            patient_label = patient_label + np.array(data_dict[key][i]['label'])
            slide_list.append(data_dict[key][i]['slide_id'])


            #print(comb_prob)
            #(count)
        slide_list_pat.append(slide_list)
        patient_pred[count] = comb_prob/num_slides
        patient_labels[count] = patient_label/num_slides
        count += 1
        #print(count)
    #print(patient_pred,)
    patient_results = {}
    acc_pat = accuracy_score(patient_labels, np.argmax(patient_pred, axis = 1))
    #print('ACC PAT', acc_pat)
    if args.n_classes == 2:
        auc_score_pat = roc_auc_score(patient_labels, patient_pred[:, 1])
        #print('AUCSCORE IS' ,auc_score_pat )
    else:
        binary_labels = label_binarize(patient_labels, classes=[i for i in range(args.n_classes)])
        #print(binary_labels, patient_labels)
        for class_idx in range(args.n_classes):
            if class_idx in patient_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], patient_pred[:, class_idx])
                aucs.append(auc(fpr, tpr))
            else:
                aucs.append(float('nan'))
        if args.micro_average:
            binary_labels = label_binarize(patient_labels, classes=[i for i in range(args.n_classes)])
            fpr, tpr, _ = roc_curve(binary_labels.ravel(), patient_pred.ravel())
            auc_score_pat = auc(fpr, tpr)
        else:
            auc_score_pat = np.nanmean(np.array(aucs))


    patient_dict= {}
    #print(np.argmax(patient_pred[0]))
    count = 0
    for keys in data_dict:
        #print(data_dict[keys])

        patient_dict.update({keys:{  'slide_id':slide_list_pat[count] ,'prob_1': patient_pred[count][0],'prob_2': patient_pred[count][1],  'label': patient_labels[count], 'Y_pred': int(np.argmax(patient_pred[count]))}})
        count +=1
    #print(patient_dict)
    #assert 0
    #print(data_dict)
    return patient_dict, test_error, auc_score, df, acc_logger, acc_pat,auc_score_pat
