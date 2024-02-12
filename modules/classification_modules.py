import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report

import math
from datetime import datetime

def start_time():
    start = datetime.now()
    start_time = 10**6 * (int(start.strftime("%H"))*3600 + int(start.strftime("%M")) * 60 + int(start.strftime("%S"))) + int(start.strftime("%f"))
    return start_time  # in microsecond

def time_check(start_time):
    now = datetime.now()
    current_time = 10**6 * (int(now.strftime("%H"))*3600+int(now.strftime("%M"))*60+int(now.strftime("%S"))) + int(now.strftime("%f"))
    time_diff = current_time - start_time  # in microsecond
    return time_diff * 10**(-6)  # convert to seconds

def shuffle_per_class(FSL_dataset, data_per_class, classes):
    """ Shuffles data per class from a sorted dataset (acc to target) """
    FSLData = FSL_dataset['data']
    FSLTarget = FSL_dataset['target']
    if type(classes) == int:
        length = classes
    else:
        length = len(classes)
    curr_idx = 0
    for idx in range(length):
        FSLData[curr_idx:data_per_class*(idx+1)], FSLTarget[curr_idx:data_per_class*(idx+1)] = shuffle(FSLData[curr_idx:data_per_class*(idx+1)], FSLTarget[curr_idx:data_per_class*(idx+1)], random_state=42)
        curr_idx += data_per_class*(idx+1)
    return np.array(FSLData), np.array(FSLTarget)

def kfoldDivideData(FSLData, FSLTarget, data_per_class, folds=5):
    """ Partitions the dataset into k folds, each fold with equal number of data from the same class."""
    fivefoldData, fivefoldTarget = [[] for i in range(folds)], [[] for i in range(folds)]
    feat, lab = [], []
    fold = 0
    dpcpf = math.ceil(data_per_class/folds)
    cut = dpcpf

    for i in range(len(FSLTarget)):
        if i < cut:
            feat.append(FSLData[i]), lab.append(FSLTarget[i])
        else:
            fivefoldData[fold] += feat
            fivefoldTarget[fold] += lab

            feat, lab = [], []
            feat.append(FSLData[i]), lab.append(FSLTarget[i])
            fold += 1
            cut += dpcpf

        if fold == folds:
            fold = 0
    fivefoldData[folds-1] += feat
    fivefoldTarget[folds-1] += lab

    for f in range(folds):
        fivefoldData[f], fivefoldTarget[f] = shuffle(fivefoldData[f], fivefoldTarget[f], random_state=np.random.randint(folds))
    return fivefoldData, fivefoldTarget

def get_specificity(confusionMatrix, label_lists):
    specificity = {}
    for idx, label in enumerate(label_lists):
        tp, tn, fp, fn =0, 0, 0, 0
        tp = confusionMatrix[idx, idx]
        fn = sum(confusionMatrix[idx]) - tp
        for i in range(len(label_lists)):
            for j in range(len(label_lists)):
                if i == idx or j == idx:
                    continue
                else:
                    tn += confusionMatrix[i,j]
        for i in range(len(label_lists)):
            if i==idx:
                continue
            else:
                fp += confusionMatrix[idx][i]
        specificity[str(label)] = tn/(tn+fp)
    return specificity

def get_classification_report(method_labels):
    classificationReport = {key: classification_report(method_labels['true_labels'], method_labels[key], output_dict=True) for key in method_labels.keys() if key != 'true_labels'}
    for key in classificationReport.keys():
        cf = confusion_matrix(method_labels['true_labels'], method_labels[key], labels=np.unique(method_labels['true_labels']))
        specificity = get_specificity(cf, np.unique(method_labels['true_labels']))

        avg = 0
        for label in specificity.keys():
            avg += specificity[label]
            classificationReport[key][label]['specificity'] = specificity[label]

        classificationReport[key]['macro avg']['specificity'] = avg/len(specificity)
        classificationReport[key]['weighted avg']['specificity'] = avg/len(specificity)
    return classificationReport

def partitionChunks(numData:int, numChunks:int):
    perChunk = math.ceil(numData/numChunks)
    chunkInds = []
    for i in range(numChunks):
        if i == 0:
            indPair = (0, perChunk*(i+1))
        elif i == numChunks-1:
            indPair = (perChunk*i, numData)
        else:
            indPair = (perChunk*i, perChunk*(i+1))
        chunkInds.append(indPair)
    return chunkInds

def generateGraphs_chunking(classificationReport:dict, classificationMeasurements:list, classes, data_name:str, figSize:tuple, numChunks:int=5, save=0):
    colors = ['#3a2f6b','#36669c','#41a0ae','#3ec995','#77f07f']
    if type(classes) == int:
        numData = classes
    else:
        numData = len(classes)
    chunkInds = partitionChunks(numData, numChunks)
    for key in classificationReport.keys():
        accuracy = classificationReport[key]['accuracy']
        if type(classes) == int:
            clss = np.arange(classes)
        else:
            clss = np.array(classes)

        fig, axs = plt.subplots(len(classificationMeasurements)-1, numChunks, figsize=figSize)
        fig.suptitle(f'{key}_{data_name} \n accuracy = {round(accuracy, 4)}', fontsize=20)
        fig.subplots_adjust(top=0.5)

        for idx, measurement in enumerate(classificationMeasurements):
            metric_val = []
            if measurement != 'accuracy':
                for cidx, chunk in enumerate(chunkInds):
                    values = [classificationReport[key][str(c)][measurement] for c in clss[chunk[0]:chunk[1]]]
                    metric_val.extend(values)
                    axs[idx][cidx].bar(clss[chunk[0]:chunk[1]], values, color=colors[idx])

                    # formatting plot
                    axs[idx][cidx].set_ylim([0,1.15])
                    # axs[0][cidx].set_title(f'classes: {chunk[0]}-{chunk[1]-1}')
                    # axs[idx][cidx].set_xticks(clss[chunk[0]:chunk[1]])

                    if measurement == 'support':
                        axs[idx][cidx].set_ylim([0, 22])

                # average metric value
                axs[idx][0].set_ylabel(measurement+' (ave= '+str(round(np.mean(metric_val), 2)) + ')')
        plt.tight_layout()
        if save == 1:
            plt.savefig(f'final_results/{key}_{data_name}.png')