import numpy as np
from modules import PHCA, MultiLevelPHCA
from modules import reduce_features
from modules import start_time, time_check, shuffle_per_class, kfoldDivideData, get_classification_report, generateGraphs_chunking

############### Filipino Sign Language Recognition using PHCA and Multi-Level PHCA #########################

############### Parameters

classes = 105
num_categories = 10
folds = 5
zip_path = 'clips.zip'
scaled = True
pca = True
category_type = 'natural'  # options: standard, natural, kmeans

data_name = f'{classes}classes_{num_categories}{category_type}_categories'
metrics = ['precision', 'recall', 'f1-score', 'specificity', 'support', 'accuracy']
models = ['phca', 'mlphca']


############## Dataset Collection (Sum of Frame Differences)

print('Data collection in progress ...')
# import SFD pixels
FSL_dataset = np.load(f'Dataset (SFD)\FSLdataset_{classes}classes.npy', allow_pickle=True)
FSL_dataset = {'data': FSL_dataset.item().get('data'),
               'target': FSL_dataset.item().get('target')}

# sorting dataset in increasing target values
sorted_inds = np.array(FSL_dataset['target']).argsort()
FSL_dataset['data'], FSL_dataset['target'] = [FSL_dataset['data'][i] for i in sorted_inds], [FSL_dataset['target'][i] for i in sorted_inds]

# shuffling dataset per class
X, y = shuffle_per_class(FSL_dataset, data_per_class=20, classes=classes)
print('Data collection finished. Features extracted using Sum of Frame Differences.')
print(f'There are {len(y)} instances having {X.shape[1]} dimensions each. \n')

############## Data Preparation and Feature Reduction

from sklearn.model_selection import train_test_split

if folds == 1:
    print('Preparing dataset for classification ...')
    # splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    print(f'There are {len(y_train)} training data and {len(y_test)} test data.')

    # feature reduction and scaling
    X_train, X_test = reduce_features(X_train, X_test, scaled, pca)

############## Classification Proper

method_labels = {key:[] for key in ['true_labels']+models}
time_learning = {key:[] for key in models}
time_predicting = {key:[] for key in models}

if folds == 1:
    print('Starting Validation --------------- \n')
    # PHCA
    start_fit = start_time()
    print('The PHCA model is learning from the data ...')
    PHCAmodel = PHCA(dim=0)
    PHCAmodel.fit(X_train, y_train)
    time_learning['phca'].append(time_check(start_fit)/len(y_train))

    # Multi-Level PHCA
    start_fit = start_time()
    print('The multi-level PHCA model is learning from the data ...')
    mlPHCA = MultiLevelPHCA(num_categories, dim=0, method=category_type)
    mlPHCA.fit(X_train, y_train)
    time_learning['mlphca'].append(time_check(start_fit)/len(y_train))
    print('Models finished learning. \n')

    method_labels['true_labels'] = y_test

    start_pred = start_time()
    print('The PHCA model is now predicting new data ...')
    method_labels['phca'] = PHCAmodel.predict(X_test)
    time_predicting['phca'].append(time_check(start_pred)/len(y_test))

    start_pred = start_time()
    print('The multi-level PHCA model is now predicting new data ...')
    method_labels['mlphca'] = mlPHCA.predict(X_test)
    time_predicting['mlphca'].append(time_check(start_pred)/len(y_test))
    print('Models finished predicting.')

else:
    fivefold_X, fivefold_y = kfoldDivideData(X, y, data_per_class=20, folds=folds)
    print(f'Starting {str(folds)}-fold validation ------------- \n')
    for val in range(folds):
        print(f'Running validation {val} ... \n')

        x_train, y_train, x_test, y_test = [], [], [], []
        for j in range(5):
            if j == val:
                x_test.extend(fivefold_X[j])
                y_test.extend(fivefold_y[j])
            else:
                x_train.extend(fivefold_X[j])
                y_train.extend(fivefold_y[j])
        x_train, x_test = reduce_features(x_train, x_test, scaled, pca)

        # PHCA
        start_fit = start_time()
        print('The PHCA model is learning from the data ...')
        PHCAmodel = PHCA(dim=0)
        PHCAmodel.fit(x_train, y_train)
        time_learning['phca'].append(time_check(start_fit)/len(y_train))
        
        # Multi-Level PHCA
        start_fit = start_time()
        print('The multi-level PHCA model is learning from the data ...')
        mlPHCA = MultiLevelPHCA(num_categories, dim=0, method=category_type)
        mlPHCA.fit(x_train, y_train)
        time_learning['mlphca'].append(time_check(start_fit)/len(y_train))
        print('Models finished learning. \n')

        method_labels['true_labels'].extend(y_test)

        start_pred = start_time()
        print('The PHCA model is now predicting new data ...')
        method_labels['phca'].extend(PHCAmodel.predict(x_test))
        time_predicting['phca'].append(time_check(start_pred)/len(y_test))
        
        start_pred = start_time()
        print('The multi-level PHCA model is now predicting new data ...')
        method_labels['mlphca'].extend(mlPHCA.predict(x_test))
        time_predicting['mlphca'].append(time_check(start_pred)/len(y_test))  # time prediction per data point
        print('Models finished predicting. \n')

# saving predictions and measured time
np.save(f'predictions/predicted_labels_{data_name}.npy', method_labels)
np.save(f'timer/time_learning_{data_name}.npy', time_learning)
np.save(f'timer/time_predicting_{data_name}.npy', time_predicting)

############## Classification Report        
from sklearn.metrics import classification_report

print()
print('Prediction results ---------------------- ')
for mod in models:
    print(f'{mod} \n')
    print(classification_report(method_labels['true_labels'], method_labels[mod]))

report = get_classification_report(method_labels)
generateGraphs_chunking(report, metrics, classes, data_name, figSize=(20,15), numChunks=5, save=1)
