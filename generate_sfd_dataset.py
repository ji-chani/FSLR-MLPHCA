from tqdm import tqdm
import numpy as np
from modules import FilipinoSignLanguage, SumFrameDiff

import warnings
warnings.filterwarnings('ignore')

######### PARAMETERS ############
zip_path = 'clips.zip'
classes = 95
save = 1

####### Collecting video paths
FSL = FilipinoSignLanguage(zip_path)
dataset = FSL.load_fsl(classes)

####### Implementing Sum of Frame Difference
FSL_dataset = {'data': [], 'target': dataset['target']}
SFD = SumFrameDiff(flattened=True)

print('Implementing Sum of Frame Difference on dataset ...')
for i, data in enumerate(tqdm(dataset['data'])):
    img, _, _ = SFD.sum_frame_diff(data)
    FSL_dataset['data'].append(img)
print('Collection of SFD images is finished.')

####### Saving Dataset
if save == 1:
    np.save(f'Dataset (SFD)/FSLdataset_{classes}classes.npy', FSL_dataset)
    print('Data saved in an npy file.')