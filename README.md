Code Description
utils.py

utils.py contains basic utility functions shared by different methods, such as dataset selection, learning rate scheduling, and other common training utilities.

INR_base/

INR_base contains the implementation of the baseline INR methods used for comparison.

load_data.py: data loading and preprocessing for baseline methods.
model.py: model definitions of the compared INR methods, including CoordNet, KD-INR, FINER, and NGP.
train.py: data loading, training, inference, and output generation for the baseline methods.
OURS_base/

OURS_base follows a similar structure to INR_base, but implements our proposed method.

load_data.py: data loading and preprocessing for our method.
model.py: model definition of our frequency decomposition INR framework.
train.py: data loading, training, inference, and output generation for our method.
INR.py

INR.py is the main script for training and inference of the baseline methods.

Example usage:

python INR.py --dataset vortex --model KDINR --mode train

For evaluation or inference only:

python INR.py --dataset vortex --model KDINR --mode eval
OURS.py

OURS.py is the main script for training and inference of our proposed method.

Example usage:

python OURS.py --dataset vortex --mode train

For evaluation or inference only:

python OURS.py --dataset vortex --mode eval
Supported Baselines

The following baseline methods are implemented in INR_base/model.py:

CoordNet
KD-INR
FINER
NGP
Dataset

The dataset/ folder contains example volume data used by the scripts. The raw volume files are read during training and inference according to the dataset name specified by the command line arguments.

Notes

The training and inference outputs are saved under the result directory specified in the scripts. Please check the corresponding output folders after running training or inference.
