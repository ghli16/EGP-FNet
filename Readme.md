EGP-FNet

Code and Datasets for "EGP-FNet: A Multimodal Fusion Network Integrating ESM-2, Graph Representations, and Physicochemical Features for Antimicrobial Peptide Prediction"

Datasets

data/CARS-AMP_train.csv and data/CARS-AMP_test.csv: The anti-Staphylococcus aureus AMP dataset created by Guan et al. used in this study. The dataset is split into training and test sets in a 9:1 ratio, as originally defined. During training, we randomly selected 10% of the training data as a validation set for early stopping.

esm2_fea_CARS-AMP/: Pre-computed ESM-2 (esm2_t33_650M_UR50D) features for all peptide sequences in the dataset.

Model

code/models_binary/final_model.pt: The trained EGP-FNet model weights. It can be loaded for inference or further evaluation on independent test sets.

code/models_binary/physico_pca.joblib: The fitted PCA model for reducing the 588-dimensional handcrafted physicochemical descriptors.

Usage

Environment Requirement

The code has been tested running under Python 3.7.12. The required packages are as follows:

torch == 1.12.1

torch-geometric == 2.3.1

torch-scatter == 2.1.1

torch-sparse == 0.6.17

torch-cluster == 1.6.1

torch-spline-conv == 1.2.2

numpy == 1.21.2

pandas == 1.3.5

scipy == 1.7.1 

scikit-learn == 1.0.2 

joblib == 1.3.2 

tqdm == 4.62.2

Training

git clone https://github.com/ghli16/EGP-FNet
cd code
python main.py

Testing

git clone https://github.com/ghli16/EGP-FNet
cd code
python test.py

Contact

Please feel free to contact us if you need any help.