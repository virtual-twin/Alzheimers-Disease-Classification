# Alzheimers-Disease-Classification
ML Classification of Alzheimer's Disease vs. Mild Cognitive Impairment vs. Healthy Controls by combining empirical and simulated features as implemented for Triebkorn et al. 2022 by the [Brain Simulation Section](https://www.brainsimulation.org/bsw/) of the Berlin Institute of Health at Charité Universitätsmedizin Berlin. 

For details, see our [published manuscript](https://alz-journals.onlinelibrary.wiley.com/doi/full/10.1002/trc2.12303).

### Usage
This code was written specifically to use the dataset used in the aforementioned publication. However, to make use of this code for your own dataset, simply replace the data loading section with your own data loader, making sure to follow the following data formats:

*Xdict* is a dictionary containing the different feature matrices of shape N_subjects x N_features each. There can be any number of such feature matrices.
*Y* is a 1-dim array of class labels.
*emp_varnames* is a list of strings identifying which items in *Xdict* are empirical features.
*sim_varnames* is a list of strings identifying which items in *Xdict* are simulated features.

### Acknowledgements
If you use or reference this code, please cite our publication and this code as follows.

1. Triebkorn, Paul, et al. "Brain simulation augments machine‐learning–based classification 
of dementia." Alzheimer's & Dementia: Translational Research & Clinical Interventions 
8.1 (2022): e12303.

2. 

### Maintenance
This code is maintained and owned by Kiret Dhindsa (kiretd@gmail.com or jaskiret.dhindsa@bih-charite.de)
