# %%
import numpy as np
import os
import nibabel as nib
import pandas as pd
import sys
current_directory = os.getcwd()
functions_directory = os.path.join(current_directory, '..', 'Functions')
sys.path.append(functions_directory)
from function_plot_Kong_ptseries_dlabel_specific_parcels import plot_Kong_parcellation

# %%
path_base_input = os.path.join(current_directory,'../Data/brain_map')
path_output = os.path.join(current_directory,'../Results/SM_Figure_2')
os.makedirs(path_output, exist_ok=True)

fig_num = 'SM_Fig_2c'

# %%
path_input_FM = os.path.join(path_base_input, 'stat_FeatureMatching_FDR_p_0.05/Matching_and_Non-Matching_t_map_FDR_0.05.ptseries.nii')
path_input_Asso = os.path.join(path_base_input, 'stat_Association_FDR_p_0.05/Associated_yes_and_no_t_map_FDR_0.05.ptseries.nii')


# %%
nii_FM = nib.load(path_input_FM)
nii_Asso = nib.load(path_input_Asso)

data_FM = nii_FM.get_fdata()[0]
data_Asso = nii_Asso.get_fdata()[0]

# Replace positive values with 1, negative values with -1, and keep NaN values
data_FM_modif = np.where(np.isnan(data_FM), np.nan, np.where(data_FM > 0, 1, -1))
data_Asso_modif = np.where(np.isnan(data_Asso), np.nan, np.where(data_Asso > 0, 2, -2))

# Replace NaN values with 0 and sum the arrays element-wise
data_FM_Asso = np.nan_to_num(data_FM_modif, nan=0) + np.nan_to_num(data_Asso_modif, nan=0)
data_FM_Asso_modified = np.where(data_FM_Asso == 0, np.nan, data_FM_Asso)

fig_file = os.path.join(path_output, '%s_Association_and_feature_matching.png'%(fig_num))
plot_Kong_parcellation(data_FM_Asso_modified, fig_file,  title='Association and feature matching', cmap = 'Asso_and_FM')

# %%



