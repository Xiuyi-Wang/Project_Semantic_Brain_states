# %%
import numpy as np
import os
import nibabel as nib
import sys
current_directory = os.getcwd()
functions_directory = os.path.join(current_directory, '..', 'Functions')
sys.path.append(functions_directory)
from function_plot_Kong_ptseries_dlabel_baihan import plot_Kong_parcellation

# %%
path_non_fm_t = os.path.join(current_directory,'../Data/brain_map/stat_FeatureMatching_thre_p_no_thre/Non-Matching_pm_t_map.ptseries.nii')
path_output = os.path.join(current_directory,'../Results/SM_Figure_1')
os.makedirs(path_output, exist_ok=True)

fig_num = 'SM_Fig_1d'

# %%
nii_non_fm = nib.load(path_non_fm_t)
data_non_fm = nii_non_fm.get_fdata()[0]

fig_non_fm_output = os.path.join(path_output, '%s_Difficulty_effect_of_feature_matching_non-matching_trials.png'%(fig_num))
plot_Kong_parcellation(data_non_fm, fig_non_fm_output, title='Non-matching trials')


