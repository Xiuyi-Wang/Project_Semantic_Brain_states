# %%
import numpy as np
import os
import nibabel as nib
import sys
current_directory = os.getcwd()
functions_directory = os.path.join(current_directory, '..', 'Functions')
sys.path.append(functions_directory)
from function_plot_Kong_ptseries_dlabel_specific_parcels import plot_Kong_parcellation

# %%
path_base_input = os.path.join(current_directory,'../Data/brain_map')
path_output = os.path.join(current_directory,'../Results/Figure_5')
os.makedirs(path_output, exist_ok=True)

fig_num = 'Fig_5e'

# %%
folder_Match = 'stat_FeatureMatching_thre_p_no_thre'
suffix_Match = 'Non-Matching_vs_Matching-pm_t_map'
path_Match = os.path.join(path_base_input, folder_Match, '%s.ptseries.nii'%(suffix_Match))

nii_Match = nib.load(path_Match)
data_Match = nii_Match.get_fdata()[0]

fig_FM_file = os.path.join(path_output, '%s_%s.png'%(fig_num, suffix_Match))
plot_Kong_parcellation(data_Match, fig_FM_file, cmap = 'nipy_spectral')

# %%



