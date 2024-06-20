# %%
import numpy as np
import os
import nibabel as nib
import sys
current_directory = os.getcwd()
functions_directory = os.path.join(current_directory, '..', 'Functions')
sys.path.append(functions_directory)
from function_plot_Kong_ptseries_dlabel_use_fig_color import plot_Kong_parcellation_pos_neg

# %%
path_base_input = os.path.join(current_directory,'../Data/brain_map')
path_output = os.path.join(current_directory,'../Results/SM_Figure_2')
os.makedirs(path_output, exist_ok=True)

fig_num = 'SM_Fig_2b'

# %%
folder_FM = 'stat_FeatureMatching_FDR_p_0.05'
suffix_FM = 'Matching_and_Non-Matching_t_map_FDR_0.05'

path_FM = os.path.join(path_base_input, folder_FM, '%s.ptseries.nii'%(suffix_FM))

nii_FM = nib.load(path_FM)
data_FM = nii_FM.get_fdata()[0]

fig_FM_file = os.path.join(path_output, '%s_%s.png'%(fig_num, suffix_FM))
plot_Kong_parcellation_pos_neg(data_FM, fig_FM_file, color = 'psych', title='Feature matching')


