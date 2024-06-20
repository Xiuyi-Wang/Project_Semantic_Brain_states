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
path_output = os.path.join(current_directory,'../Results/Figure_6')
os.makedirs(path_output, exist_ok=True)

fig_num = 'Fig_6a'

# %%
folder_spatial = 'stat_Spatial_thre_p_no_thre'
suffix_spatial = 'task-Spatial_contrast-Hard_vs_Easy_t_map'

path_spatial = os.path.join(path_base_input, folder_spatial, '%s.ptseries.nii'%(suffix_spatial))

# %%
nii_spatial = nib.load(path_spatial)
data_spatial = nii_spatial.get_fdata()[0]
fig_spatial_file = os.path.join(path_output, '%s_%s.png'%(fig_num, suffix_spatial))
plot_Kong_parcellation(data_spatial, fig_spatial_file, cmap = 'nipy_spectral',vrange=[-11,13.2], title='Difficulty effect of spatial working memory')

# %%



