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

fig_num = 'Fig_5f'

# %%
folder_Asso = 'stat_Association_thre_p_no_thre'
suffix_Asso = 'Associated_no_vs_yes_pm_t_map'
path_Asso = os.path.join(path_base_input, folder_Asso, '%s.ptseries.nii'%(suffix_Asso))

nii_Asso = nib.load(path_Asso)
data_Asso = nii_Asso.get_fdata()[0]

fig_Asso_file = os.path.join(path_output, '%s_%s.png'%(fig_num, suffix_Asso))
plot_Kong_parcellation(data_Asso, fig_Asso_file, cmap = 'nipy_spectral')

# %%



