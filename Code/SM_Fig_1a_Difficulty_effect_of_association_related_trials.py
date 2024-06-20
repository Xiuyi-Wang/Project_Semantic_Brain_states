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
path_asso_neg_t = os.path.join(current_directory,'../Data/brain_map/stat_Association_thre_p_no_thre/Associated_pm_neg_t_map.ptseries.nii')
path_output = os.path.join(current_directory,'../Results/SM_Figure_1')
os.makedirs(path_output, exist_ok=True)

fig_num = 'SM_Fig_1a'

# %%
nii_asso = nib.load(path_asso_neg_t)
data_asso = nii_asso.get_fdata()[0]

fig_asso_output = os.path.join(path_output, '%s_Related_trials.png'%(fig_num))
plot_Kong_parcellation(data_asso, fig_asso_output, title = 'Related trials')

# %%



