# %%
import os
import sys
import importlib.util
import nibabel as nib
current_directory = os.getcwd()
functions_directory = os.path.join(current_directory, '..', 'Functions')
sys.path.append(functions_directory)
from function_plot_Kong_ptseries_dlabel_use_fig_color import plot_Kong_parcellation_pos_neg

# %%
path_base_input = os.path.join(current_directory,'../Data/brain_map')
path_output = os.path.join(current_directory,'../Results/Figure_3')
os.makedirs(path_output, exist_ok=True)

fig_num = 'Fig_3a'

folder_Asso = 'stat_Association_FDR_p_0.05'
suffix_Asso = 'Associated_no_vs_yes_pm_t_map_FDR_0.05'

path_Asso = os.path.join(path_base_input, folder_Asso, '%s.ptseries.nii'%(suffix_Asso))



# %%
nii_Asso = nib.load(path_Asso)

data_Asso = nii_Asso.get_fdata()[0]

fig_Asso_file = os.path.join(path_output, '%s_%s.png'%(fig_num, suffix_Asso))

plot_Kong_parcellation_pos_neg(data_Asso, fig_Asso_file, color = 'psych', title='Difficulty effect of association strength', scale=[-6.6,8.7])

# %%



