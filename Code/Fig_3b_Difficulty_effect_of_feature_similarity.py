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

fig_num = 'Fig_3b'

folder_Match = 'stat_FeatureMatching_FDR_p_0.05'
suffix_Match = 'Non-Matching_vs_Matching-pm_t_map_FDR_0.05'

path_Match = os.path.join(path_base_input, folder_Match, '%s.ptseries.nii'%(suffix_Match))

# %%
#load
nii_Match = nib.load(path_Match)

#Access the data array
data_Match = nii_Match.get_fdata()[0]

fig_FM_file = os.path.join(path_output, '%s_%s.png'%(fig_num, suffix_Match))

plot_Kong_parcellation_pos_neg(data_Match, fig_FM_file, color = 'psych', title='Difficulty effect of feature similarity', scale=[-6.6,8.7])

# %%
# module_path = os.path.abspath(os.path.join('..', 'atlas', 'templates', 'function_plot_Kong_ptseries_dlabel_yanni.py'))
# spec = importlib.util.spec_from_file_location("plot_kong", module_path)
# plot_kong = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(plot_kong)


