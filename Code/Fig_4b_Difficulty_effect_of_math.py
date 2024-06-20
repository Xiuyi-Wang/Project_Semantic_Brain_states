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
path_output = os.path.join(current_directory,'../Results/Figure_4')
os.makedirs(path_output, exist_ok=True)

fig_num = 'Fig_4b'

folder_Math = 'stat_Maths_FDR_p_0.05'
suffix_Math = 'task-Maths_contrast-Hard_vs_Easy_t_map_FDR_0.05'

# %%
path_Math = os.path.join(path_base_input, folder_Math, '%s.ptseries.nii'%(suffix_Math))

nii_Math = nib.load(path_Math)
data_Math = nii_Math.get_fdata()[0]

fig_math_file = os.path.join(path_output, '%s_%s.png'%(fig_num, suffix_Math))

plot_Kong_parcellation_pos_neg(data_Math, fig_math_file, color = 'psych', title='Difficulty effect of math')

# %%



