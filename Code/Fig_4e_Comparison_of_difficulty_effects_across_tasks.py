# %%
import numpy as np
import pandas as pd
import os
import nibabel as nib
from scipy.io import loadmat
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import sys
current_directory = os.getcwd()
functions_directory = os.path.join(current_directory, '..', 'Functions')
sys.path.append(functions_directory)
from function_plot_Kong_ptseries_dlabel_specific_parcels import plot_Kong_parcellation
from function_plot_violin_baihan import  plot_violin_v7

# %%
#Figure 4e left: Overlap between spatial and math brainmap
path_brainmap_input = os.path.join(current_directory,'../Data/brain_map')
path_output = os.path.join(current_directory,'../Results/Figure_4')
os.makedirs(path_output, exist_ok=True)

fig_num = 'Fig_4e'

folder_Spatial = 'stat_Spatial_FDR_p_0.05'
folder_Math = 'stat_Maths_FDR_p_0.05'

suffix_Spatial = 'task-Spatial_contrast-Hard_vs_Easy_t_map_FDR_0.05'
suffix_Math = 'task-Maths_contrast-Hard_vs_Easy_t_map_FDR_0.05'

# %%
path_Spatial = os.path.join(path_brainmap_input, folder_Spatial, '%s.ptseries.nii'%(suffix_Spatial))
path_Math = os.path.join(path_brainmap_input, folder_Math, '%s.ptseries.nii'%(suffix_Math))

nii_Spatial = nib.load(path_Spatial)
nii_Math = nib.load(path_Math)

data_Spatial = nii_Spatial.get_fdata()[0]
data_Math = nii_Math.get_fdata()[0]

#get positive
data_Spatial[data_Spatial<0] = np.nan
data_Math[data_Math<0] = np.nan 

# %%
spatial_overlap_math = np.full((400,), np.nan)
spatial_overlap_math[np.isfinite(data_Spatial) & np.isfinite(data_Math)] = 1

# Create a mask for non-NaN values
non_nan_mask = ~np.isnan(spatial_overlap_math)

# Generate random values between 0.5 and 1 for non-NaN positions
random_values = np.random.uniform(0.5, 1, size=spatial_overlap_math.shape)

# Replace non-NaN values with random values
spatial_overlap_math[non_nan_mask] = random_values[non_nan_mask]

fig_spatial_overlap_math = os.path.join(path_output, '%s_Overlap_between_spatial_and_math_brainmap.png'%(fig_num))


plot_Kong_parcellation(spatial_overlap_math, fig_spatial_overlap_math, cmap = 'non-semantic',title = 'Overlap between spatial and math')

# %%
#Figure 4e right: Difficulty effects violin stats 
path_stat_file = os.path.join(current_directory,'../Data/compare_feat_asso_beta_in_MDN/stat.xlsx')
df_stat = pd.read_excel(path_stat_file)

path_activation_file = os.path.join(current_directory,'../Data/compare_feat_asso_beta_in_MDN/activation_strength_each_condition.mat')
data = loadmat(path_activation_file)["beta_each_condition"]


# %%
condition_names = ['Association', 'Feature Matching']
condition_names_part = condition_names[0:4]

column_names = ['feature_MDN', 'association_MDN']
df = pd.DataFrame(data = data, columns =column_names)

# Define the desired column order
desired_column_order = ['association_MDN', 'feature_MDN']  # Example new order of columns

# Reorder the columns based on the desired order
df = df[desired_column_order]


df['sub_ID'] = np.arange(1,28)

df_long = pd.melt(df, id_vars=['sub_ID'], var_name='ROI', value_name='beta')

# %%
# Convert HEX to RGB
hex_color_1 = "#E8E830"
rgb_color_1 = list(int(hex_color_1[i:i + 2], 16)/255 for i in (1, 3, 5))

rgb_color_2 = [0,0,0.88]


hex_color_3 = "#FF4500"
rgb_color_3 = list(int(hex_color_3[i:i + 2], 16)/255 for i in (1, 3, 5))
rgb_color_4 = [0.54,0.17,0.89]
palette = {column_names[0]: rgb_color_1,column_names[1]: rgb_color_3}

rgb_black = [0,0,0]
# Define colors for x-axis tick labels
label_colors = {condition_names[0]: rgb_black,condition_names[1]: rgb_black}


fig_violin = os.path.join(path_output,'%s_Difficulty_effects_across_tasks_violin.png'%(fig_num))

# %%
# calculate the stat values
# compare whether each adjacent part is significantly different.

data_1 = data[:, 0]
data_2 = data[:, 1]

data_1_mean = np.round(np.mean(data_1),3)
data_2_mean = np.round(np.mean(data_2),3)

t, p = ttest_rel(data_1,data_2)

# %%
order_mean = desired_column_order
sns.set(font_scale=2)
sns.set_style("white")
fig_size = (6, 8)
fig, ax = plt.subplots(figsize=fig_size)
plt.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.05)
ax, fontprob_label = plot_violin_v7(figfile=fig_violin, ax=ax, df=df_long,data = df, x_variable="ROI",
                                    y_variable='beta', df_stats=df_stat, p_variable='p_value',
                                    order=order_mean,x_ticklabels = condition_names_part,label_colors = label_colors,
                                    title='', x_label='',
                                    y_label='Difficulty effect', fontsize_title=20, fontsize_label_x = 20, fontsize_label_y = 20,
                                    palette=palette,
                                    y_lim=None, y_ticks=None, y_ticklabels=None, rotation=0)

# %%



