# %%
import nibabel as nib
import numpy as np
import os
from scipy.io import loadmat, savemat
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from matplotlib import font_manager as fm
import pandas as pd
import sys
current_directory = os.getcwd()
functions_directory = os.path.join(current_directory, '..', 'Functions')
sys.path.append(functions_directory)
from function_plot_violin_baihan import  plot_violin_v6
from statsmodels.stats.multitest import  fdrcorrection

# %%
file_dist = os.path.join(current_directory, '../Data/global_minimum_distance/distance_each_condition.mat')
file_stat = os.path.join(current_directory, '../Data/global_minimum_distance/stat.xlsx')

path_output = os.path.join(current_directory,'../Results/Figure_4')
os.makedirs(path_output, exist_ok=True)

fig_num = 'Fig_4f'

# %%
df_stat = pd.read_excel(file_stat)

# load the distance data
data = loadmat(file_dist)['distance_each_condition'][:,0:4]

column_names = ['feature_only', 'feature_MDN', 'association_MDN', 'association_only']

df = pd.DataFrame(data = data, columns =column_names)

# Define the desired column order
desired_column_order = ['association_only', 'association_MDN','feature_MDN', 'feature_only']  # Example new order of columns


# Reorder the columns based on the desired order
df = df[desired_column_order]

df['sub_ID'] = np.arange(1,246)

condition_names = ['Association only','Association and non-semantic', 'Feature matching and non-semantic', 'Feature matching only']

df_long = pd.melt(df, id_vars=['sub_ID'], var_name='ROI', value_name='Distance')

# %%
#Convert HEX to RGB
#Feature mathching only
hex_color_1 = "#E8E830"
rgb_color_1 = list(int(hex_color_1[i:i + 2], 16)/255 for i in (1, 3, 5))
#FM overlap spatial or math
rgb_color_2 = '#006400'
#Asso overlap spatial or math
rgb_color_3 = '#098b8b'
#Association only
hex_color_4 = "#FF4500"
rgb_color_4 = list(int(hex_color_4[i:i + 2], 16)/255 for i in (1, 3, 5))
palette = {column_names[0]: rgb_color_1,column_names[1]: rgb_color_2, column_names[2]:rgb_color_3, column_names[3]: rgb_color_4}

# %%
fig_violin = os.path.join(path_output,'%s_Global_minimum_distance_to_sensory-motor_cortex.png'%(fig_num))

# calculate the stat values
# compare whether each adjacent part is significantly different.

p_all = []
for j in range(3):
    data_1 = data[:, j]
    data_2 = data[:, j+1]

    data_1_mean = np.round(np.mean(data_1),3)
    data_2_mean = np.round(np.mean(data_2),3)

    t, p = ttest_rel(data_1,data_2)
    p_all.append(p)

    # print (data_1_mean, data_2_mean, t, p)

q_all_b, q_all = fdrcorrection(p_all, alpha=0.05, method='indep', is_sorted=False)

# print (q_all_b, q_all)

order_mean = desired_column_order
sns.set(font_scale=2)
sns.set_style("white")
fig_size = (8, 8)
fig, ax = plt.subplots(figsize=fig_size)
plt.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.05)
ax, fontprob_label = plot_violin_v6(figfile=fig_violin, ax=ax, df=df_long,data = df, x_variable="ROI",
                                    y_variable='Distance', df_stats=df_stat, p_variable='p_value',
                                    order=order_mean,x_ticklabels = condition_names,
                                    title='', x_label='',
                                    y_label='Global minimum distance', fontsize_title=20, fontsize_label_x = 12, fontsize_label_y = 20,
                                    palette=palette,
                                    y_lim=None, y_ticks=None, y_ticklabels=None,rotation=0)

# %%



