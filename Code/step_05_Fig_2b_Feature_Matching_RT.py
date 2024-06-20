# %%
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import font_manager as fm

# %%
#file input
current_directory = os.getcwd()

path_fm_input = os.path.join(current_directory, '../data/behavior_data/item_analysis_matching_trials.xlsx')
path_non_fm_input = os.path.join(current_directory, '../data/behavior_data/item_analysis_nonmatching_trials.xlsx')

df_fm_input = pd.read_excel(path_fm_input)
df_non_fm_input = pd.read_excel(path_non_fm_input)

# %%
fig, ax =plt.subplots(figsize=(6, 6))

font_file = os.path.join(current_directory,'../templates/atlas/font/arial.ttf')
colors = {'df_fm_input': '#ea4325', 'df_non_fm_input': '#285ff6'}

fontprob_label_y=fm.FontProperties(fname=font_file, size=16)
fontprob_label_x=fm.FontProperties(fname=font_file, size=16)
fontprob_label=fm.FontProperties(fname=font_file, size=15)
sns.set(font_scale=0.7)
sns.set_style("white")

scatter_plot_rt = sns.scatterplot(x='11_feature_simi', y='RT', hue='Dataset', data=pd.concat([df_fm_input.assign(Dataset='df_fm_input'), df_non_fm_input.assign(Dataset='df_non_fm_input')]),palette=colors,size=0.6)

sns.regplot(x='11_feature_simi', y='RT', data=df_fm_input, scatter=False, ax=scatter_plot_rt, color=colors['df_fm_input'], label='df_fm_input')
sns.regplot(x='11_feature_simi', y='RT', data=df_non_fm_input, scatter=False, ax=scatter_plot_rt, color=colors['df_non_fm_input'], label='df_non_fm_input')

plt.ylim(0.5, 2.8)
plt.xlabel('Feature Similarity', fontproperties=fontprob_label_y)
plt.ylabel('RT',  fontproperties=fontprob_label_y)

plt.setp(ax.get_yticklabels(), fontproperties=fontprob_label)
plt.setp(ax.get_xticklabels(), fontproperties=fontprob_label)

ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

save_path = os.path.join(current_directory,'../results/Figure_2/Fig_2b_FeatureSimilarity_RT.png')
plt.savefig(save_path, dpi=300)

# %%



