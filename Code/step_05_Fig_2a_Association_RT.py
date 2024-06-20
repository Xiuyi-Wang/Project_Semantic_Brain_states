# %%
import os
import pandas as pd
import seaborn as sns
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt

# %%
#file input
current_directory = os.getcwd()

path_asso_input = os.path.join(current_directory,'../data/behavior_data/item_analysis_associated_trials.xlsx')
path_non_asso_input = os.path.join(current_directory,'../data/behavior_data/item_analysis_non_associated_trials.xlsx')

df_asso_input = pd.read_excel(path_asso_input)
df_non_asso_input = pd.read_excel(path_non_asso_input)

# %%
#plot figure
font_file = os.path.join(current_directory,'../templates/atlas/font/arial.ttf')
colors = {'df_asso_input': '#ea4325', 'df_non_asso_input': '#285ff6'}

fig, ax =plt.subplots(figsize=(6, 6))

fontprob_label_y=fm.FontProperties(fname=font_file, size=16)
fontprob_label_x=fm.FontProperties(fname=font_file, size=16)
fontprob_label=fm.FontProperties(fname=font_file, size=15)
sns.set(font_scale=0.7)
sns.set_style("white")

scatter_plot = sns.scatterplot(x='13_association', y='RT', hue='Dataset', data=pd.concat([df_asso_input.assign(Dataset='df_asso_input'), df_non_asso_input.assign(Dataset='df_non_asso_input')]),palette=colors,size=0.6)


sns.regplot(x='13_association', y='RT', data=df_asso_input, scatter=False, ax=scatter_plot, color=colors['df_asso_input'], label='df_asso_input')
sns.regplot(x='13_association', y='RT', data=df_non_asso_input, scatter=False, ax=scatter_plot, color=colors['df_non_asso_input'], label='df_non_asso_input')


y_min, y_max=ax.get_ylim()
plt.ylim(0.5,2.8)
plt.xlabel('Association Strength', fontproperties=fontprob_label_y)
plt.ylabel('RT',  fontproperties=fontprob_label_y)

ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

plt.setp(ax.get_yticklabels(), fontproperties=fontprob_label)
plt.setp(ax.get_xticklabels(), fontproperties=fontprob_label)

plt.legend('').set_visible(False)

save_path = os.path.join(current_directory,'../results/Figure_2/Fig_2a_AssociationStrength_RT.png')
plt.savefig(save_path, dpi=300)

# %%



