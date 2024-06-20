# plot the violin figure
import os
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import itertools
from scipy import stats
import seaborn as sns
import pandas as pd


current_directory = os.getcwd()
font_file = os.path.join(current_directory, '../Templates/Atlas/Font/arial.ttf')
file_atlas = os.path.join(current_directory, '../Templates/Atlas/Parcellation_Kong_2022/Schaefer_417.xlsx')

cols_short=['Vis-A','Vis-B','Vis-C','Aud','SM-A','SM-B','VAN-A', 'VAN-B', 'DAN-A','DAN-B','Cont-A','Cont-B','Cont-C','Lang','DMN-A','DMN-B','DMN-C']

def ttest_violin(df, order, pairlist, x_variable, y_variable, groupnum=0, adjacent=True, method= 't-test_paired'):
    
    # do t-test for only adjacent pair under order, or all unique comparison between group
    if adjacent:
        order_filter = [net for net in order if net in pairlist]
        stats_pairs = [(order_filter[k-1], order_filter[k]) for k in range(1,len(order_filter))]
    else:
        stats_pairs = itertools.combinations(pairlist, 2)

    N1=list(); N2=list(); t_value=list(); p_value=list()
    for (x1,x2) in stats_pairs:
        y1, y2 = df.loc[df[x_variable]==x1, y_variable], df.loc[df[x_variable]==x2, y_variable]
        if method == 't-test_paired':
            t, p = stats.ttest_rel(y1, y2)
        elif method == 't-test_ind':
            t, p = stats.ttest_ind(y1, y2)
        else:
            raise ValueError('method only accept t-test_paired or t-test_ind')

        N1.append(x1); N2.append(x2); t_value.append(t); p_value.append(p)
    
    df_stats = pd.DataFrame({"N1":N1, "N2":N2, "t_value":t_value, "p_value":p_value, "group":groupnum})

    return df_stats

def star_sig(p):
    # change p value to star string
    if   p<.001:
        return '***'
    elif p< .01:
        return '**'
    elif p< .05:
        return '*'
    else:
        return 'ns'

def plot_violin_v3(df,x_variable,y_variable,fig_size, order, figfile,
                    title=None, x_label='', y_label='', fontsize_title=10,fontsize_label=10, palette=None,
                    y_lim=None, y_ticks=None):

    """
    this is for plotting the confusion matrix, only including the top 7 networks
    :param df: is the dataframe for plot: columns:x_variable,y_variable
    :param x_variable: the xlabel
    :param y_variable: y values
    :param fig_size:
    :param y_label:
    :param order: the order of x_variable
    :param palette:
    :param title:
    :param figfile:
    :return:
    """
    fontprob_label=fm.FontProperties(fname=font_file, size=fontsize_label)

    a4_dims = fig_size
    fig, ax = plt.subplots(figsize=a4_dims)

    plt.rcParams.update({'font.family':'Arial', 'font.size': fontsize_label})
    sns.violinplot(ax = ax,x=x_variable, y=y_variable, data=df, order=order, linewidth=0.8,palette=palette)
    sns.stripplot(ax = ax, x=x_variable, y=y_variable, data=df, order=order, palette=palette, jitter=True, dodge=True, size=3, linewidth=0.5)
    
    ax.set_xticklabels(order,fontproperties=fontprob_label,rotation=90)
    plt.setp(ax.get_yticklabels(), fontproperties=fontprob_label)
    #ax.tick_params(axis='y', labelsize=fontsize_label)
    
    if title is not None:
        ax.set_title(title,fontsize=fontsize_title,fontweight='bold', y=1)
    
    
    ax.set_xlabel(x_label, fontproperties=fontprob_label)
    ax.set_ylabel(y_label, fontproperties=fontprob_label, x=-0.2)
    
    
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
   

    # Only show ticks on the left and bottom spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.locator_params(axis='y', nbins=5) 

    plt.subplots_adjust(left=0.12, right=0.99, bottom=0.2, top=0.92, wspace=0.0, hspace=0)
    
    #save
    plt.savefig(figfile,dpi=600)
    figfile2=figfile[:-4]+'.pdf'
    plt.savefig(figfile2,format="pdf")
    plt.close()

def plot_violin_v4(ax, df,x_variable,y_variable,  fig_size, order, figfile,
                    title=None, x_label='', y_label='', fontsize_title=10,fontsize_label=10, palette=None,
                    y_lim=None, y_ticks=None,y_ticklabels=None):

    """
    this is for plotting the confusion matrix, compared to v4, this added the statistical significance given pairlist and p-values.
    :param df: is the dataframe for plot: columns:x_variable,y_variable
    :param x_variable: the xlabel
    :param y_variable: y values
    :param fig_size:
    :param y_label:
    :param order: the order of x_variable
    :param palette:
    :param title:
    :param figfile:
    :return:
    """
    fontprob_label=fm.FontProperties(fname=font_file, size=fontsize_label)

    # plot 
    sns.violinplot(ax = ax,x=x_variable, y=y_variable, data=df, order=order, linewidth=0.8,palette=palette, cut=2)
    sns.stripplot(ax = ax, x=x_variable, y=y_variable, data=df, order=order, palette=palette, jitter=True, dodge=True, size=4, edgecolor='white', linewidth=0.5)
    
    #labeling
    ax.set_xticklabels(order,fontproperties=fontprob_label,rotation=90)
    plt.setp(ax.get_yticklabels(), fontproperties=fontprob_label)

    #labeling
    ax.set_xlabel(x_label, fontproperties=fontprob_label)
    ax.set_ylabel(y_label, fontproperties=fontprob_label, x=-0.2)

    if title is not None:
        ax.set_title(title,fontsize=fontsize_title,fontweight='bold', y=1)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if y_ticklabels is not None:
        ax.set_yticklabels(y_ticklabels)
   

    # Only show ticks on the left and bottom spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #plt.locator_params(axis='y', nbins=5) 


    return ax, fontprob_label


def plot_violin_v5(figfile, ax, df,x_variable,y_variable, df_stats, p_variable,  order,x_ticklabels,
                    title=None, x_label='', y_label='', fontsize_title=10,fontsize_label=10, palette=None,
                    y_lim=None, y_ticks=None,y_ticklabels=None,rotation=90):

    """
    this is for plotting the confusion matrix, only including the top N networks
    :param df: is the dataframe for plot: columns:x_variable,y_variable
    :param x_variable: the xlabel
    :param y_variable: y values
    :param fig_size:
    :param y_label:
    :param order: the order of x_variable
    :param palette:
    :param title:
    :param figfile:
    :return:
    """
    fontprob_label=fm.FontProperties(fname=font_file, size=fontsize_label)
    # fontprob_label_s=fm.FontProperties(fname=font_file, size=fontsize_label-1)
    fontprob_label_s=fm.FontProperties(fname=font_file, size=fontsize_label)
    # plot
    sns.violinplot(ax = ax,x=x_variable, y=y_variable, data=df, order=order, linewidth=0.8,palette=palette, cut=2)
    sns.stripplot(ax = ax, x=x_variable, y=y_variable, data=df, order=order, palette=palette, jitter=True, dodge=True, size=4, edgecolor='k', linewidth=0.5)
    

    #labeling
    # ax.set_xticklabels(order,fontproperties=fontprob_label,rotation=rotation)
    ax.set_xticklabels(x_ticklabels,fontproperties=fontprob_label,rotation=rotation)
    plt.setp(ax.get_yticklabels(), fontproperties=fontprob_label)

    #labeling
    ax.set_xlabel(x_label, fontproperties=fontprob_label)
    ax.set_ylabel(y_label, fontproperties=fontprob_label, x=-0.2)

    if title is not None:
        ax.set_title(title,fontsize=fontsize_title, y=1.05)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if y_ticklabels is not None:
        ax.set_yticklabels(y_ticklabels)
    
    y_min, y_max=ax.get_ylim()
    
    # add the star statistical annotation
    line_width=1
    y, h, col = y_max, (y_max-y_min)*0.02 , 'k'
    for index, row in df_stats.iterrows():
        x1, x2 = order.index(row['N1'])+0.05, order.index(row['N2'])-0.05  # x coordinates of two networks
        sig_str = star_sig(row[p_variable])
        
        ax.plot([x1, x1, x2, x2], [y-h, y, y, y-h], lw=line_width, c=col)
        if sig_str == 'ns':
            # baihan ori value
            # ax.text((x1+x2)*.5, y-h*0.4, sig_str, ha='center', va='bottom', color=col, fontproperties=fontprob_label_s)
            ax.text((x1+x2)*.5, y-h*0.01, sig_str, ha='center', va='bottom', color=col, fontproperties=fontprob_label_s)
        else:
            # baihan ori value
            # ax.text((x1+x2)*.5, y-h*1.5, sig_str, ha='center', va='bottom', color=col, fontproperties=fontprob_label)
            ax.text((x1+x2)*.5, y-h*0.1, sig_str, ha='center', va='bottom', color=col, fontproperties=fontprob_label)

    # Only show ticks on the left and bottom spines
    # ax.spines.right.set_visible(False)
    # ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #plt.locator_params(axis='y', nbins=5)

    plt.savefig(figfile, dpi=300)
    plt.close()
        
    return ax, fontprob_label


def plot_violin_v6(figfile, ax, df,data, x_variable, y_variable, df_stats, p_variable, order, x_ticklabels,
                   title=None, x_label='', y_label='', fontsize_title=10, fontsize_label_x=10,fontsize_label_y=10, palette=None,
                   y_lim=None, y_ticks=None, y_ticklabels=None, rotation=90):
    """
    this is for plotting the confusion matrix, only including the top N networks
    :param df: is the dataframe for plot: columns:x_variable,y_variable
    :param x_variable: the xlabel
    :param y_variable: y values
    :param fig_size:
    :param y_label:
    :param order: the order of x_variable
    :param palette:
    :param title:
    :param figfile:
    :return:
    """
    fontprob_label_x = fm.FontProperties(fname=font_file, size=fontsize_label_x)
    fontprob_label_y = fm.FontProperties(fname=font_file, size=fontsize_label_y)
    # fontprob_label_s=fm.FontProperties(fname=font_file, size=fontsize_label-1)
    fontprob_label_s = fm.FontProperties(fname=font_file, size=fontsize_label_x)
    # plot
    sns.violinplot(ax=ax, x=x_variable, y=y_variable, data=df, order=order, linewidth=0.8,  palette=palette, cut=2)
    sns.stripplot(ax=ax, x=x_variable, y=y_variable, data=df, order=order, palette=palette, jitter=True, dodge=False,
                  size=4, edgecolor='k', linewidth=0.5)

    # Remove upper and right frame lines
    sns.despine(top=True, right=True)

    # Show the mean of each column using white large circles
    for i, column in enumerate(order):
        mean_value = data[column].mean()
        ax.scatter(i, mean_value,  color='white', s=200, zorder=3, marker='o', edgecolor='black',facecolors='none')

    # labeling
    # ax.set_xticklabels(order,fontproperties=fontprob_label,rotation=rotation)
    ax.set_xticklabels(x_ticklabels, fontproperties=fontprob_label_x, rotation=rotation)
    plt.setp(ax.get_yticklabels(), fontproperties=fontprob_label_y)

    # labeling
    ax.set_xlabel(x_label, fontproperties=fontprob_label_x)
    ax.set_ylabel(y_label, fontproperties=fontprob_label_y, x=-0.2)

    if title is not None:
        ax.set_title(title, fontsize=fontsize_title, y=1.05)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if y_ticklabels is not None:
        ax.set_yticklabels(y_ticklabels)

    y_min, y_max = ax.get_ylim()

    # add the star statistical annotation
    line_width = 1
    y, h, col = y_max, (y_max - y_min) * 0.02, 'k'
    for index, row in df_stats.iterrows():
        x1, x2 = order.index(row['N1']) + 0.05, order.index(row['N2']) - 0.05  # x coordinates of two networks
        sig_str = star_sig(row[p_variable])

        ax.plot([x1, x1, x2, x2], [y - h, y, y, y - h], lw=line_width, c=col)
        if sig_str == 'ns':
            # baihan ori value
            # ax.text((x1+x2)*.5, y-h*0.4, sig_str, ha='center', va='bottom', color=col, fontproperties=fontprob_label_s)
            ax.text((x1 + x2) * .5, y - h * 0.01, sig_str, ha='center', va='bottom', color=col,
                    fontproperties=fontprob_label_s)
        else:
            # baihan ori value
            # ax.text((x1+x2)*.5, y-h*1.5, sig_str, ha='center', va='bottom', color=col, fontproperties=fontprob_label)
            ax.text((x1 + x2) * .5, y - h * 0.1, sig_str, ha='center', va='bottom', color=col,
                    fontproperties=fontprob_label_y)

    # Only show ticks on the left and bottom spines
    # ax.spines.right.set_visible(False)
    # ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # plt.locator_params(axis='y', nbins=5)
    
    plt.tight_layout()
    
    ax.set_xlim(-0.5,3.5)
    plt.savefig(figfile, dpi=300)
    plt.close()
    
    # plt.show()

    return ax, fontprob_label_y

# # Customize x-axis tick labels with different colors
def plot_violin_v7(figfile, ax, df,data, x_variable, y_variable, df_stats, p_variable, order, x_ticklabels,label_colors,
                   title=None, x_label='', y_label='', fontsize_title=10, fontsize_label_x=10,fontsize_label_y=10, palette=None,
                   y_lim=None, y_ticks=None, y_ticklabels=None, rotation=90):
    """
    this is for plotting the confusion matrix, only including the top N networks
    :param df: is the dataframe for plot: columns:x_variable,y_variable
    :param x_variable: the xlabel
    :param y_variable: y values
    :param fig_size:
    :param y_label:
    :param order: the order of x_variable
    :param palette:
    :param title:
    :param figfile:
    :return:
    """
    fontprob_label_x = fm.FontProperties(fname=font_file, size=fontsize_label_x)
    fontprob_label_y = fm.FontProperties(fname=font_file, size=fontsize_label_y)
    # fontprob_label_s=fm.FontProperties(fname=font_file, size=fontsize_label-1)
    fontprob_label_s = fm.FontProperties(fname=font_file, size=fontsize_label_x)
    # plot
    sns.violinplot(ax=ax, x=x_variable, y=y_variable, data=df, order=order, linewidth=0.8, palette=palette, cut=2)
    sns.stripplot(ax=ax, x=x_variable, y=y_variable, data=df, order=order, palette=palette, jitter=True, dodge=False,
                  size=4, edgecolor='k', linewidth=0.5)

    # Remove upper and right frame lines
    sns.despine(top=True, right=True)

    # Show the mean of each column using white large circles
    for i, column in enumerate(order):
        mean_value = data[column].mean()
        ax.scatter(i, mean_value,  color='white', s=200, zorder=3, marker='o', edgecolor='black',facecolors='none')

    # labeling
    # ax.set_xticklabels(order,fontproperties=fontprob_label,rotation=rotation)
    ax.set_xticklabels(x_ticklabels, fontproperties=fontprob_label_x, rotation=rotation)
    plt.setp(ax.get_yticklabels(), fontproperties=fontprob_label_y)


    # labeling
    ax.set_xlabel(x_label, fontproperties=fontprob_label_x)
    ax.set_ylabel(y_label, fontproperties=fontprob_label_y, x=-0.2)

    # Customize x-axis tick labels with different colors
    for label in ax.get_xticklabels():
        label.set_color(label_colors[label.get_text()])

    if title is not None:
        ax.set_title(title, fontsize=fontsize_title, y=1.05)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if y_ticklabels is not None:
        ax.set_yticklabels(y_ticklabels)

    y_min, y_max = ax.get_ylim()


    # add the star statistical annotation
    line_width = 1
    y, h, col = y_max, (y_max - y_min) * 0.02, 'k'
    for index, row in df_stats.iterrows():
        x1, x2 = order.index(row['N1']) + 0.05, order.index(row['N2']) - 0.05  # x coordinates of two networks
        sig_str = star_sig(row[p_variable])

        ax.plot([x1, x1, x2, x2], [y+1 - h, y+1, y+1, y+1 - h], lw=line_width, c=col)
        if sig_str == 'ns':
            # baihan ori value
            # ax.text((x1+x2)*.5, y-h*0.4, sig_str, ha='center', va='bottom', color=col, fontproperties=fontprob_label_s)
            ax.text((x1 + x2) * .5, y - h * 0.01, sig_str, ha='center', va='bottom', color=col,
                    fontproperties=fontprob_label_s)
        else:
            # baihan ori value
            # ax.text((x1+x2)*.5, y-h*1.5, sig_str, ha='center', va='bottom', color=col, fontproperties=fontprob_label)
            ax.text((x1 + x2) * .5, y+1 - h * 0.1, sig_str, ha='center', va='bottom', color=col,
                    fontproperties=fontprob_label_y)

    # Only show ticks on the left and bottom spines
    # ax.spines.right.set_visible(False)
    # ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # plt.locator_params(axis='y', nbins=5)
    
    # ax.set_ylim(-3, 5)
    # ax.set_xlim(-0.5, 1.5)
    
    # plt.show()

    # Adjust figure layout to prevent clipping of axis labels
    plt.tight_layout()
    
    plt.savefig(figfile, dpi=300)
    plt.close()

    return ax, fontprob_label_y