from PIL import Image, ImageFont, ImageDraw
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wbplot.utils.plots import check_vrange, check_cmap_plt
import os
from os.path import join


current_directory = os.getcwd()
font_file = os.path.join(current_directory, '../Templates/Atlas/Font/arial.ttf')
 
titlefont = ImageFont.truetype(font=font_file,size=100)
fontprop = fm.FontProperties(fname=font_file, size=52)
font = ImageFont.truetype(font=font_file,size=40)

def make_transparent(img_file):
    """
    Make each white pixel in an image transparent.

    Parameters
    ----------
    img_file : str
        absolute path to a PNG image file_hctsa

    Returns
    -------
    None

    Notes
    -----
    This function overwrites the existing file_hctsa.

    """

    img = Image.open(img_file)
    img = img.convert("RGBA")
    pixdata = img.load()
    width, height = img.size
    for y in range(height):
        for x in range(width):
            if pixdata[x, y] == (0, 0, 0, 255):  # if black
                pixdata[x, y] = (255, 255, 255, 255)  # set alpha = 255
    img.save(img_file, "PNG")

def add_title(img_file,title,title_position=300):
    """
    Add the title of the image: the contrast

    Parameters
    ----------
    img_file : str absolute path to a PNG image file_hctsa
    title: the title being shown on the figure
    Returns
    -------
    None

    Notes
    -----
    This function overwrites the existing file_hctsa.

    """
    img = Image.open(img_file)
    image_editable = ImageDraw.Draw(img)
    image_editable.text((title_position, 15), title, (0,0,0), font=titlefont)
    img.save(img_file, "PNG", dpi=(300.0, 300.0))

def add_title_middle(img_file,title):
    """
    Add the title of the image: the contrast

    Parameters
    ----------
    img_file : str absolute path to a PNG image file_hctsa
    title: the title being shown on the figure
    Returns
    -------
    None

    Notes
    -----
    This function overwrites the existing file_hctsa.

    """
    img = Image.open(img_file)
    image_editable = ImageDraw.Draw(img)
    image_editable.text((700, 15), title, (0,0,0), font=titlefont)
    img.save(img_file, "PNG")

def add_title_little(img_file,title):
    """
    Add the title of the image: the contrast

    Parameters
    ----------
    img_file : str absolute path to a PNG image file_hctsa
    title: the title being shown on the figure
    Returns
    -------
    None

    Notes
    -----
    This function overwrites the existing file_hctsa.

    """
    font = ImageFont.truetype(font=font_file, size=30)
    img = Image.open(img_file)
    image_editable = ImageDraw.Draw(img)
    image_editable.text((20, 15), title, (0,0,0), font=titlefont)
    img.save(img_file, "PNG", dpi=(300.0, 300.0))


# add color bar11
def add_color_bar(img_file,vmin_neg_min, vmin_neg_max,vmin_pos_min,vmin_pos_max,color='psych'):
    """
    add a color bar for the image_file
    :param img_file:
    :param vmin: the minimum value for the color bar
    :param vmax: the maximum value for the color bar
    :return: None
    This function overwrites the existing file_hctsa.
    """
    if color =='psych':
        file_color_bar = os.path.join(current_directory, '../templates/atlas/colorbar/color_bar_psych_fixed.png')
    elif color == 'spectral':
        file_color_bar = os.path.join(current_directory, '../templates/atlas/colorbar/color_bar_spectral.png')

    # open color bar
    img_colorbar = Image.open(file_color_bar)

    # open the original figure
    img = Image.open(img_file)
    # Resize the image to fit the new figsize
    img_colorbar = img_colorbar.resize((600, 100))
    # add the color bar to the original figure
    img.paste(img_colorbar,(180,570))

    # add the ticks
    tick_height = 585
    vmin_neg_mid = np.round((vmin_neg_min+vmin_neg_max)/2,2)
    vmin_pos_mid = np.round((vmin_pos_min+vmin_pos_max)/2,2)
    image_editable = ImageDraw.Draw(img)
    image_editable.text((180, tick_height), str(vmin_neg_min), (0, 0, 0), font=font)
    image_editable.text((380, tick_height), str(vmin_neg_mid), (0, 0, 0), font=font)
    image_editable.text((550, tick_height), str(vmin_neg_max), (0, 0, 0), font=font)
    image_editable.text((660, tick_height), str(vmin_pos_min), (0, 0, 0), font=font)
    image_editable.text((840, tick_height), str(vmin_pos_mid), (0, 0, 0), font=font)
    image_editable.text((1060,tick_height), str(vmin_pos_max), (0, 0, 0), font=font)

    # save it
    img.save(img_file)



# add color bar11
def add_colorbar2(data, img_file, cmap, vrange=None, orientation='horizontal'):
    """
    add a color bar for the image_file
    :param img_file:
    :param vmin: the minimum value for the color bar
    :param vmax: the maximum value for the color bar
    :return: None
    This function overwrites the existing file_hctsa.
    """
    import uuid
    
    # open the original figure
    img = Image.open(img_file) 
    img_x,img_y = img.size
    
    # draw a colorbar and open it
    colorbar_id = uuid.uuid1().hex
    fig_dir = os.path.dirname(img_file)
    file_colorbar = join(fig_dir, 'colorbar_%s.png'%(colorbar_id))
  
    draw_colorbar(data, figsize=(9.2, 1.45), file_colorbar=file_colorbar, cmap = cmap, vrange=vrange, orientation=orientation)
    
    img_colorbar = Image.open(file_colorbar)
    img_colorbar_x, img_colorbar_y = img_colorbar.size

    # add the color bar to the original figure
    img.paste(img_colorbar,(np.int16(img_x/2-img_colorbar_x/2), np.int16(img_y/2-img_colorbar_y/2)) )

    # save it
    img.save(img_file, dpi=(300.0, 300.0))

    os.system('rm -v %s'%(file_colorbar))

def draw_colorbar(data, figsize, file_colorbar, cmap, vrange=None, orientation='horizontal'):
       
    num_decimal = 2

    # divide it to positive and negative values
    data_neg = data[data < 0]
    data_pos = data[data > 0]

    # find the positive values and negative values (extreme)
    # these values would be used when adding color bar
    if data_pos.shape[0] != 0:
        vmin_pos_max = np.round(data_pos.max(), num_decimal)
        vmin_pos_min = np.round(data_pos.min(), num_decimal)
    else:
        vmin_pos_max = 0
        vmin_pos_min = 0
    if data_neg.shape[0] != 0:
        vmin_neg_max = np.round(data_neg.max(), num_decimal)
        vmin_neg_min = np.round(data_neg.min(), num_decimal)
    else:
        vmin_neg_max = 0
        vmin_neg_min = 0
    
    if vrange==None:
        ticks = [np.round(vmin_neg_min, 1), np.round(vmin_pos_max, 1)]
        label_mid = [vmin_neg_min, 0, vmin_pos_max]
        norm = mpl.colors.Normalize(vmin=ticks[0], vmax=ticks[1], clip=True)
    else:
        ticks = [vrange[0], vrange[1]]
        label_mid = [vrange[0], 0, vrange[1]]
        norm = mpl.colors.Normalize(vmin=vrange[0], vmax=vrange[1], clip=True)

    #draw a custom colorbar and save a temporary picture
    fig = plt.figure(constrained_layout=True,figsize=figsize)
    #plt.rcParams.update({'font.sans-serif':'Arial'})
    ax = fig.subplots(1,1)
    

    # fcb = mpl.colorbar.ColorbarBase(norm=norm, cmap=mpl.cm.get_cmap(cmap), orientation=orientation,ax=ax ,ticks = ticks )
    fcb = mpl.colorbar.ColorbarBase(norm=norm, cmap=cmap, orientation=orientation,ax=ax ,ticks = ticks )
    fcb.ax.set_xticklabels(ticks, fontproperties=fontprop)

    
    # Get the associated axis object
    colorbar_axis = fcb.ax
    
        # Get the x-limits of the figure
    x_limits = ax.get_xlim()

    # Calculate the middle horizontal position
    left_pos = x_limits[0]
    middle_pos = (x_limits[0] + x_limits[1]) / 2
    right_pos = x_limits[1]
    pos_mid = [left_pos, middle_pos, right_pos]
    
    # Set the tick positions and labels
    colorbar_axis.set_xticks(pos_mid)
    colorbar_axis.set_xticklabels(label_mid, fontproperties=fontprop)
    
    # ax2 = fig.subplots(1,2)
    # norm_neg = mpl.colors.Normalize(vmin=np.nanmin(data_neg), vmax=np.nanmax(data_neg), clip=True)
    # # fcb = mpl.colorbar.ColorbarBase(norm=norm, cmap=mpl.cm.get_cmap(cmap), orientation=orientation,ax=ax ,ticks = ticks )
    # fcb_neg = mpl.colorbar.ColorbarBase(norm=norm_neg, cmap=cmap_neg, orientation=orientation,ax=axes[1] ,ticks = ticks_neg )
    # fcb_neg.ax.set_xticklabels(ticks_neg, fontproperties=fontprop)

    # plt.show()
    #save
    fig.savefig(file_colorbar, dpi=300,transparent=True)
    plt.close()