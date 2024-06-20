import numpy as np
import os
from os import system
from function_edit_scene_figures_use_fig_color import add_title, add_color_bar, add_colorbar2
import nibabel as nib
import uuid
from os.path import join, exists
from wbplot_images_use_fig_color import write_parcellated_image
import nibabel as nib
from zipfile import ZipFile
from nibabel.cifti2.parse_cifti2 import Cifti2Parser
from PIL import Image
import matplotlib.colors as colors



def plot_Kong_parcellation_pos_neg(data, fig, color, title, title_position = 180, vrange = None, scale=None):
    """
    :param array: is 400 values of the network, shape is (400,1)
    :param file_output: the name of png
    :return:
    """

    if type(data) is np.ndarray:
        pscalar = data
    elif isinstance(data, str):
        d = nib.load(data).get_fdata()
        pscalar = d[0]
    else:
        raise ValueError('data must be a data array or a cifti file ')
    
    if title is None:
        title = os.path.basename(fig).split('.')[0]
        
    # [CHANGE]
    current_directory = os.getcwd()
    path_base = os.path.join(current_directory, '../Templates/Scene/Scene_Kong_dlabel')
    path_wb = '/home/publicapps/workbench/bin_rh_linux64/wb_command'
    
     # The original scene file you created
    scene_zip_file=join(path_base,'Schaefer_417_dlabel.zip')
    filename_scene = 'Schaefer_417_dlabel.scene'
    temp_name = 'ImageParcellated.dlabel.nii'
    
    fig_id = uuid.uuid1().hex
    fig_dir = os.path.dirname(fig)
    temp_dir = os.path.join(fig_dir,'temp_scene_'+fig_id)
    os.mkdir(temp_dir)
    
    # copy the scene file & SchaeferParcellations directory to the
    # temp directory as well
    with ZipFile(scene_zip_file, "r") as z:  # unzip to temp dir
        z.extractall(temp_dir)
    scene_file = join(temp_dir, 'scene_Kong_dlabel', filename_scene)
    if not exists(scene_file):
        raise RuntimeError(
            "scene file was not successfully copied to {}".format(scene_file))
        

    
    #prepare cmap
    if color =='psych':
        file_color_bar = os.path.join(current_directory, '../Templates/Atlas/Colorbar/color_bar_psych_fixed.png')
    elif color == 'spectral':
        file_color_bar = os.path.join(current_directory, '../Templates/Atlas/Colorbar/color_bar_spectral.png')
    
    # Open the image
    image = Image.open(file_color_bar) 
    
    # Convert the image to RGB mode 
    image = image.convert('RGB')

    # Get the pixel data as a NumPy array
    pixel_array = np.array(image)

    # Extract the first line pixels' RGB values
    rgb_values = pixel_array[0]
    
    # Normalize the RGB values to the range [0, 1]
    normalized_rgb = rgb_values / 255.0
    normalized_rgb_neg = normalized_rgb[0:int(len(normalized_rgb)/2)]
    normalized_rgb_pos = normalized_rgb[int(len(normalized_rgb)/2)+2: len(normalized_rgb)]
    # Create a custom colormap using the RGB values
    cmap_neg = colors.ListedColormap(normalized_rgb_neg)
    cmap_pos = colors.ListedColormap(normalized_rgb_pos)
    
    cmap = colors.ListedColormap(normalized_rgb)
    
    # Write `pscalars` to the neuroimaging file which is pre-loaded into the
    # scene file, and update the colors for each parcel using the file metadata
    temp_cifti1 = join(temp_dir, temp_name)
    temp_cifti2 = join(temp_dir,'scene_Kong_dlabel',temp_name)
    
    if scale != None:
        vrange = scale
        write_parcellated_image(data=pscalar.ravel(), fout=temp_cifti1,  cmap_neg=cmap_neg, cmap_pos = cmap_pos, vrange=scale)
    else:
        write_parcellated_image(data=pscalar.ravel(), fout=temp_cifti1,  cmap_neg=cmap_neg, cmap_pos = cmap_pos, vrange=vrange)
    
    #overwrite the original files
    cmd='cp -vf %s %s'%(temp_cifti1,temp_cifti2)
    system(cmd)
    
    # copy scene file
    scene=1
    width = 3600
    height = 3600   # height = 835
    cmd = path_wb + ' -show-scene "{}" {} "{}" {} {}'.format(scene_file, scene, fig, width, height)
    system(cmd)
    
    # remove the copied scene folder
    cmd = 'rm -rf %s'%(temp_dir)
    system(cmd)

    # add title and colorbar
    add_colorbar2(pscalar, fig, cmap = cmap, vrange = vrange, orientation='horizontal')
    add_title(fig, title, title_position)