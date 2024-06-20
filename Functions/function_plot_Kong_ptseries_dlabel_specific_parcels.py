# plot the Kong map

# the raw input is 400 values

# the output is a figure
import uuid
import numpy as np
import os
from os import system
from os.path import join, exists
from function_edit_scene_figures import add_title, add_colorbar2, draw_colorbar
from wbplot_images import write_parcellated_image
import nibabel as nib
from zipfile import ZipFile
import matplotlib.colors as colors

num_decimal = 2
path_wb = 'wb_command'


# [CHANNGE]
current_directory = os.getcwd()
path_template = os.path.join(current_directory, '../Templates/Atlas/Parcellation_Kong_2022')
path_scene = os.path.join(current_directory, '../Templates/Scene/Scene_Kong_dlabel')
path_wb = '/home/publicapps/workbench/bin_rh_linux64/wb_command'
path_output_scene = os.path.join(current_directory, '../Results')
os.makedirs(path_output_scene, exist_ok = True)


def save_ptseries_file(array, file_map):
    """
    generate pconn file_hctsa
    :param array: 400, shape (1,400)
    :param file_map: ptseries.nii file_hctsa
    :return:
    """
    import nibabel as nib
    import os

    if array.shape==(400,):
        array=np.reshape(array,(1,400))
    # step 2: read the excel file_hctsa and then save the data
    cifti_template_file = os.path.join(path_template,'template.ptseries.nii')
    cifti_template = nib.load(cifti_template_file)

    new_img = nib.Cifti2Image(array, header=cifti_template.header, nifti_header=cifti_template.nifti_header)
    new_img.to_filename(file_map)



def save_dlabel_Kong (array,file_output):
    """

    :param array: is a list or array that includes that network IDs that to be plotted
    :param file_output: is the dlabel.nii file_hctsa, networks that need to be plotted are the network ID, others are np.nan
    :return:
    """
    import nibabel as nib
    from os.path import  join
    import numpy as np

    filename = 'Schaefer2018_400Parcels_Kong2022_17Networks_order.dlabel.nii'

    file = join(path_template,filename)

    data = nib.load(file).get_fdata()

    # replace the value, only keep the values in the array
    data_new = np.where(np.in1d(data,array), data, np.nan)

    cifti_template = nib.load(file)
    new_img = nib.Cifti2Image(data_new, header=cifti_template.header, nifti_header=cifti_template.nifti_header)
    new_img.to_filename(file_output)


def plot_Kong_parcellation(data, figfile, cmap= 'nipy_spectral', vrange = None, title=None, title_position=180):
    """
    :param data: is 400 values of the network, shape is (400,1)
    :param figfile: the output figure filename ending with '.png'
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
        title = os.path.basename(figfile).split('.')[0]
    
    
     # The original scene file you created 
    scene_zip_file=join(path_scene,'Schaefer_417_dlabel.zip')
    filename_scene = 'Schaefer_417_dlabel.scene'
    temp_name = 'ImageParcellated.dlabel.nii'
    
    #generate temporary working folder in output figure directory
    fig_id = uuid.uuid1().hex
    fig_dir = os.path.dirname(figfile)
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
    
    if cmap == 'FM':
        #color FM overlap Spatial or Math
        cmap = colors.ListedColormap(['#e8e830','#006400','#940118'])
    elif cmap =='Asso':
        #color Asso overlap Spatial or Math
        cmap = colors.ListedColormap(['#FF4500','#098b8b','#940118'])
    elif cmap == 'non-semantic':
        cmap = colors.ListedColormap(['#854daf'])
    elif cmap == 'FM_Asso_conjunction':
        #color overlap FM and Asso
        cmap = colors.ListedColormap(['#00008e','#0078f8','#86cbbe','#b6b6b6', '#f8f801', '#ff4800', '#940118' ])
    elif cmap == 'FPCNA_DMN':
        cmap = colors.ListedColormap(['#fe9900','#ce585e' ])
    elif cmap == 'Asso_and_FM':
        cmap = colors.ListedColormap(['#02195a','#2f79be','#32c0c6', '#b6b6b6', '#e8e830', '#ea4325', '#8f2738' ])

    # Write `pscalars` to the neuroimaging file which is pre-loaded into the
    # scene file, and update the colors for each parcel using the file metadata
    temp_cifti1 = join(temp_dir, temp_name)
    temp_cifti2 = join(temp_dir,'scene_Kong_dlabel',temp_name)
    write_parcellated_image(data=pscalar, fout=temp_cifti1,  cmap=cmap, vrange = vrange)

    #overwrite the original files
    cmd='cp -f %s %s'%(temp_cifti1,temp_cifti2)
    system(cmd)
    
    # copy scene file
    scene=1
    width = 2400
    height = 2400   # height = 835
    cmd = path_wb + ' -show-scene "{}" {} "{}" {} {}'.format(scene_file, scene, figfile, width, height)
    system(cmd)
    
    # remove the copied scene folder
    cmd = 'rm -rf %s'%(temp_dir)
    system(cmd)

    # add title and colorbar
    # add_colorbar2(pscalar, figfile, cmap, vrange, orientation='horizontal')
    if not title is None:
        add_title(figfile, title, title_position)



# def plot_Kong_dlabel(array, fig, title, title_position,title_position_vertical):
def plot_Kong_dlabel(array, fig):
    """
    :param array: is the network id need to be plotted [1,3,5]
    :param file_output: the name of png
    :return:
    """
    
    # The original scene file you created
    filename_scene = 'Schaefer_417_dlabel.scene'
    folder_scene = 'Schaefer_417_inflated'

    # the original ptseries file you created
    filename = 'ImageParcellated.dlabel.nii'
    file_map = os.path.join(path_output_scene, filename)

    file_scene_1 = os.path.join(path_scene, filename_scene)
    file_scene_folder = os.path.join(path_scene, folder_scene)


    # check whether scene file exists, if not, copy them
    if not os.path.exists(os.path.join(path_output_scene,folder_scene )):
        cmd_1 = "cp -r {} {}".format(file_scene_folder, path_output_scene)
        system(cmd_1)

    if not os.path.exists(os.path.join(path_output_scene,filename_scene)):
        cmd_2 = "cp  {} {}".format(file_scene_1, path_output_scene)
        system(cmd_2)

    # save the ptseries file
    save_dlabel_Kong(array, file_map)

    scene_file = os.path.join(path_output_scene,filename_scene)
    scene=1
    width = 1263
    # height = 835
    height = 1100

    # copy scene file
    cmd = path_wb + ' -show-scene "{}" {} "{}" {} {}'.format(scene_file, scene,  fig, width, height)

    system(cmd)

    # add title
    # add_title(fig, title,title_position,title_position_vertical)


