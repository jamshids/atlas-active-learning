import numpy as np
from os import listdir
from os.path import isfile, join
import nibabel as nib
import pdb

def enigma_releases(path, id_loc):
    """Loading data released by ENIGMA challenge 
    
    The data include both raw images and segmentation labels.
    """
    
    # extracting all the files existing in the specified path
    onlyfiles = [f for f in listdir(path) 
                 if isfile(join(path, f))]
    # the first 4 letters of the file-names are numeric identifiers
    first_letters = [name[id_loc[0]:id_loc[1]] for name in onlyfiles]
    offset = onlyfiles[0][:id_loc[0]]
    # since each identifier has multiple file formats, 
    # we only take the unique values
    unique_digits = np.unique(np.array([int(digit) for digit
                                        in first_letters])).tolist()
    for i, digits in enumerate(unique_digits):
        num_digits = len(str(unique_digits[i]))
        if num_digits<4:
            unique_digits[i] = (4-num_digits)*'0'+str(unique_digits[i])
        else:
            unique_digits[i] = str(unique_digits[i])
        
    
    '''Now, read the files with the extracted names
    '''
    data_list = list()
    labels_list = list()
    #pdb.set_trace()
    for image_id in unique_digits:
        
        # loading raw images
        fname = image_id + "_mprage_deface.nii"
        try:
            img = nib.load(path + offset +  fname)
            data_list += [img.get_data()]
        except:
            print("Image with ID "+ image_id +" could not be found/read")
        
        
        # loading the labels
        fname = image_id + "_mprage_deface.nii"
        try:
            labels = nib.load(path + offset + fname)
            labels_list += [labels.get_data()]
        except:
            print("Mask with ID " + image_id + " could not be found/read")
            
    return data_list, labels_list

