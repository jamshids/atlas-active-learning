import numpy as np
from os import listdir
from os.path import isfile, join
import nibabel as nib
import copy
import pdb

def enigma_releases(path, labels_flag=False):
    """Loading data released by ENIGMA challenge 
    
    File names  are specified manually within the function below
    """
    
    # reading files from M-Release
    # ---------------------------
    file_ids = ['0935', '2261', '2381', '2386', '2489',
                '2501', '2526', '2617', '2621', '2659', 
                '2719', '2734', '2821', '2827', '2828',
                '2839', '2846', '2855', '2869', '2873']
    file_comp = '_mprage_deface'     
    
    # reading the files one-by-one
    M_data = list()
    M_labels = list()
    M_path = path + 'M_Release/'
    
    for ids in file_ids:
        
        # loading raw images
        fname = ids + file_comp
        try:
            img = nib.load(M_path + fname)
            M_data += [copy.deepcopy(img.get_data())]
        except:
            print("File " + M_path + fname + " could not be found/read")
        
        
        # loading the labels, if necessary
        if labels_flag:
            fname = image_id + "_manual_cerebellum.nii"
            try:
                labels = nib.load(M_path + fname)
                M_labels += [copy.deepcopy(labels.get_data())]
            except:
                print("File " + M_path + fname + " could not be found/read")
            
    # reading files from T-Release
    # ---------------------------
    # T-release has two parts: those that are labeled by experts (with one 
    # segmentation) and those that are labeled by two inexperts (wih two
    # segmentations).
    
    # Expert-segmented images            
    file_ids = ['at1000', 'at1006', 'at1007', 'at1017', 'at1021',
                'at1025', 'at1029', 'at1031', 'at1033', 'at1034',
                'at1040', 'at1044', 'at1048' 'at1049', 'at1084']
    file_comp = '_mprage_deface'
    
    # reading the files one-by-one
    T_exp_data = list()
    T_exp_labels = list()
    T_path = path + 'T_Release/'
    
    for ids in file_ids:
        
        # loading raw images
        fname = ids + file_comp
        try:
            img = nib.load(path + fname)
            T_exp_data += [copy.deepcopy(img.get_data())]
        except:
            print("File " + path + fname + " could not be found/read")
        
        
        # loading the labels, if necessary
        if labels_flag:
            fname = image_id + "_manual_cerebellum.nii"
            try:
                labels = nib.load(path + fname)
                T_exp_labels += [copy.deepcopy(labels.get_data())]
            except:
                print("File " + path + fname + " could not be found/read")
                
    # Inexpert-segmented images
    file_ids = ['at1005', 'at1013', 'at1014', 'at1016', 'at1018',
                'at1023', 'at1024', 'at1027', 'at1028', 'at1032',
                'at1036', 'at1043', 'at1045', 'at1060', 'at1079']
    file_comp = '_mprage_deface'
    
    # reading the files one-by-one
    T_inexp_data = list()
    T_inexp_labels_A = list()
    T_inexp_labels_B = list()
    
    for ids in file_ids:
        
        # loading raw images
        fname = ids + file_comp
        try:
            img = nib.load(path + fname)
            T_inexp_data += [copy.deepcopy(img.get_data())]
        except:
            print("File " + path + fname + " could not be found/read")
        
        
        # loading the labels, if necessary
        if labels_flag:
            fname_A = image_id + "_inexpert_A_cerebellum.nii"
            try:
                labels = nib.load(path + fname)
                T_inexp_labels_A += [copy.deepcopy(labels.get_data())]
            except:
                print("File " + path + fname + " could not be found/read")
            
            fname_B = image_id + "_inexpert_B_cerebellum.nii"
            try:
                labels = nib.load(path + fname)
                T_inexp_labels_B += [copy.deepcopy(labels.get_data())]
            except:
                print("File " + path + fname + " could not be found/read")
            
    return (M_data, T_exp_dat, T_inexp_dat, M_labels, 
            T_exp_labels, T_inexp_labels_A, T_inexp_labels_B)

