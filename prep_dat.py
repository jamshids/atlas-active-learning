import numpy as np
from os import listdir
from os.path import isfile, join
import nibabel as nib
import copy
import pdb

class enigma:
    """Class of functions for reading data in ENIGMA callenge
    """
    def __init__(self, path):
        """Set the path of directory that contains the data
        """
        self.path = path
    
    def enigma_releases(self, labels_flag=False):
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
        M_path = self.path + 'M_Release/'

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
                img = nib.load(self.path + fname)
                T_exp_data += [copy.deepcopy(img.get_data())]
            except:
                print("File " + self.path + fname + " could not be found/read")


            # loading the labels, if necessary
            if labels_flag:
                fname = image_id + "_manual_cerebellum.nii"
                try:
                    labels = nib.load(self.path + fname)
                    T_exp_labels += [copy.deepcopy(labels.get_data())]
                except:
                    print("File " + self.path + fname + " could not be found/read")

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
                img = nib.load(self.path + fname)
                T_inexp_data += [copy.deepcopy(img.get_data())]
            except:
                print("File " + self.path + fname + " could not be found/read")


            # loading the labels, if necessary
            if labels_flag:
                fname_A = image_id + "_inexpert_A_cerebellum.nii"
                try:
                    labels = nib.load(self.path + fname)
                    T_inexp_labels_A += [copy.deepcopy(labels.get_data())]
                except:
                    print("File " + self.path + fname + " could not be found/read")

                fname_B = image_id + "_inexpert_B_cerebellum.nii"
                try:
                    labels = nib.load(self.path + fname)
                    T_inexp_labels_B += [copy.deepcopy(labels.get_data())]
                except:
                    print("File " + self.path + fname + " could not be found/read")

        return (M_data, T_exp_dat, T_inexp_dat, M_labels, 
                T_exp_labels, T_inexp_labels_A, T_inexp_labels_B)


class MSSEG:
    """Class of functions for reading data of MS segmentatin challenge
    """
    
    def __init__(self, path):
        """Path of the data directory
        """
        
        self.path = path
        self.IDs = ['01016SACH', '01038PAGU', '01039VITE', '01040VANE',
                    '01042GULE', '07001MOEL', '07003SATH', '07010NABO',
                    '07040DORE', '07043SEME', '08002CHJE', '08027SYBR',
                    '08029IVDI', '08031SEVE', '08037ROGU']

    def pre_processed_dat(self, patient, modals=['FLAIR', 'T1', 'T2']):
        """Reading three-preprocessed file from a given patient: FLAIR,
        T1-weighted and T2-weighted images
        """
        
        if patient not in self.IDs:
            raise ValueError("Patient ID is not present in the data..")
        
        preprop_path = self.path + "Pre-processed/" + patient + "/"
        
        # reading data
        if 'FLAIR' in modals:
            flair = nib.load(preprop_path+"FLAIR_preprocessed.nii.gz")
        if 'T1' in modals:
            T1 = nib.load(preprop_path+"T1_preprocessed.nii.gz")
        if 'T2' in modals:
            T2 = nib.load(preprop_path+"T2_preprocessed.nii.gz")
            
        return flair.get_data(), T1.get_data(), T2.get_data()
    
    def pre_processed_seg(self, patient, seg_id=None):
        """Reading segmentation labels of the pre-processed data
        
        If no segmentor ID is not given, the consensus segmentation
        will be read
        """
        
        if patient not in self.IDs:
            raise ValueError("Patient ID is not present in the data..")
        
        seg_path = self.path + "ManualSegmentation/" + patient + "/"
        
        if seg_id:
            seg = nib.load(seg_path+"ManualSegmentation_%d.nii.gz" % seg_id)
        else:
            seg = nib.load(seg_path+"Consensus.nii.gz")
        seg = seg.get_data()
            
        return seg
    
    def pre_processed_mask(self, patient):
        """Reading brain mask of a given patient
        """
        
        if patient not in self.IDs:
            raise ValueError("Patient ID is not present in the data..")
        
        mask_path = self.path + "Pre-processed/" + patient + "/Mask_registered.nii.gz"
        mask = nib.load(mask_path)
        
        return mask.get_data()
    
    
    def extract_patch_features(patient, save_path, patch_size=[5,5,5]):
        """Exracting patch-wise feature vectors for all the voxels that are
        should be consiered according to mask (for example they are part of the brain)

        "modals" is a list of dictionary, where each dictionary contains images of a 
        specific modality for all the patients. "mask_dict" and "seg_dict" are single 
        dictionaries that has masks for all the patients. There are two assumptions:

        (1) Keys of all the dictionaries should be the same (same patients)
        (2) Image dimensionality of all the modalities plus masks should be the same
        """

        # first, read all the data
        # pre-processed images
        flair, T1, T2 = MSSEG_reader.pre_processed_dat(patient)
        # segmentations for preprocessed images
        seg = np.int16(MSSEG_reader.pre_processed_seg(patient))
        # mask for pre-processed images
        mask = np.int16(MSSEG_reader.pre_processed_mask(patient))

        # then, extract and all the vexels' feature vectors
        patch_rad = np.int16(np.array(patch_size) / 2)
        # reading only the unmasked voxels
        unmasked_vox = np.where(mask>0)
        # feature vectors, and labels of this image:
        # last row will be the labels
        X = np.zeros((3*np.prod(patch_size)+1, len(unmasked_vox[0])))
        labels = np.zeros(len(unmasked_vox[0]))
        with open(save_path, "w") as f:
            for j in range(len(unmasked_vox[0])):
                (x,y,z) = (unmasked_vox[0][j], unmasked_vox[1][j], unmasked_vox[2][j])
                # extract vicinity patch of the voxel in all the modalities an
                # store them in an array
                patch = flair[x-patch_rad[0]:x+patch_rad[0],
                              y-patch_rad[1]:y+patch_rad[1],
                              z-patch_rad[2]:z+patch_rad[2]]
                patch = np.reshape(patch, np.prod(patch.shape))
                X[:len(patch), j] = patch
                patch = T1[x-patch_rad[0]:x+patch_rad[0],
                              y-patch_rad[1]:y+patch_rad[1],
                              z-patch_rad[2]:z+patch_rad[2]]
                patch = np.reshape(patch, np.prod(patch.shape))
                X[len(patch):2*len(patch), j] = patch
                patch = T2[x-patch_rad[0]:x+patch_rad[0],
                           y-patch_rad[1]:y+patch_rad[1],
                           z-patch_rad[2]:z+patch_rad[2]]
                patch = np.reshape(patch, np.prod(patch.shape))
                X[2*len(patch):, j] = patch
                # store label of the voxel
                labels[j] = seg[x,y,z]
                # saving these data into the files
                f.write("".join(" ".join(map(str, x)) for x in (X[:,j],))+'\n')

        return X, labels
    
class Mindboggle_101():
    """Class of functions for reading data images of data set
    Mindboggle-101
    """
    
    def __init__(self, base_dir):
        """Take the directory address that contains the data set's 
        folders, and read in all the images from each of the five
        different groups
        
        We assume that data of each subject is stored in a different
        folder but with the same name given as an input argument.
        """
        
        self.base_dir = base_dir
        self.group_names = [
            'Extra-18_volumes_in_MNI152',
            'MMRR-21_volumes_in_MNI152',
            'NKI-RS-22_volumes_in_MNI152',
            'NKI-TRT-20_volumes_in_MNI152',
            'OASIS-TRT-20_volumes_in_MNI152']
        
        self.data_fname = 't1weighted_brain.MNI152.nii.gz'
        self.DKT25_fname = 'labels.DKT25.manual.MNI152.nii.gz'
        self.DKT31_fname = 'labels.DKT31.manual.MNI152.nii.gz'
        
        # read the data from all the groups
        self.img_num = 101
        self.img_shape = (182, 218, 182)

    def read_images(self):
        """Reading brain images
        """
        
        #imgs = np.zeros((self.img_num,)+self.img_shape)
        imgs = {group:[] for group in self.group_names}
        
        for i in range(len(self.group_names)):
            dirs = extract_dirs(self.base_dir)
            group_size = len(dirs)
            group_imgs = np.zeros(
                (group_size,)+self.img_shape)
            for j in range(len(dirs)):
                # read brain images
                fpath = '%s/%s/%s'% (
                    self.base_dir,
                    self.group_names[j],
                    self.data_fname)
                
                img = nib.load(fpath)
                group_imgs[j,:] = img.get_data()
                
            imgs[self.group_names[i]] = [group_imgs]
            
        return imgs
                
    def read_labels(self, label_type, imgs):
        """Loading labels of the images into the data
        dictionary
        
        This method should be called after loading the 
        images. Type of the labels could be either 
        '25' (for DKT-25), or '31' (for DKT-31) or both 
        """
        
        # --------------------------------------------
        # if DKT-25 labels are to be loaded
        if '25' in label_type:
            for i in range(len(self.group_names)):
                dirs = extract_dirs(self.base_dir)
                group_size = len(dirs)
                group_masks = np.zeros(
                    (group_size,)+self.img_shape)
                for j in range(len(dirs)):
                    # read brain images
                    fpath = '%s/%s/%s'% (
                        self.base_dir,
                        self.group_names[j],
                        self.DKT25_fname)
                    
                    mask = nib.load(fpath)
                    group_masks[j,:,:,:] = mask.get_data()
                    
                imgs[self.group_names[i]] += [group_masks]
        
        # --------------------------------------------
        # if DKT-31 labels are to be loaded
        if '31' in label_type:
            for i in range(len(self.group_names)):
                dirs = extract_dirs(self.base_dir)
                group_size = len(dirs)
                group_masks = np.zeros(
                    (group_size,)+self.img_shape)
                for j in range(len(dirs)):
                    # read brain images
                    fpath = '%s/%s/%s'% (
                        self.base_dir,
                        self.group_names[j],
                        self.DKT31_fname)
                    
                    mask = nib.load(fpath)
                    group_masks[j,:,:,:] = mask.get_data()
                    
                imgs[self.group_names[i]] += [group_masks]
        return imgs
                

def extract_dirs(addr):
    """Function for extracting all existing directories
    within a given folder
    """
    
    dirs = [
        os.path.join(addr,o) for o in os.listdir(addr) 
        if os.path.isdir(os.path.join(addr,o))]
    # only keep name of the directories
    dirs = [a[len(addr):] for a in dirs]
    
    return dirs


def gen_batch_inds(data_size, batch_size):
    """Generating a list of random indices to extract batches
    """
    
    # determine size of the batches
    quot, rem = np.divmod(data_size, batch_size)
    batches = list()
    
    # random permutation of the sample indices
    rand_perm = np.random.permutation(data_size).tolist()
    
    # assigning indices to batches
    for i in range(quot):
        if i<quot-1:
            this_batch = rand_perm[slice(i*batch_size, (i+1)*batch_size)]
        else:
            this_batch = rand_perm[slice(i*batch_size, data_size)]
        
        batches += [this_batch]
        
    return batches

def gen_batch_matrices(dat, batch_inds, col=True):
    """Generating a list of batch matrices  with data loaded into
    each batch accodting 
    
    The data shape is assumed to be (n_samples, n_features)
    """
    batches = []
    
    # loading the data into batches one-by-one
    for i in range(len(batch_inds)):
        # figuring shape of the current data batch
        batch_size = len(batch_inds[i])
        # column-wise
        if col:
            batch_dat = np.zeros((dat.shape[0], batch_size))
            for j in range(batch_size):
                batch_dat[:, j] = dat[:, batch_inds[i][j]]
        # row-wise
        else:
            batch_dat = np.zeros((batch_size, dat.shape[1]))
            for j in range(batch_size):
                batch_dat[j,:] = dat[batch_inds[i][j],:]
        
        batches += [batch_dat]
    
    return batches

def gen_batch_tensors(dat, batch_inds):
    """Generating a list of batch tensors with data loaded into
    each batch according to the batch_inds.
    
    Data samples are assumed to have dimension (data_size x 
    height x width x 1)
    """
    
    batches = []
    
    # loading the data into batches one-by-one
    for i in range(len(batch_inds)):
        # figuring shape of the current data batch
        batch_size = len(batch_inds[i])
        batch_dat_shape = (batch_size,) + dat.shape[1:]
        batch_dat = np.zeros(batch_dat_shape)
        for j in range(batch_size):
            batch_dat[j,:,:,:] = dat[batch_inds[i][j],:,:,:]
        
        batches += [batch_dat]
    
    return batches
