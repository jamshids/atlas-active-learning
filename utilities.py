import numpy as np
import nrrd

def pairwise_masked_CC(cc_path, index_path, mask):
    """Extracging CC values of a pair of images only at a 
    maksed region.
    """
    
    CC = np.loadtxt(cc_path)
    CC = CC[:-1]
    index_file = open(index_path, "rb")
    XYZ = index_file.readlines()

    # preparing the CC-map
    valid_CCs = np.where(CC>0)[0]
    #coords = np.zeros((len(valid_CCs), 3))
    cc_map = np.zeros(mask.shape)

    for i in range(len(valid_CCs)):        
        # get the 3D coordinate
        row_idx = valid_CCs[i]
        
        coord_str = XYZ[row_idx].decode("utf-8")[1:-2].split(',')
        coords = [int(coord) for coord in coord_str]
        # continue if the point is not in the masked region
        if not(mask[coords[0], coords[1], coords[2]]):
            continue
        # copying the CC to the right location
        cc_map[coords[0],coords[1],coords[2]] = CC[row_idx]
        
    return cc_map

def local_CC_similarity(data_obj, cc_dir,indices_dir,LOI, start_ID):
    """Computing a similarity matrix based on local cross-correlations
    
    *Parameters:*
    
      data_obj: object of Mindboggle_101 class
        make sure to run data_obj.list_data_path() before calling this
        function, so that the paths to all images and masks are listed
      
      cc_dir: string
        directory path where the cross-correlation files exists; these
        include values of cross-correlation of the pactches in each
        image versus another
        
     indices_dir: string
       directory path to the folder where indices of non-ingored patches
       are store; for each pair of images we have one such a file which
       specifies the patches that are non-homogeneous enough to take
       part in computing cross-correlation between that pair of images
       
     masks: 3D array
          label map of the first image 
          
     LOI: list of integers
       labels of interest
    """
    
    n_samples = 100

    #S = np.zeros((n_samples,n_samples))
    S = np.loadtxt('Heschle_gyrus_sim.txt')
    for i in range(start_ID, n_samples):
        S[i,i] = 1.
        if i==n_samples-1:
            break
        
        # reading mask of the first image
        seg_path = data_obj.segs_path[i]
        seg, options = nrrd.read(seg_path)
        
        # keeping only the labels of interest
        mask = np.zeros(seg.shape, dtype=bool)
        for label in LOI:
            mask = np.logical_or(mask, seg==label)
        
        # reading the CC map between the i-th image and others
        for j in range(i+1, n_samples):
            cc_path = '%s/cc_%d-%d.txt'% (cc_dir, i, j)
            index_path = '%s/index_%d-%d.txt'% (indices_dir,  i, j)
            
            cc_map = pairwise_masked_CC(cc_path, index_path, mask)
            S[i,j] = np.mean(cc_map[cc_map>0])
            S[j,i] = S[i,j]
            
            print("(%d, %d)"% (i,j), end=', ')
            np.savetxt('Heschle_gyrus_sim_2.txt', S)
    return S
            
