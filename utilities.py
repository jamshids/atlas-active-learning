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
