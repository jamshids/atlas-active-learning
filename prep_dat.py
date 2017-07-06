import numpy as np
import pdb

def normalize_nii(dat):
    """Normalizing the Nifti images to values between 0 and 256
    """
    
    return np.round(256. * dat/dat.max())
    
    

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

def gen_batch_tensors(dat, batch_inds):
    """Generating a list of batches with data loaded into each 
    batch according to the batch_inds.
    
    Data samples are assumed to have dimension (data_size x 
    height x width x 1)
    """
    
    batches = []
    
    # getting data shape using just the first sample
    batch_size = len(batch_inds[0])
    batch_dat_shape = (batch_size,) + dat.shape[1:]
    
    # loading the data into batches one-by-one
    for i in range(len(batch_inds)):
        batch_dat = np.zeros(batch_dat_shape)
        #pdb.set_trace()
        for j in range(batch_size):
            batch_dat[j,:,:,:] = dat[batch_inds[i][j],:,:,:]
        
        batches += [batch_dat]
    
    return batches
