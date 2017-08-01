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

def save_patches(img, s, out_addr):
    """Writing patches of an image into a text file, such that each 
    patch is saved as one row of the file (after vectorization)
    
    Input variables include the input 3D image, size of each patch
    (as a list of 3 elements), and name of the output file
    """
    
    (xs,ys,zs) = img.shape
    x_pnum = int(xs / s[0])
    y_pnum = int(ys / s[1])
    z_pnum = int(zs / s[2])

    # information threshold
    info_thr = 2.

    # saving the first image
    with open(out_addr,"w") as f:
        for i in range(x_pnum):
            for j in range(y_pnum):
                for k in range(z_pnum):

                    # 5x5x5 patches
                    # preparing the ranges:
                    # x
                    if i<x_pnum-1:
                        x_rng = [i*s[0] , (i+1)*s[0]]
                    else:
                        x_rng = [i*s[0] , xs]
                    # y
                    if j<y_pnum-1:
                        y_rng = [j*s[1] , (j+1)*s[1]]
                    else:
                        y_rng = [j*s[1] , ys]
                    # z
                    if k<z_pnum-1:
                        z_rng = [k*s[2] , (k+1)*s[2]]
                    else:
                        z_rng = [k*s[2] , zs]

                    patch = img[x_rng[0]:x_rng[1],
                                y_rng[0]:y_rng[1],
                                z_rng[0]:z_rng[1]]

                    # reshaping the patch to a row vector
                    patch = np.reshape(patch, np.prod(patch.shape))

                    # how much information the patch has
                    #info_metric = np.std(patch)
                    patch_hist, b_edges = np.histogram(patch, bins=20, density=True)
                    w = b_edges[1] - b_edges[0]
                    patch_hist *= w
                    patch_hist[patch_hist==0] += 1e-6
                    info_metric = -sum(patch_hist*np.log(patch_hist))

                    # save the patch only if it has sufficient info
                    if info_metric > info_thr:
                        f.write("".join(" ".join(map(str, x)) for x in (patch,))+'\n')
