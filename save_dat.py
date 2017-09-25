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

def save_patches(img1, img2, s, out_addrs):
    """Writing patches of two images into text files, such that each 
    patch is saved as one row of the file (after vectorization), also
    saving an edge list showing neighbors of each saved patch in the 
    first image in terms of the row number of the patch-matrix of the
    second image
    
    Input variables include the input 3D images, size of each patch
    (as a list of 3 elements), and name of the output file
    """
    
    # supposing that the dimensions are the same
    (xs,ys,zs) = img1.shape
    half_pxs = int(s[0] / 2)
    half_pys = int(s[1] / 2)
    half_pzs = int(s[2] / 2)

    # information threshold
    info_thr = 2.
    
    # local neighborhood sizes
    lx, ly, lz = (5, 5, 5)
    
    # saving the first image
    locs_tensor = np.zeros(img1.shape, dtype=int)*np.nan
    row_cnt = 0
    with open(out_addrs[0],"w") as f:
        for i in range(half_pxs, xs - half_pxs):
            for j in range(half_pys, ys - half_pys):
                for k in range(half_pzs, zs - half_pzs):
                    
                    # 5x5x5 patches
                    patch = img1[i-half_pxs : i+half_pxs,
                                 j-half_pys : j+half_pys,
                                 k-half_pzs : k+half_pzs]

                    # save the patch, and its row-number if in the location-tensor
                    # only if it has sufficient info
                    if qualify_patch(patch, info_thr):
                        patch = np.reshape(patch, np.prod(patch.shape))
                        f.write("".join(" ".join(map(str, x)) for x in (patch,))+'\n')
                        locs_tensor[i,j,k] = row_cnt
                        row_cnt += 1
                        
    # initializing the string containing all the edges:
    edges = [str(x)+"\n" for x in range(row_cnt)]
            
    # udating the row-count to use it for patches of the second image
    row_cnt = 0
    # saving the second image, plus the edge list
    with open(out_addrs[1], "w") as f:
        for i in range(half_pxs, xs - half_pxs):
            for j in range(half_pys, ys - half_pys):
                for k in range(half_pzs, zs - half_pzs):
                    # 5x5x5 patches
                    patch = img2[i-half_pxs : i+half_pxs,
                                 j-half_pys : j+half_pys,
                                 k-half_pzs : k+half_pzs]

                    # save the patch, and its row-number if in the location-tensor
                    # only if it has sufficient info
                    if qualify_patch(patch, info_thr):
                        patch = np.reshape(patch, np.prod(patch.shape))
                        f.write("".join(" ".join(map(str, x)) for x in (patch,))+'\n')
                        row_cnt += 1

                        # consider neighborhood of this patch in the first image
                        nborders = np.array([[i-lx, j-ly, k-lz],
                                             [i+lx, j+ly, k+lz]])
                        # check if the patches in that neighborhood is already saved
                        # as selected patches from the first image
                        for ii in range(i-lx, i+lx+1):
                            if ii<0 or ii>=xs: continue
                            for jj in range(j-ly, j+ly+1):
                                if jj<0 or jj>=ys: continue
                                for kk in range(k-lz, k+lz+1):
                                    if kk<0 or kk>=zs: continue
                                    # if it is marked, add the patch in the
                                    # second image to the list
                                    if ~np.isnan(locs_tensor[ii,jj,kk]):
                                        #pdb.set_trace()
                                        new_edges = edges[int(locs_tensor[ii,jj,kk])][:-1] + \
                                            " " + str(row_cnt) + "\n"
                                        edges[int(locs_tensor[ii,jj,kk])] = new_edges
                                        
    # now, save the edge list into a separate file
    with open(out_addrs[2], "w") as f:
        f.write("".join(edges))
                                        
            
            
def qualify_patch(patch, thr):
    """Qualifying a patch to be saved or not
    
    This function is useful when saving a lot of patches to compute
    cross-correlation between two images.
    """
    
    # reshaping the patch to a row vector
    patch = np.reshape(patch, np.prod(patch.shape))

    # how much information the patch has
    info_metric = np.std(patch)
    #patch_hist, b_edges = np.histogram(patch, bins=20, density=True)
    #w = b_edges[1] - b_edges[0]
    #patch_hist *= w
    #patch_hist[patch_hist==0] += 1e-6
    #info_metric = -sum(patch_hist*np.log(patch_hist))
    
    check = info_metric > thr
    
    return check
