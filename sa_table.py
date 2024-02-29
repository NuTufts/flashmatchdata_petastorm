import os,sys,time
import numpy as np

def read_csv_file(filepath='SA_FullDetector_voxelsize5_32PMTs.csv'):
    my_data = np.loadtxt(filepath, delimiter=',')

    sa_coords = my_data[:,0:3].astype(np.int64)
    sa_values = my_data[:,3:] #(N,32)

    print("pre-reshape coord to test: ",sa_coords[10,:])
    print("pre-reshape SA for row: ",sa_values[10,:])
    print("pre-reshape coord: ",sa_coords.shape)
    print("pre-reshape SA: ",sa_values.shape)

    # need to reshape to (nx, ny, nz, 32)
    sa_values = sa_values.reshape( (54, 49, 210, 32 ) )
    print("post-reshape check of row: ",sa_coords[10,:])
    row = sa_coords[10,:] # (1,3) or (3,)
    print(sa_values[row[0],row[1],row[2],:])
    print("post-reshape SA: ",sa_values.shape)
    
    return sa_coords,sa_values

def save_reshaped_array( outpath, sa_coords, sa_values ):
    np.savez_compressed( outpath, sa_coords=sa_coords, sa_values=sa_values )

def load_satable_fromnpz( filepath="sa_5cmvoxels.npz" ):
    array_dict = np.load( filepath )
    return array_dict['sa_coords'],array_dict['sa_values']

def get_satable_maxindices():
    return (54,49,210)


if __name__=="__main__":

    # load from original textfile, made from grid scripts running solidAngle.py
    if True:
        start=time.time()
        sa_coords,sa_values = read_csv_file()
        dt = time.time()-start
        print(sa_values.shape)
        print("time to read: ",dt," secs")

    if True:
        print("save compressed numpy array")
        save_reshaped_array("sa_5cmvoxels.npz",sa_coords, sa_values)

    if True:
        print("loading compressed numpy array file")
        start = time.time()
        sa_coords, sa_values = load_satable_fromnpz()
        dt = time.time()-start
        print("sa_coords: ",sa_coords.shape)
        print("sa_values: ",sa_values.shape)
        print("time to load from npz: ",dt," secs")
