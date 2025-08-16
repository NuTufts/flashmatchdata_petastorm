import os,sys
import time

from read_flashmatch_hdf5 import FlashMatchVoxelDataset

filelist = sys.argv[1]

print("filelist: ",filelist)
dataset = FlashMatchVoxelDataset(filelist, load_to_memory=False)
