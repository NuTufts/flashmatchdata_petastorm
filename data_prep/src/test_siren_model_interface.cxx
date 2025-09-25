#include <iostream>


int main( int nargs, char** argv ) {

    std::cout << "Test Siren Model Interface" << std::endl;

    // TODOs
    //  1. Read in flashmatch HDF file 
    //     -- schema of hdf5 file can be found in FlashMatchHDF5Output.cxx. See function FlashMatchHDF5Output::writeBatch()
    //  2. Create SirenTorchModel and load weights using SirenTorchModel::load_model_file( ... )
    //  3. Create output root file and TTree
    //  4. Loop over entries in HDF file. For each entry:
    //  5.   load arrays: planecharge, avepos
    //  6.   pass this info into ModelInputInterface::_prepare_mlp_input_variables in ModelInputInterface.cxx to create input tensors to the model
    //  7.   run model
    //  8.   sum over voxels
    //  9.   rescale predicted pe output
    // 10.   save to TTree

    return 0;
}