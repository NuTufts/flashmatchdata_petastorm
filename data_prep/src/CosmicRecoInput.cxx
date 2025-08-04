#include "CosmicRecoInput.h"

#include <iostream>
#include <stdexcept>

namespace flashmatch {
namespace dataprep {

CosmicRecoInput::CosmicRecoInput( std::string inputfile )
    : _input_file(inputfile),
      _root_file(nullptr),
      _flashmatchtree(nullptr),
      _num_entries(0),
      _current_entry(-1),
      _br_track_v(nullptr),
      _br_opflash_v(nullptr),
      _br_crttrack_v(nullptr),
      _br_crthit_v(nullptr),
      _br_trackhits_v(nullptr)
{
    _load_tfile_and_ttree();
}

CosmicRecoInput::~CosmicRecoInput()
{
    _close();
}

void CosmicRecoInput::_load_tfile_and_ttree()
{
    // Open the ROOT file
    _root_file = TFile::Open(_input_file.c_str(), "READ");
    if (!_root_file || _root_file->IsZombie()) {
        throw std::runtime_error("Failed to open ROOT file: " + _input_file);
    }

    // Get the FlashMatchData tree
    _flashmatchtree = (TTree*)_root_file->Get("FlashMatchData");
    if (!_flashmatchtree) {
        throw std::runtime_error("Failed to find FlashMatchData tree in file: " + _input_file);
    }

    // Get the number of entries
    _num_entries = _flashmatchtree->GetEntries();
    std::cout << "Loaded " << _num_entries << " entries from " << _input_file << std::endl;

    // Initialize branch pointers
    // _br_track_v    = new std::vector<larlite::track>();
    // _br_opflash_v  = new std::vector<larlite::opflash>();
    // _br_crttrack_v = new std::vector<larlite::crttrack>();
    // _br_crthit_v   = new std::vector<larlite::crthit>();
    // _br_trackhits_v = new std::vector< std::vector< std::vector<float> > >();

    // Set branch addresses
    _flashmatchtree->SetBranchAddress("run",          &run);
    _flashmatchtree->SetBranchAddress("subrun",       &subrun);
    _flashmatchtree->SetBranchAddress("event",        &event);
    _flashmatchtree->SetBranchAddress("track_v",      &_br_track_v);
    _flashmatchtree->SetBranchAddress("opflash_v",    &_br_opflash_v);
    _flashmatchtree->SetBranchAddress("crttrack_v",   &_br_crttrack_v);
    _flashmatchtree->SetBranchAddress("crthit_v",     &_br_crthit_v);
    _flashmatchtree->SetBranchAddress("trackhits_vv", &_br_trackhits_v);
}

void CosmicRecoInput::load_entry( int ientry )
{
    if (ientry < 0 || ientry >= _num_entries) {
        throw std::out_of_range("Entry " + std::to_string(ientry) + " out of range [0, " + 
                               std::to_string(_num_entries-1) + "]");
    }

    _flashmatchtree->GetEntry(ientry);
    _current_entry = ientry;
}

void CosmicRecoInput::next_entry()
{
    if (_current_entry + 1 >= _num_entries) {
        throw std::out_of_range("No more entries to read");
    }

    load_entry(_current_entry + 1);
}

void CosmicRecoInput::_close()
{
    // Clean up branch pointers
    if (_br_track_v) {
        delete _br_track_v;
        _br_track_v = nullptr;
    }
    if (_br_opflash_v) {
        delete _br_opflash_v;
        _br_opflash_v = nullptr;
    }
    if (_br_crttrack_v) {
        delete _br_crttrack_v;
        _br_crttrack_v = nullptr;
    }
    if (_br_crthit_v) {
        delete _br_crthit_v;
        _br_crthit_v = nullptr;
    }
    if (_br_trackhits_v) {
        delete _br_trackhits_v;
        _br_trackhits_v = nullptr;
    }

    // Close ROOT file
    if (_root_file) {
        _root_file->Close();
        delete _root_file;
        _root_file = nullptr;
    }

    _flashmatchtree = nullptr;
    _num_entries = 0;
    _current_entry = -1;
}

}
}