#include "FlashMatchOutputData.h"

#include <iostream>
#include <stdexcept>

namespace flashmatch {
namespace dataprep {

FlashMatchOutputData::FlashMatchOutputData( std::string output_rootfile, bool allow_overwrite ) 
  : run(-1), subrun(-1), event(-1), _file(nullptr), _matched_tree(nullptr), matchindex(0)
{
  // Open output file
  if ( allow_overwrite ) {
    _file = new TFile( output_rootfile.c_str(), "RECREATE" );
  } else {
    _file = new TFile( output_rootfile.c_str(), "CREATE" );
  }
  
  if ( !_file || !_file->IsOpen() ) {
    throw std::runtime_error("Failed to open output file: " + output_rootfile);
  }
  
  // Create the output tree
  makeMatchTTree();
}

FlashMatchOutputData::~FlashMatchOutputData() 
{
  if ( _file && _file->IsOpen() ) {
    writeTree();
    closeFile();
  }
}

FlashMatchOutputData::CRTHitPos_t FlashMatchOutputData::nullCRTHitPos() 
{
  return CRTHitPos_t{0.0f, 0.0f, 0.0f, 0.0f};
}

FlashMatchOutputData::CRTMatchLine_t FlashMatchOutputData::nullCRTMatchLine() 
{
  return CRTMatchLine_t{ nullCRTHitPos(), nullCRTHitPos() };
}

bool FlashMatchOutputData::isNullCRTHitPos( CRTHitPos_t& hit ) 
{
  return (hit[0] == 0.0f && hit[1] == 0.0f && hit[2] == 0.0f && hit[3] == 0.0f);
}

bool FlashMatchOutputData::isNullCRTMatchLine( CRTMatchLine_t& matchline ) 
{
  return isNullCRTHitPos(matchline[0]) && isNullCRTHitPos(matchline[1]);
}

int FlashMatchOutputData::addTrackFlashMatch( larlite::track& track,  
                                              larlite::opflash& opflash, 
                                              larlite::larflowcluster* hitcluster,
                                              CRTMatchLine_t* crtmatch )
{
  // Add track
  track_v.push_back(track);
  
  // Add opflash
  opflash_v.push_back(opflash);
  
  // Add hit cluster if provided
  if ( hitcluster != nullptr ) {
    track_hits_v.push_back(*hitcluster);
  } else {
    // Add empty cluster
    larlite::larflowcluster empty_cluster;
    track_hits_v.push_back(empty_cluster);
  }
  
  // Add CRT match if provided
  if ( crtmatch != nullptr ) {
    crtmatch_v.push_back(*crtmatch);
  } else {
    // Add null CRT match
    crtmatch_v.push_back(nullCRTMatchLine());
  }
  
  // Return the index of the match (0-based)
  return track_v.size() - 1;
}

void FlashMatchOutputData::clear() 
{
  track_v.clear();
  track_hits_v.clear();
  opflash_v.clear();
  crtmatch_v.clear();
  
  // Clear tree branch variables
  track_segments_v.clear();
  track_hitpos_v.clear();
  track_hitfeat_v.clear();
  opflash_pe_v.clear();
  crtmatch_endpts_v.clear();
}

void FlashMatchOutputData::makeMatchTTree() 
{
  if ( !_file || !_file->IsOpen() ) {
    throw std::runtime_error("Cannot create TTree - output file is not open");
  }
  
  _matched_tree = new TTree("flashmatch", "Flash-matched tracks");
  
  // Set up branches
  _matched_tree->Branch("run", &run, "run/I");
  _matched_tree->Branch("subrun", &subrun, "subrun/I");
  _matched_tree->Branch("event", &event, "event/I");
  _matched_tree->Branch("matchindex", &matchindex, "matchindex/I");
  _matched_tree->Branch("track_segments_v", &track_segments_v);
  _matched_tree->Branch("track_hitpos_v", &track_hitpos_v);
  _matched_tree->Branch("track_hitfeat_v", &track_hitfeat_v);
  _matched_tree->Branch("opflash_pe_v", &opflash_pe_v);
  _matched_tree->Branch("crtmatch_endpts_v", &crtmatch_endpts_v);
}

int FlashMatchOutputData::saveEventMatches() 
{
  if ( !_matched_tree ) {
    throw std::runtime_error("Cannot save matches - TTree not initialized");
  }
  
  int n_matches_saved = 0;
  
  // Loop over all matches in this event
  for ( size_t imatch=0; imatch < track_v.size(); imatch++ ) {
    
    // Clear branch variables
    track_segments_v.clear();
    track_hitpos_v.clear();
    track_hitfeat_v.clear();
    opflash_pe_v.clear();
    crtmatch_endpts_v.clear();
    
    // Set match index
    matchindex = imatch;
    
    // Fill track segments
    const larlite::track& track = track_v[imatch];
    for ( size_t ipt=0; ipt < track.NumberTrajectoryPoints(); ipt++ ) {
      auto pt = track.LocationAtPoint(ipt);
      std::vector<float> segment_pt = { (float)pt.X(), (float)pt.Y(), (float)pt.Z() };
      track_segments_v.push_back(segment_pt);
    }
    
    // Fill track hit positions and features if available
    if ( imatch < track_hits_v.size() ) {
      const larlite::larflowcluster& hitcluster = track_hits_v[imatch];
      for ( size_t ihit=0; ihit < hitcluster.size(); ihit++ ) {
        const larlite::larflow3dhit& hit = hitcluster[ihit];
        std::vector<float> hitpos = { (float)hit[0], (float)hit[1], (float)hit[2] };
        track_hitpos_v.push_back(hitpos);
        
        // Features: store charge from each plane
        std::vector<float> hitfeat(3,0);
        // TODO: recall how the charge is stored
        // for ( int p=0; p < 3; p++ ) {
        //   hitfeat.push_back( hit.pixeladc[p] );
        // }
        track_hitfeat_v.push_back(hitfeat);
      }
    }
    
    // Fill opflash PE values
    const larlite::opflash& flash = opflash_v[imatch];
    for ( size_t ipmt=0; ipmt < flash.nOpDets(); ipmt++ ) {
      opflash_pe_v.push_back( flash.PE(ipmt) );
    }
    
    // Fill CRT match endpoints
    if ( imatch < crtmatch_v.size() ) {
      const CRTMatchLine_t& crtmatch = crtmatch_v[imatch];
      for ( int iend=0; iend < 2; iend++ ) {
        std::vector<float> endpoint;
        for ( int i=0; i < 4; i++ ) {
          endpoint.push_back( crtmatch[iend][i] );
        }
        crtmatch_endpts_v.push_back(endpoint);
      }
    } else {
      // Add null CRT match
      for ( int iend=0; iend < 2; iend++ ) {
        std::vector<float> endpoint = {0.0f, 0.0f, 0.0f, 0.0f};
        crtmatch_endpts_v.push_back(endpoint);
      }
    }
    
    // Fill the tree
    _matched_tree->Fill();
    n_matches_saved++;
  }
  
  return n_matches_saved;
}

void FlashMatchOutputData::writeTree() 
{
  if ( _file && _file->IsOpen() && _matched_tree ) {
    _file->cd();
    _matched_tree->Write();
  }
}

void FlashMatchOutputData::closeFile() 
{
  if ( _matched_tree ) {
    delete _matched_tree;
    _matched_tree = nullptr;
  }
  
  if ( _file ) {
    if ( _file->IsOpen() ) {
      _file->Close();
    }
    delete _file;
    _file = nullptr;
  }
}

} // namespace dataprep
} // namespace flashmatch