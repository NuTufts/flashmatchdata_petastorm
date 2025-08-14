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

void FlashMatchOutputData::clear() 
{
  track_v.clear();
  track_hits_v.clear();
  opflash_v.clear();
  crtmatch_v.clear();
  
  // Clear tree branch variables
  track_segments_v.clear();
  track_hitpos_v.clear();
  track_hitimgpos_v.clear();
  track_sce_segpts_v.clear();
  opflash_pe_v.clear();
  opflash_center.clear();
  opflash_time = 0.0;
  opflash_y_width = 0.0;
  opflash_z_width = 0.0;
  crtmatch_endpts_v.clear();
  predicted_pe_v.clear();
  match_type = -1;

  voxel_planecharge_vv.clear();
  voxel_indices_vv.clear();
  voxel_avepos_vv.clear();
  voxel_centers_vv.clear();

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
  _matched_tree->Branch("track_hitimgpos_v", &track_hitimgpos_v);
  _matched_tree->Branch("track_sce_segpts_v", &track_sce_segpts_v );
  _matched_tree->Branch("opflash_pe_v", &opflash_pe_v);
  _matched_tree->Branch("opflash_center", &opflash_center );
  _matched_tree->Branch("opflash_time", &opflash_time, "opflash_time/F" );
  _matched_tree->Branch("opflash_z_width", &opflash_z_width, "opflash_z_width/F" );
  _matched_tree->Branch("opflash_y_width", &opflash_y_width, "opflash_y_width/F" );
  _matched_tree->Branch("opflash_pe_total", &opflash_pe_total, "opflash_pe_total/F");
  _matched_tree->Branch("crtmatch_endpts_v", &crtmatch_endpts_v);
  _matched_tree->Branch("predicted_pe_v", &predicted_pe_v );
  _matched_tree->Branch("predicted_pe_total", &predicted_pe_total, "predicted_pe_total/F");
  _matched_tree->Branch("match_type", &match_type, "match_type/I");
  _matched_tree->Branch("voxel_planecharge_vv", &voxel_planecharge_vv );
  _matched_tree->Branch("voxel_indices_vv",     &voxel_indices_vv );
  _matched_tree->Branch("voxel_avepos_vv",      &voxel_avepos_vv );
  _matched_tree->Branch("voxel_centers_vv",     &voxel_centers_vv );


}

int FlashMatchOutputData::storeMatches( EventData& matched_data ) {

  if ( !_matched_tree ) {
    throw std::runtime_error("Cannot save matches - TTree not initialized");
  }

  int n_matches_saved = matched_data.cosmic_tracks.size();

  run    = matched_data.run;
  subrun = matched_data.subrun;
  event  = matched_data.event;

  for (int imatch=0; imatch<n_matches_saved; imatch++) {

    clear();

    auto const& cosmic_track = matched_data.cosmic_tracks.at(imatch);
    auto const& opflash      = matched_data.optical_flashes.at(imatch);
    auto const& crthit       = matched_data.crt_hits.at(imatch);
    auto const& crttrack     = matched_data.crt_tracks.at(imatch);
    match_type               = matched_data.match_type.at(imatch);

    track_segments_v.reserve( cosmic_track.points.size() );
    track_sce_segpts_v.reserve( cosmic_track.points.size() );
    for (auto const& pt : cosmic_track.points ) {
      std::vector<float> segment_pt(3,0);
      segment_pt[0] = pt[0];
      segment_pt[1] = pt[1];
      segment_pt[2] = pt[2];
      track_segments_v.push_back(segment_pt);
    }
    track_hitpos_v = cosmic_track.hitpos_v;
    track_hitimgpos_v = cosmic_track.hitimgpos_v;

    for (auto const& pt : cosmic_track.sce_points ) {
      std::vector<float> segment_pt(3,0);
      segment_pt[0] = pt[0];
      segment_pt[1] = pt[1];
      segment_pt[2] = pt[2];
      track_sce_segpts_v.push_back(segment_pt);
    }

    opflash_pe_v = opflash.pe_per_pmt;
    opflash_time = opflash.flash_time;
    opflash_center.resize(3,0);
    for (int i=0; i<3; i++)
      opflash_center[i] = opflash.flash_center[i];
    opflash_z_width = opflash.flash_width_z;
    opflash_y_width = opflash.flash_width_y;
    opflash_pe_total = 0.0;
    for (int i=0; i<32; i++) {
      opflash_pe_total += opflash_pe_v[i];
    }

    if ( crttrack.index>=0 ) {
      // valid CRT track
      std::vector<float> crthit1(3,0);
      std::vector<float> crthit2(3,0);
      for (int i=0; i<3; i++) {
        crthit1[i] = crttrack.start_point[i];
        crthit2[i] = crttrack.end_point[i];
      }
      crtmatch_endpts_v.push_back( crthit1 );
      crtmatch_endpts_v.push_back( crthit2 );
    }
    else if ( crthit.index>=0 ) {
      // valid CRT hit match
      std::vector<float> crthit1(3,0);
      for (int i=0; i<3; i++) {
        crthit1[i] = crthit.position[i];
      }
      crtmatch_endpts_v.push_back( crthit1 );
    }

    if ( matched_data.predicted_flashes.size()>0 
          && imatch<(int)matched_data.predicted_flashes.size() ) {

      predicted_pe_v.resize(32,0);
      predicted_pe_total = 0.0;
      for (int i=0; i<32; i++) {
        predicted_pe_v[i] = matched_data.predicted_flashes.at(imatch).pe_per_pmt[i];
        predicted_pe_total += predicted_pe_v[i];
      }

    }

    voxel_planecharge_vv = matched_data.voxel_planecharge_vv;
    voxel_indices_vv     = matched_data.voxel_indices_vv;
    voxel_avepos_vv      = matched_data.voxel_avepos_vv;
    voxel_centers_vv     = matched_data.voxel_centers_vv;

    _matched_tree->Fill();

  }

  clear(); // clear storage after use: for Herb.

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