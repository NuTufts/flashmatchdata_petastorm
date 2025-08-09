#include "LarliteDataInterface.h"

#include <sstream>
#include <algorithm>

namespace flashmatch {
namespace dataprep {
    
/**
 * @brief Convert larlite opflash object into flashmatch opflash object
 */
OpticalFlash convert_opflash( const larlite::opflash& opflash )
{
    OpticalFlash out;

    out.flash_time = opflash.Time();
    out.pe_per_pmt.resize(32,0.0); // hardcoded for microboone. TODO: use larlite's geometry information to make more generic
    out.total_pe = 0.0;

    // Get PMT PE values
    // MicroBooNE has 32 PMTs.
    // The experiment saves 4 versions of each waveform coming from a given PMT
    //   1. (channels 0-31):    unbiased readout (for the beam). waveforms recorded based on beam trigger
    //   2. (channels 100-131): reduced unbiased readout where signal is reduced to 25% (for PMTs with large signals)
    //   3. (channels 200-231): a triggered cosmic readout (for out-of-time). waveform chunks saved if pulse is large enough.
    //   4. (channels 300-331): reduced cosmic readout where signal is reduced by 25%
    // The information in (1) and (2) are combined and saved into channels 0-31
    //   while the information in (3) and (4) are combined and saved into channels 200-231
    // We look into both, favoring the intime readout if both exists

    // These opflashes are the result of processing the above waveforms
    // It looks for waveform pulses and integrates the area under the pulse


    // To handle the different streams, we record the PE per PMT
    // for both the expected beam readout channels, [0,32),
    // and the expected cosmic readout channels, [200,32).
    // If there are signals in both, we default to the beam readout.
    // Otherwise, we use the readout with non-zero values.

    int cosmic_offset = 200; // first index where cosmic waveforms saved
    std::vector<float> beam_readout(32,0.0);
    std::vector<float> cosmic_readout(32,0.0);
    double beam_pe_total = 0.0;
    double cosmic_pe_total = 0.0;

    for (int ipmt=0; ipmt<32; ipmt++) {
        beam_readout[ipmt] = opflash.PE(ipmt);
        if ( opflash.nOpDets()>cosmic_offset+ipmt ) {
            cosmic_readout[ipmt] = opflash.PE(cosmic_offset+ipmt);
        }
        beam_pe_total   += beam_readout[ipmt];
        cosmic_pe_total += cosmic_readout[ipmt];
    }

    if ( beam_pe_total  > 0 ) {
        out.pe_per_pmt = beam_readout;
        out.total_pe   = beam_pe_total;
        out.readout    = 0;
    }
    else if ( cosmic_pe_total > 0 ) {
        out.pe_per_pmt = cosmic_readout;
        out.total_pe   = cosmic_pe_total;
        out.readout    = 1;
    }
    else {
        throw std::runtime_error("flashmatch::dataprep::LarliteDataInterface::convert_opflash(): no PE data found in either beam or cosmic channels");
    }

    out.flash_center[0] = -10.0;
    out.flash_center[1] = opflash.YCenter();
    out.flash_center[2] = opflash.ZCenter(); 
    out.flash_width_y   = opflash.YWidth();
    out.flash_width_z   = opflash.ZWidth();

    return out;
}

std::vector<OpticalFlash> convert_event_opflashes( 
    larlite::storage_manager& ioll,
    std::vector< std::string > opflash_src_treenames )
{
    if ( opflash_src_treenames.size()==0 ) {
        // hard-coded defaults
        opflash_src_treenames.push_back( "simpleFlashBeam");
        opflash_src_treenames.push_back( "simpleFlashCosmic");
    }

    // Make container for output objects
    std::vector< OpticalFlash > out_v;  

    // Loop over the name of trees to get opflashes from 
    // out of the larlite io manager
    int index =0 ;
    for ( auto& treename : opflash_src_treenames ) {
        larlite::event_opflash* ev_opflash = 
            (larlite::event_opflash*)ioll.get_data( larlite::data::kOpFlash, treename );
        if ( !ev_opflash || ev_opflash->size()==0 ) {
            continue;
        }
        for ( auto const& ll_opflash : *ev_opflash ) {
            OpticalFlash out_opflash = convert_opflash( ll_opflash );
            out_opflash.index = index;
            out_v.emplace_back( std::move(out_opflash) );
        }
    }

    return out_v;
}

std::vector<OpticalFlash> convert_event_opflashes( const std::vector<larlite::opflash>& opflash_v )
{
    // Make container for output objects
    std::vector< OpticalFlash > out_v;

    int index = 0;
    for ( auto const& ll_opflash : opflash_v ) {
        OpticalFlash out_opflash = convert_opflash( ll_opflash );
        out_opflash.index = index;
        index++;
        out_v.emplace_back( std::move(out_opflash) );
    }

    return out_v;
}


/**
 * @brief Copy over info for the tracks
 * 
 * @param track   Line segment description of path along particle track
 * @param hitinfo 3D positions of energy deposits used to fit the track path
 */
CosmicTrack convert_trackinfo( 
    const larlite::track& track,
    const std::vector< std::vector<float> >& hitinfo )
{

    CosmicTrack out;

    // transfer info from the energy deposits
    out.hitpos_v.reserve( hitinfo.size() );
    out.hitimgpos_v.reserve( hitinfo.size() );

    for ( auto const& hit : hitinfo ) {
        std::vector<float> hitpos(3);
        std::vector<float> imgpos(4);
        for (int v=0; v<3; v++)
            hitpos[v] = hit[v];
        for (int v=0; v<4; v++)
            imgpos[v] = hit[3+v];
        out.hitpos_v.push_back( hitpos );
        out.hitimgpos_v.push_back( imgpos );
    }

    // transfer points
    int npts = track.NumberTrajectoryPoints();
    if ( npts>0 )
        out.start_point = track.LocationAtPoint(0);
    if ( npts>1 )
        out.end_point = track.LocationAtPoint(npts-1);

    double tracklen = 0.0;
    for (int ipt=0; ipt<npts; ipt++) {
        out.points.push_back( track.LocationAtPoint(ipt) );
        if ( ipt>=1 ) {
            auto const& xcurrent = track.LocationAtPoint(ipt);
            auto const& xlast    = track.LocationAtPoint(ipt-1);
            tracklen += (xcurrent-xlast).Mag();
        }
    }
    out.track_length = tracklen;

    return out;
}

std::vector< CosmicTrack > convert_event_trackinfo( 
    const std::vector< larlite::track >& track_list,
    const std::vector< std::vector< std::vector<float> > >& hitinfo_list )
{
    std::vector< CosmicTrack > out_v;

    int ntracks = track_list.size();
    if ( ntracks!=(int)hitinfo_list.size() ) {
        std::stringstream msg;
        msg << "[LarliteDataInterface.cxx] flashmatch::dataprep::convert_event_trackinfo" << std::endl;
        msg << "  number of larlite::tracks (" << ntracks << ") != ";
        msg << "  number of hitinfo lists (" << hitinfo_list.size() << ")" << std::endl;
        throw std::runtime_error( msg.str() );
    }

    for ( int itrack=0; itrack<ntracks; itrack++ ) {
        auto const& ll_track = track_list.at(itrack);
        auto const& hitinfo  = hitinfo_list.at(itrack);

        CosmicTrack ctrack = convert_trackinfo( ll_track, hitinfo );
        ctrack.index = itrack;
        out_v.emplace_back( std::move(ctrack) );
    }

    return out_v;
}

CRTTrack convert_crttrack( const larlite::crttrack& ll_crttrack )
{

    CRTTrack out;
    
    out.start_point[0] = ll_crttrack.x1_pos;
    out.start_point[1] = ll_crttrack.y1_pos;
    out.start_point[2] = ll_crttrack.z1_pos;
    out.end_point[0]   = ll_crttrack.x2_pos;
    out.end_point[1]   = ll_crttrack.y2_pos;
    out.end_point[2]   = ll_crttrack.z2_pos;
    out.startpt_time   = ll_crttrack.ts2_ns_h1*0.001;
    out.endpt_time     = ll_crttrack.ts2_ns_h2*0.001;
    out.direction  = out.end_point - out.start_point;
    out.length     = out.direction.Mag();
    out.direction  = out.direction.Unit();

    return out;
}

std::vector< CRTTrack > convert_event_crttracks( const std::vector< larlite::crttrack>& ll_crttrack_list )
{
    std::vector< CRTTrack > out_v;
    for ( auto const& ll_crttrack : ll_crttrack_list ) {
        CRTTrack crttrack = convert_crttrack( ll_crttrack );
        crttrack.index = (int)out_v.size();
        out_v.emplace_back( std::move(crttrack ) );
    }
    return out_v;
}

CRTHit convert_crthit( const larlite::crthit& ll_crthit )
{

    CRTHit out;
    
    out.position[0] = ll_crthit.x_pos;
    out.position[1] = ll_crthit.y_pos;
    out.position[2] = ll_crthit.z_pos;
    out.time        = ll_crthit.ts2_ns*0.001;

    return out;
}

std::vector< CRTHit > convert_event_crthits( const std::vector< larlite::crthit>& ll_crthit_list )
{
    std::vector< CRTHit > out_v;
    int index =0;
    for ( auto const& ll_crthit : ll_crthit_list ) {
        CRTHit crthit = convert_crthit( ll_crthit );
        crthit.index = index;
        index++;
        out_v.emplace_back( std::move(crthit ) );
    }

    std::sort( out_v.begin(), out_v.end() );

    return out_v;
}

}
}