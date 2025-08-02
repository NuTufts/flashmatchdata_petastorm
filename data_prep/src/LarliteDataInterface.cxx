#include "LarliteDataInterface.h"

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
    std::vector<double> beam_readout(32,0.0);
    std::vector<double> cosmic_readout(32,0.0);
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
    for ( auto& treename : opflash_src_treenames ) {
        larlite::event_opflash* ev_opflash = 
            (larlite::event_opflash*)ioll.get_data( larlite::data::kOpFlash, treename );
        if ( !ev_opflash || ev_opflash->size()==0 ) {
            continue;
        }
        for ( auto const& ll_opflash : *ev_opflash ) {
            OpticalFlash out_opflash = convert_opflash( ll_opflash );
            out_v.emplace_back( std::move(out_opflash) );
        }
    }

    return out_v;
}



}
}