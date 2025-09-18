#include "PrepareVoxelOutput.h"

namespace flashmatch {
namespace dataprep {

PrepareVoxelOutput::PrepareVoxelOutput()
: _sce(nullptr)
{
    _sce = new larutil::SpaceChargeMicroBooNE( larutil::SpaceChargeMicroBooNE::kMCC9_Backward );
}

PrepareVoxelOutput::~PrepareVoxelOutput()
{
    if ( _sce )
        delete _sce;
    _sce = nullptr;
}

int PrepareVoxelOutput::makeVoxelChargeTensor( 
    const CosmicTrack& cosmic_track, 
    const std::vector<larcv::Image2D>& adc_v,
    const larflow::voxelizer::VoxelizeTriplets& voxelizer, 
    std::vector< std::vector<int> >& voxel_indices_vv,
    std::vector< std::vector<float> >& voxel_centers_vv,
    std::vector< std::vector<float> >& voxel_avepos_vv,
    std::vector< std::vector<float> >& voxel_planecharge_vv )
{

    // we loop through the spacepoints associated with the track and create 
    // a unique list of voxels the spacepoints occupy
    // then we get the voxel centers (and position mean)
    // and get the charge sum of the voxels

    // clear the output containers
    voxel_planecharge_vv.clear();
    voxel_indices_vv.clear();
    voxel_avepos_vv.clear();
    voxel_centers_vv.clear();

    typedef std::array<int,3> vindex_t;

    std::map< vindex_t, std::vector<int> > voxelindex_to_hitindex;
    std::map< vindex_t, std::vector<float> > voxelindex_to_avepos;

    //double x_t0_offset = opflash.flash_time*0.1098; 

    int num_outside_voxels = 0;
    for (int hitidx=0; hitidx<(int)cosmic_track.hitpos_v.size(); hitidx++) {

        std::vector<float> hit = cosmic_track.hitpos_v.at(hitidx);
        //hit[0] -= x_t0_offset; //already removed x offset
        bool applied_sce = false;
        std::vector<double> hit_sce 
            = _sce->ApplySpaceChargeEffect( hit[0], hit[1], hit[2], applied_sce);
        std::vector<float> fhit_sce = { (float)hit_sce[0], (float)hit_sce[1], (float)hit_sce[2] };

        vindex_t voxelindex;
        try {
            auto ivoxel_v = voxelizer.get_voxel_indices( fhit_sce );
            for (int i=0; i<3; i++)
                voxelindex[i] = ivoxel_v[i];
        }
        catch (...) {
            num_outside_voxels++;
            continue;
        }

        auto it_voxel_hitlist = voxelindex_to_hitindex.find( voxelindex );
        if ( it_voxel_hitlist==voxelindex_to_hitindex.end() ) {
            // voxel not yet registered
            voxelindex_to_hitindex[voxelindex] = std::vector<int>();
            voxelindex_to_avepos[voxelindex] = std::vector<float>(3,0);
            it_voxel_hitlist = voxelindex_to_hitindex.find( voxelindex );
        }

        // add hit
        it_voxel_hitlist->second.push_back( hitidx );
        voxelindex_to_avepos[voxelindex][0] += fhit_sce[0];
        voxelindex_to_avepos[voxelindex][1] += fhit_sce[1];
        voxelindex_to_avepos[voxelindex][2] += fhit_sce[2];
    }

    std::cout << "Flash-Matched track:" << std::endl;
    std::cout << "  number of hits: " << cosmic_track.hitpos_v.size() << std::endl;
    std::cout << "  nvoxels: " << voxelindex_to_hitindex.size() << std::endl;
    std::cout << "  hits outside voxelized volume: " << num_outside_voxels << std::endl;

    // finished hit-to-voxel assignment
    // now need to sum up position and charge values for each voxel.
    // this struct represents a pixel and is responsible for:
    //   1. storing the (row,col) position of the pixel and pixel value
    //   2. count the number of hits that project into a pixel. 
    //        will divide pixel value evenly across the hits
    struct Pixel_t {
        int row;
        int col;
        float pixval;
        int num_hits;
        vindex_t index;
        Pixel_t()
        : row(-1),col(-1),pixval(0.0),num_hits(0)
        {};
    };

    // For each plane, calculate the charge values assigned to the voxel
    int nplanes = adc_v.size();
    std::map<vindex_t,std::vector<float> > voxelindex_to_chargevalues;

    for (int plane=0; plane<nplanes; plane++) {

        auto const& meta = adc_v.at(plane).meta();
        auto const& img  = adc_v.at(plane);

        std::vector< Pixel_t > pixel_v; // store the pixels associated with the hits of this track
        std::map< std::pair<int,int>, int > pix_to_index; // key (row,col) -> value is index in pixel_v
        int pixcount = 0;
        std::map< vindex_t, std::vector<int> > voxelindex_to_pixindexlist; // voxel index to vector of indices to pixel_v

        // loop once to get which pixels we project into
        // we also count the number of times we project down

        for (auto it_voxel=voxelindex_to_hitindex.begin(); it_voxel!=voxelindex_to_hitindex.end();it_voxel++ ) {

            // loop over all hits assigned to this voxel
            for (auto& hitidx : it_voxel->second ) {

                auto const& imgpos = cosmic_track.hitimgpos_v.at(hitidx);
                int row = meta.row( imgpos[0] ); // tick to row
                int col = imgpos[plane+1];
                std::pair<int,int> pix(row,col);

                auto it_pixel = pix_to_index.find( pix );
                if ( it_pixel==pix_to_index.end() ) {
                    // not in map. make pixel.
                    Pixel_t pixel;
                    pixel.row = row;
                    pixel.col = col;
                    pixel.index = it_voxel->first;
                    pixel.pixval = img.pixel(row,col);
                    pixel.num_hits = 0;
                    pix_to_index[pix] = pixcount;
                    pixel_v.emplace_back( std::move(pixel) );
                    pixcount++;
                }

                // increment counter for number of hits projecting into this pixel
                int pixindex = pix_to_index[pix];
                pixel_v.at(pixindex).num_hits++;

                // provide a list of pixels whose hits fall within a voxel
                auto it_vox2pix = voxelindex_to_pixindexlist.find( it_voxel->first );
                if ( it_vox2pix==voxelindex_to_pixindexlist.end() ) {
                    voxelindex_to_pixindexlist[it_voxel->first] = std::vector<int>();
                }
                voxelindex_to_pixindexlist[it_voxel->first].push_back( pixindex );

            }//end of loop over hit indices assigned to voxel

            

        }// end of loop over voxels

        // loop again, using the assignments to sum the pixel values for each voxel
        for (auto it_voxel=voxelindex_to_hitindex.begin(); it_voxel!=voxelindex_to_hitindex.end();it_voxel++ ) {

            auto it_voxel_charge = voxelindex_to_chargevalues.find( it_voxel->first );
            if ( it_voxel_charge==voxelindex_to_chargevalues.end() ) {
                voxelindex_to_chargevalues[it_voxel->first] = std::vector<float>(3,0.0);
            }

            auto it_vox2pix = voxelindex_to_pixindexlist.find( it_voxel->first );

            // sum charge to set value for voxel
            float charge_sum = 0.0;

            for ( auto& pixindex : it_vox2pix->second ) {
                auto const& pixdata = pixel_v.at(pixindex);
                charge_sum += pixdata.pixval/float(pixdata.num_hits);
            }

            // assign to voxel
            voxelindex_to_chargevalues[it_voxel->first].at(plane) = charge_sum;

        }//end of loop over occupied voxels

    }//end of loop over planes

    // now loop over voxels one last time and collect the voxel charge and position data.
    // our goal is to fill
    //   voxel_planecharge_vv
    //   voxel_indices_vv
    //   voxel_avepos_vv
    //   voxel_centers_vv

    auto const& origin = voxelizer.get_origin();
    auto const& dimlen = voxelizer.get_dim_len();

    for (auto it_voxel=voxelindex_to_hitindex.begin(); it_voxel!=voxelindex_to_hitindex.end();it_voxel++ ) {
        std::vector<float> ave_pos(3,0);

        ave_pos = voxelindex_to_avepos[it_voxel->first];
        for (int i=0; i<3; i++)
            ave_pos[i] /= float(it_voxel->second.size());

        voxel_avepos_vv.push_back( ave_pos );
        voxel_planecharge_vv.push_back( voxelindex_to_chargevalues[it_voxel->first] );

        std::vector<int> voxel_axis_indices = voxelizer.get_voxel_indices( ave_pos );
        std::vector<float> centerpos(3,0);
        for (int i=0; i<3; i++) {
            centerpos[i] = (float(voxel_axis_indices[i])+0.5)*dimlen[i] + origin[i];
        }
        voxel_centers_vv.push_back( centerpos );
        voxel_indices_vv.push_back( voxel_axis_indices );
    }

    return voxel_planecharge_vv.size();
}

}
}