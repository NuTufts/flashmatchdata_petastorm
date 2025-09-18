#include "PMTPositions.h"

namespace flashmatch {

bool PMTPositions::_loaded_array = false;

std::vector< std::array<float,3> > PMTPositions::_uboone_pmt_positions;

void PMTPositions::load_pmtpos_array() {

    _uboone_pmt_positions.clear();
    _uboone_pmt_positions.resize(36);

    // We Index by OpDet ID
    _uboone_pmt_positions[0]  = std::array<float,3>{-20.0000, -28.625, 990.356};
    _uboone_pmt_positions[1]  = std::array<float,3>{-20.0000,  27.607, 989.712};
    _uboone_pmt_positions[2]  = std::array<float,3>{-20.0000, -56.514, 951.865};
    _uboone_pmt_positions[3]  = std::array<float,3>{-20.0000,  55.313, 951.861};
    _uboone_pmt_positions[4]  = std::array<float,3>{-20.0000, -56.309, 911.939};
    _uboone_pmt_positions[5]  = std::array<float,3>{-20.0000,  55.822, 911.065};
    _uboone_pmt_positions[6]  = std::array<float,3>{-20.0000,  -0.722, 865.599};
    _uboone_pmt_positions[7]  = std::array<float,3>{-20.0000,  -0.502, 796.208};
    _uboone_pmt_positions[8]  = std::array<float,3>{-20.0000, -56.284, 751.905};
    _uboone_pmt_positions[9]  = std::array<float,3>{-20.0000,  55.625, 751.884};
    _uboone_pmt_positions[10] = std::array<float,3>{-20.0000, -56.408, 711.274};
    _uboone_pmt_positions[11] = std::array<float,3>{-20.0000,  55.800, 711.073};
    _uboone_pmt_positions[12] = std::array<float,3>{-20.0000,  -0.051, 664.203};
    _uboone_pmt_positions[13] = std::array<float,3>{-20.0000,  -0.549, 585.284};
    _uboone_pmt_positions[14] = std::array<float,3>{-20.0000,  55.822, 540.929};
    _uboone_pmt_positions[15] = std::array<float,3>{-20.0000, -56.205, 540.616};
    _uboone_pmt_positions[16] = std::array<float,3>{-20.0000, -56.323, 500.221};
    _uboone_pmt_positions[17] = std::array<float,3>{-20.0000,  55.771, 500.134};
    _uboone_pmt_positions[18] = std::array<float,3>{-20.0000,  -0.875, 453.096};
    _uboone_pmt_positions[19] = std::array<float,3>{-20.0000,  -0.706, 373.839};
    _uboone_pmt_positions[20] = std::array<float,3>{-20.0000, -57.022, 328.341};
    _uboone_pmt_positions[21] = std::array<float,3>{-20.0000,  54.693, 328.212};
    _uboone_pmt_positions[22] = std::array<float,3>{-20.0000,  54.646, 287.976};
    _uboone_pmt_positions[23] = std::array<float,3>{-20.0000, -56.261, 287.639};
    _uboone_pmt_positions[24] = std::array<float,3>{-20.0000,  -0.829, 242.014};
    _uboone_pmt_positions[25] = std::array<float,3>{-20.0000,  -0.303, 173.743};
    _uboone_pmt_positions[26] = std::array<float,3>{-20.0000,  55.249, 128.354};
    _uboone_pmt_positions[27] = std::array<float,3>{-20.0000, -56.203, 128.180};
    _uboone_pmt_positions[28] = std::array<float,3>{-20.0000, -56.615, 87.8695};
    _uboone_pmt_positions[29] = std::array<float,3>{-20.0000,  55.249, 87.7605};
    _uboone_pmt_positions[30] = std::array<float,3>{-20.0000,  27.431, 51.1015};
    _uboone_pmt_positions[31] = std::array<float,3>{-20.0000, -28.576, 50.4745};
    _uboone_pmt_positions[32] = std::array<float,3>{-161.3,-27.755 + 20/2*2.54, 760.575};    
    _uboone_pmt_positions[33] = std::array<float,3>{-161.3,-28.100 + 20/2*2.54, 550.333};
    _uboone_pmt_positions[34] = std::array<float,3>{-161.3,-27.994 + 20/2*2.54, 490.501};    
    _uboone_pmt_positions[35] = std::array<float,3>{-161.3,-28.201 + 20/2*2.54, 280.161};

    _loaded_array = true;
}

std::array<float,3> PMTPositions::getOpDetPos(int opdetid )
{
    if ( opdetid<0 || opdetid>=36 ) {
        return std::array<float,3>{0.0,0.0,0.0};
    }

    if ( !PMTPositions::_loaded_array )
        load_pmtpos_array();

    return _uboone_pmt_positions[opdetid];
}

}