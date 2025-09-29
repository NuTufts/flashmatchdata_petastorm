from __future__ import print_function

import torch

# ======================================================================
# we get the following by using dump_geometry for microboonev12.gdml
# provides position in global coordinates
# labels optical detector using Geant4 indexing which is based on some
# sorting after loading the GDML.
_opdet_pos = {
    0:(-11.4545,-28.625,990.356),
    1:(-11.4175,27.607,989.712),
    2:(-11.7755,-56.514,951.865),
    3:(-11.6415,55.313,951.861),
    4:(-12.0585,-56.309,911.939),
    5:(-11.8345,55.822,911.065),
    6:(-12.1765,-0.722,865.599),
    7:(-12.3045,-0.502,796.208),
    8:(-12.6045,-56.284,751.905),
    9:(-12.5405,55.625,751.884),
    10:(-12.6125,-56.408,711.274),
    11:(-12.6615,55.8,711.073),
    12:(-12.6245,-0.051,664.203),
    13:(-12.6515,-0.549,585.284),
    14:(-12.8735,55.822,540.929),
    15:(-12.6205,-56.205,540.616),
    16:(-12.5945,-56.323,500.221),
    17:(-12.9835,55.771,500.134),
    18:(-12.6185,-0.875,453.096),
    19:(-13.0855,-0.706,373.839),
    20:(-12.6485,-57.022,328.341),
    21:(-13.1865,54.693,328.212),
    22:(-13.4175,54.646,287.976),
    23:(-13.0075,-56.261,287.639),
    24:(-13.1505,-0.829,242.014),
    25:(-13.4415,-0.303,173.743),
    26:(-13.3965,55.249,128.354),
    27:(-13.2784,-56.203,128.18),
    28:(-13.2375,-56.615,87.8695),
    29:(-13.5415,55.249,87.7605),
    30:(-13.4345,27.431,51.1015),
    31:(-13.1525,-28.576,50.4745),
    32:[-161.3,-27.755 + 20/2*2.54, 760.575],    
    33:[-161.3,-28.100 + 20/2*2.54, 550.333],
    34:[-161.3,-27.994 + 20/2*2.54, 490.501],    
    35:[-161.3,-28.201 + 20/2*2.54, 280.161],    
}

# We Need to Map the OpDet Index
# to the OpChannelIndex, which is the index
# corresponding to electronics channels
# and the physical PMT
# We have 4 readouts of the PMT waveform,
# that is why we have 4 channel numbers per opdet
# But usually, the first channel is what
# we refer to as OpChannel
_opdet2opch_map = {0:(29,129,229,329),#P30
                  10:(23,123,223,323),#P24
                  11:(19,119,219,319),#P20
                  12:(21,121,221,321),#P22
                  13:(16,116,216,316),#P17
                  14:(14,114,214,314),#P15
                  15:(18,118,218,318),#P19
                  16:(17,117,217,317),#P18
                  17:(13,113,213,313),#P14
                  18:(15,115,215,315),#P16
                  19:(10,110,210,310),#P11
                  1:(27,127,227,327),#P28
                  20:(12,112,212,312),#P13
                  21:(8,108,208,308),#P09
                  22:(7,107,207,307),#P08
                  23:(11,111,211,311),#P12
                  24:(9,109,209,309),#P10
                  25:(3,103,203,303),#P04
                  26:(1,101,201,301),#P02
                  27:(6,106,206,306),#P07
                  28:(5,105,205,305),#P06
                  29:(0,100,200,300),#P01
                  2:(31,131,231,331),#P32
                  30:(2,102,202,302),#P03
                  31:(4,104,204,304),#P05
                  3:(26,126,226,326),#P27
                  4:(30,130,230,330),#P31
                  5:(25,125,225,325),#P26
                  6:(28,128,228,328),#P29
                  7:(22,122,222,322),#P23
                  8:(24,124,224,324),#P25
                  9:(20,120,220,320),#P21
}

# assuming coordinate system is at TPC x_min, z_min, and y_center
_tpc_origin = (-1.825,0.97,-4.0)

# need a translation from opchannel to opdet
_opch2opdet_map = {}
def fillOpCh2OpDetMap():
    global _opch2opdet_map
    for opdet in _opdet2opch_map:
        for opch in _opdet2opch_map[opdet]:
            _opch2opdet_map[ opch ] = opdet
        print("Set OpCh[",_opdet2opch_map[opdet][0],"] to OpDet[",opdet,"]")

def getPMTPosByOpDet( opdet, in_tpc_coord=True, use_v4_geom=True ):
    
    if use_v4_geom:
        pos = (-11.0, _pmtposmap[opdet][1], _pmtposmap[opdet][2] )
        return pos

    # else we use v12 geometry information.
    pos_global = _opdet_pos[opdet]
    if in_tpc_coord:
        pos_tpc_coord = [ pos_global[i]-_tpc_origin[i] for i in range(3) ]    
        return pos_tpc_coord
    return pos_global    
        
def getPMTPosByOpChannel( opch, in_tpc_coord=True, use_v4_geom=False ):
    
    if use_v4_geom:
        pos = (-11.0, _pmtposmap[opch][1], _pmtposmap[opch][2] )
        return pos

    # else we use v12 geometry information.    
    if len(_opch2opdet_map)==0:
        fillOpCh2OpDetMap()
    opdet = _opch2opdet_map[opch]
    return getPMTPosByOpDet(opdet, in_tpc_coord=in_tpc_coord)

        
#=======================================================================
# OLD PMT POS MAP, USING OLD UBOONE GDML
# where PMTs are mistakeningly in the TPC
# coordinates are in larsoft coordinates
_pmtposmap = {
     0:[2.458,  55.313, 951.861],
     1:[2.265,  55.822, 911.066],
     2:[2.682,  27.607, 989.712],
     3:[1.923, -0.7220, 865.598],
     4:[2.645, -28.625, 990.356],
     5:[2.324, -56.514, 951.865],
     6:[2.041, -56.309, 911.939],
     7:[1.559,  55.625, 751.884],
     8:[1.438,  55.800, 711.073],
     9:[1.795, -0.5020, 796.208],
    10:[1.475, -0.0510, 664.203],
    11:[1.495, -56.284, 751.905],
    12:[1.487, -56.408, 711.274],
    13:[1.226,  55.822, 540.929],
    14:[1.116,  55.771, 500.134],
    15:[1.448,  -0.549, 585.284],
    16:[1.481,  -0.875, 453.096],
    17:[1.479, -56.205, 540.616],
    18:[1.505, -56.323, 500.221],
    19:[0.913,  54.693, 328.212],
    20:[0.682,  54.646, 287.976],
    21:[1.014,  -0.706, 373.839],
    22:[0.949,  -0.829, 242.014],
    23:[1.451, -57.022, 328.341],
    24:[1.092, -56.261, 287.639],
    25:[0.703,  55.249, 128.355],
    26:[0.558,  55.249, 87.7605],
    27:[0.665,  27.431, 51.1015],
    28:[0.658,  -0.303, 173.743],
    29:[0.947, -28.576, 50.4745],
    30:[0.8211,-56.203, 128.179],
    31:[0.862, -56.615, 87.8695],
    35:[-161.3,-28.201 + 20/2*2.54, 280.161],
    34:[-161.3,-27.994 + 20/2*2.54, 490.501],
    33:[-161.3,-28.100 + 20/2*2.54, 550.333],
    32:[-161.3,-27.755 + 20/2*2.54, 760.575],
}

# _pmtposmap = {
#     26:[0.558, 55.249, 87.7605],
#     25:[0.703, 55.249, 128.355],
#     27:[0.665, 27.431, 51.1015],
#     28:[0.658, -0.303, 173.743],
#     29:[0.947, -28.576, 50.4745],
#     31:[0.862, -56.615, 87.8695],
#     30:[0.8211, -56.203, 128.179],
#     20:[0.682, 54.646, 287.976],
#     19:[0.913, 54.693, 328.212],
#     22:[0.949, -0.829, 242.014],
#     21:[1.014, -0.706, 373.839],
#     24:[1.092, -56.261, 287.639],
#     23:[1.451, -57.022, 328.341],
#     14:[1.116, 55.771, 500.134],
#     13:[1.226, 55.822, 540.929],
#     16:[1.481, -0.875, 453.096],
#     15:[1.448, -0.549, 585.284],
#     18:[1.505, -56.323, 500.221],
#     17:[1.479, -56.205, 540.616],
#     8:[1.438, 55.8, 711.073],
#     7:[1.559, 55.625, 751.884],
#     10:[1.475, -0.051, 664.203],
#     9:[1.795, -0.502, 796.208],
#     12:[1.487, -56.408, 711.274],
#     11:[1.495, -56.284, 751.905],
#     1:[2.265, 55.822, 911.066],
#     0:[2.458, 55.313, 951.861],
#     2:[2.682, 27.607, 989.712],
#     3:[1.923, -0.722, 865.598],
#     4:[2.645, -28.625, 990.356],
#     6:[2.041, -56.309, 911.939],
#     5:[2.324, -56.514, 951.865],
#     35:[-161.3,-28.201 + 20/2*2.54, 280.161],
#     34:[-161.3,-27.994 + 20/2*2.54, 490.501],
#     33:[-161.3,-28.100 + 20/2*2.54, 550.333],
#     32:[-161.3,-27.755 + 20/2*2.54, 760.575],
# }


def getPosFromID( id, origin_at_detcenter=False ):
    """ the id here is interpretted as the opchannel"""
    pos = getPMTPosByOpChannel(id)
    return pos

def getDetectorCenter():
    return [125.0,0.5*(-57.022+55.8),0.5*(990.356+51.1015)]

def create_pmtpos_tensor(apply_y_offset=False):
    # copy position data into numpy array format
    pmtpos = torch.zeros( (32, 3) )
    for i in range(32):
        opdetpos = getPMTPosByOpDet(i,use_v4_geom=True)
        for j in range(3):
            pmtpos[i,j] = opdetpos[j]
    # change coordinate system to 'tensor' system
    # main difference is y=0 is at bottom of TPC 
    if apply_y_offset:       
        pmtpos[:,1] -= -117.0
    # The pmt x-positions are wrong (!).
    # They would be in the TPC with the values I have stored.
    # So move them outside the TPC
    pmtpos[:,0] = -20.0
    # now corrected to be at -11, but need to keep things consistent
    return pmtpos

if __name__ == "__main__":

    print("OP CHANNEL POSITOINS ==============")
    for ich in range(0,36):
        pos = getPMTPosByOpChannel(ich,use_v4_geom=True)
        print("{",pos[0],",",pos[1],",",pos[2],"}",end='')
        if ich!=35:
            print(",",end='')
        print("// FEMCH%02d"%(ich))
    print("};")
        

