import ROOT as rt

"""
modules and functions to visualize using ROOT.
right now, makes a canvas for saving to png.
"""

def single_event_visualization( pe_per_pmt, pe_per_pmt_truth, vox_pos_cm, qmean_per_voxel, pmt_x, pmt_y ):
    """
    inputs
    ------

    pe_per_pmt [torch (float32) tensor] this is the predicted pe per pmt for 1 track-flash pair.
               Should be shape (C), where C is the number of PMTs (32)
    pe_per_pmt_truth [torch (float32) tensors]  This is the target pe per pmt for 1 track-flash pair. 
               Should also be shape (C)
    vox_pos_cm [float32 tensor] Contains the position of the charge voxels. In cm.
    qmean_per_voxel [float32 tensor] The mean charge (over the wire planes) at each voxel.
    pmt_x [float32 tensor] The z-position (in det coordinates) of the 32 PMTs (C,)
    pmt_y [float32 tensor] The y-position (in det coordinates) of the 32 PMTs (C,)
    """
    # make canvas
    c = rt.TCanvas("c", "",1500,600)
    c.Divide(2,2)

    # define draw tpc box
    yzbox = rt.TBox( 0.0, -116.5, 1036.0, +116.5 )
    yzbox.SetLineColor( rt.kBlack )
    yzbox.SetFillStyle(0)
    xybox = rt.TBox( 0.0, -116.5,  256.0, +116.5 )
    xybox.SetLineColor( rt.kBlack )
    xybox.SetFillStyle(0)

    # histograms to compare pe predictions
    hpe_pred    = rt.TH1F("hpe_pred","",32,0,32)
    hpe_target  = rt.TH1F("hpe_target","",32,0,32)

    
    # canvas 1: y-z projection
    c.cd(1)
    # make hist to set coordinates
    hyz = rt.TH2D("hyz",";z (cm); y(cm)", 100, -20.0, 1056.0, 100, -140, +140.0)
    hyz.Draw()
    yzbox.Draw()

    pred_circ_v = []
    for i in range(32):
        pe = pe_per_pmt[i]
        #print("pred pe[",i,"]: ",pe)
        hpe_pred.SetBinContent(i+1,pe)
        r = min(10.0*pe,50.0)
        pe_circle = rt.TEllipse( pmt_x[i], pmt_y[i], r, r )
        pe_circle.SetLineColor(rt.kRed)
        pe_circle.SetFillStyle(0)
        pe_circle.SetLineWidth(3)
        pe_circle.Draw()        
        pred_circ_v.append( pe_circle )
        
    # draw target
    target_circ_v = []
    for i in range(32):
        pe = pe_per_pmt_truth[i]
        #print("target pe[",i,"]: ",pe)
        hpe_target.SetBinContent(i+1,pe)
        r = min(10.0*pe,50.0)
        pe_circle = rt.TEllipse( pmt_x[i], pmt_y[i], r, r )
        pe_circle.SetFillStyle(0)                
        pe_circle.Draw()
        target_circ_v.append( pe_circle )

    # draw voxels: maybe a set of boxes?
    nvoxels = vox_pos_cm.shape[0]
    vox_box_xy_v = []
    vox_box_zy_v = []
    vox_graph = rt.TGraph( vox_pos_cm.shape[0] )
    for i in range(nvoxels):
        # average is maybe a sum of 0.4 of scaled value
        # an there is about 48 pixels if you move directly vertically.
        # calibrate box size to a sum of 0.4 assuming on average 65 voxels ~ 48*sqrt(2)
        l = min(qmean_per_voxel[i]*5.0*(0.4/65.0),10.0) #
        box_zy = rt.TBox( vox_pos_cm[i,2]-l/2.0, vox_pos_cm[i,1]-l/2.0,
                          vox_pos_cm[i,2]+l/2.0, vox_pos_cm[i,1]+l/2.0 )
        box_xy = rt.TBox( vox_pos_cm[i,0]-l/2.0, vox_pos_cm[i,1]-l/2.0,
                          vox_pos_cm[i,0]+l/2.0, vox_pos_cm[i,1]+l/2.0 )
        box_xy.SetFillStyle(0)
        box_zy.SetFillStyle(0)
        box_zy.Draw() # draw in zy
        vox_box_zy_v.append( box_zy )
        vox_box_xy_v.append( box_xy )
        vox_graph.SetPoint(i, vox_pos_cm[i,2], vox_pos_cm[i,1]  )
    vox_graph.SetMarkerStyle(21)
    vox_graph.SetMarkerSize(2)
    #vox_graph.Draw("Psame")

    # change to xy plot (only charge is shown)
    c.cd(2)
    hxy = rt.TH2D("hxy",";x (cm); y(cm)", 100, -10.0, 266.0, 100, -140.0, +140.0)
    hxy.Draw()
    xybox.Draw()
    # draw in the charge boxes
    for b in vox_box_xy_v:
        b.Draw()
    

    # histogram: pe
    c.cd(3)
    hpe_pred.SetMinimum(0.0)
    hpe_target.SetMinimum(0.0)
    hpe_pred.SetLineColor(rt.kRed)    
    if hpe_pred.GetMaximum()>hpe_target.GetMaximum():
        hpe_pred.Draw("hist")
        hpe_target.Draw("histsame")
    else:
        hpe_target.Draw("hist")        
        hpe_pred.Draw("histsame")

    # histogram pdf
    c.cd(4)
    hpdf_pred   = hpe_pred.Clone("hpdf_pred")
    hpdf_target = hpe_target.Clone("hpdf_target")    
    if hpdf_pred.Integral()>0.0:
        hpdf_pred.Scale( 1.0/hpdf_pred.Integral() )
        hpdf_pred.SetLineColor(rt.kRed)
    if hpdf_target.Integral()>0.0:
        hpdf_target.Scale( 1.0/hpdf_target.Integral() )
    if hpdf_pred.GetMaximum()>hpdf_target.GetMaximum():
        hpdf_pred.Draw("hist")
        hpdf_target.Draw("histsame")
    else:
        hpdf_target.Draw("hist")
        hpdf_pred.Draw("histsame")        


    c.Update()

    # return all products
    return c, (hyz, hxy, yzbox, xybox, pred_circ_v, target_circ_v,
               hpdf_pred, hpdf_target,
               vox_box_xy_v, vox_box_zy_v,
               hpe_pred, hpe_target)

