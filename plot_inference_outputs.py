import os,sys
import ROOT as rt

rt.gStyle.SetOptStat(0)

#filelist = ['corsika_382k','corsika_260k']
#filelist = ['extbnb_685k','extbnb_775k']
#filelist = ['corsika_382k','corsika_52kk']
#filelist = ['corsika_634k']
#filelist = ['temp']
#filelist = ['corsika_151k']
#filelist = ['extbnb_rare_monkey_124k']
#filelist = ['good_sun_115k']
filelist = ['colorful_haze_057k'] # trained with unbalanced loss

variable_list = ['sinkhorn','unbsinkhorn','fracerr','pe_tot','diff_sinkhorn','diff_unbsinkhorn','fracerr_remake','pe_tot_remake']

remake_factors = {
    'corsika_188k':2.5,
    'extbnb_111k':2.3,
    'extbnb_230k':1.8,
    'extbnb_371k':1.5,
    'corsika_151k':2.5,
    'extbnb_rare_monkey_124k':2.0,
    'good_sun_060k':2.6,
    'good_sun_115k':3.0,
    'colorful_haze_107k':1.0,
    'colorful_haze_057k':1.0,    
    'temp':1.0
}

filepaths = {
    'extbnb_111k':'output_siren_inference_extbnb_devoted_pyramid_iteration_00111000.root',
    'extbnb_230k':'output_siren_inference_extbnb_devoted_pyramid_iteration_00230000.root',
    'extbnb_371k':'output_siren_inference_extbnb_devoted_pyramid_iteration_00371000.root',
    'extbnb_rare_monkey_124k':'output_siren_inference_extbnb_rare_monkey_iteraction_00124000.root',
    'corsika_188k':'output_siren_inference_mccorsika_different_yogurt_iteration_00188000.root',
    'corsika_151k':'output_siren_inference_mccorsika_colorful_feather_00151000.root',
    'good_sun_060k':'output_siren_inference_extbnb_good_sun_iteraction_00060000.root',
    'good_sun_115k':'output_siren_inference_extbnb_good_sun_iteraction_00115000.root',
    'colorful_haze_107k':'output_siren_inference_extbnb_colorful_haze_checkpoint_00107000.root',
    'colorful_haze_057k':'output_siren_inference_extbnb_colorful_haze_checkpoint_00057000.root',    
    'temp':'output_siren_inference_extbnb_aveposfix_111k.root'
}

hist_list = {
    ('siren_sinkhorn',100,0,0.3,";sinkhorn divergence (AU)"),
    ('ub_sinkhorn',   100,0,0.3,";sinkhorn divergence (AU)"),
    ('siren_unbsinkhorn',100,0,0.2,";unbalanced sinkhorn divergence (AU)"),
    ('ub_unbsinkhorn',   100,0,0.2,";unbalanced sinkhorn divergence (AU)"),
    ('diff_sinkhorn', 100,-0.3,0.3,";Siren - UB model sinkhorn divergence (AU)"),    
    ('diff_unbsinkhorn', 100,-0.2,0.2,";Siren - UB model unbalanced sinkhorn divergence (AU)"),    
    ('siren_fracerr',60,-1.0,5.0,";(predicted-observed)/observed"),
    ('ub_fracerr',   60,-1.0,5.0,";(predicted-observed)/observed"),
    ('siren_fracerr_remake',60,-1.0,5.0,";(predicted-observed)/observed"),
    ('ub_fracerr_remake',   60,-1.0,5.0,";(predicted-observed)/observed"),
    ('obs_pe_tot',  20,0,10e3,";total PE"),
    ('siren_pe_tot',20,0,10e3,";total PE"),
    ('ub_pe_tot',   20,0,10e3,";total PE"),
    ('siren_pe_tot_remake',20,0,10e3,";total PE"),
    ('ub_pe_tot_remake',   20,0,10e3,";total PE")
}

var_formula = {
    'siren_pe_tot_remake':'siren_pe_tot*{remake_factor:.1f}',
    'siren_fracerr_remake':"(siren_pe_tot*{remake_factor:.1f}-obs_pe_tot)/obs_pe_tot",
    #'ub_fracerr_remake':"(ub_pe_tot*{remake_factor:.1f}-obs_pe_tot)/obs_pe_tot",
    'ub_fracerr_remake':"(ub_pe_tot-obs_pe_tot)/obs_pe_tot",
    'ub_pe_tot_remake':'ub_pe_tot',
    'diff_sinkhorn':'(siren_sinkhorn-ub_sinkhorn)',
    'diff_unbsinkhorn':'(siren_unbsinkhorn-ub_unbsinkhorn)'
}

def make_histograms(hists,filename,filepath,outfile):
    outfile.cd()

    rfile = rt.TFile( filepath, 'open' )
    ttree = rfile.Get( 'siren_inference' )

    outfile.cd()

    outhists = {}

    remake_factor = 1.0
    if filename in remake_factors:
        remake_factor = remake_factors[filename]

    for var, nbins, xmin, xmax, title in hists:
        hname = f"h{var}_{filename}"
        h = rt.TH1D(hname,title,nbins,xmin,xmax)
        varform = var
        if var in var_formula:
            varform = var_formula[var].format(remake_factor=remake_factor)
        print(var,": ",varform," >> ",hname)
        ttree.Draw(f"{varform}>>{hname}")
        outhists[(var,filename)] = h

    return outhists


if __name__ == "__main__":


    plotfolder = './inference_plots'
    os.system(f'mkdir -p {plotfolder}')
    rout = rt.TFile("temp.root","recreate")
    hists = {}
    c = rt.TCanvas("c","c",1200,1000)
    for fname in filelist:
        fpath = filepaths[fname]
        filehists = make_histograms( hist_list, fname, fpath, rout )
        hists.update(filehists)
    c.Close()

    canvs = {}
    tlen_v = {}
    for fname in filelist:

        for var in variable_list:
            c = rt.TCanvas(f"c{var}_{fname}",f"{var}: {fname}",1200,1000) 
            c.cd(1).SetGridx(1)
            c.cd(1).SetGridy(1)
            try:
                hsiren = hists[(f'siren_{var}',fname)]
                hub    = hists[(f'ub_{var}',fname)]
            except:
                hsiren = hists[(var,fname)]
                hub    = None

            hsiren.SetLineColor(rt.kRed)
            hsiren.SetLineWidth(2)
            if hub is not None:
                hub.SetLineColor(rt.kBlue-4)
                hub.SetLineWidth(2)


            tlen = rt.TLegend(0.6,0.7,0.9,0.9)
            tlen.AddEntry(hsiren,"Siren Model", "L")
            if hub is not None:
                tlen.AddEntry(hub,"UB Light Model", "L")            

            hvars = [hsiren,hub]
            if var in ['pe_tot','pe_tot_remake']:
                hobs = hists[('obs_pe_tot',fname)]
                hobs.SetLineWidth(2)
                hobs.SetLineColor(rt.kBlack)
                tlen.AddEntry(hobs,"Observed","L")
                hvars.append(hobs)

            hmax = None
            maxval = 0.0
            for h in hvars:
                if h is None:
                    continue
                if maxval < h.GetMaximum():
                    maxval = h.GetMaximum()
                    hmax = h
            
            hmax.Draw("hist")
            if hub is not None:
                hub.Draw("histsame")
            hsiren.Draw("histsame")
            if var in ['pe_tot','pe_tot_remake']:
                hobs.Draw("histsame")

            tlen.Draw()
            tlen_v[(var,fname)] = tlen

            c.Update()
            canvs[(var,fname)] = c

            if var in ["diff_sinkhorn","diff_unbsinkhorn"]:
                zero_bin = hsiren.FindBin(0.0)
                nbetter = hsiren.Integral(1,zero_bin-1)
                frac_better = nbetter/hsiren.Integral()
                print(f"For {var}, fraction of events have a better sinkhorn divergence: {frac_better:.2f}")
            
            #c.SaveAs(f"{plotfolder}/{fname}_{var}.png")
            c.SaveAs(f"{plotfolder}/{fname}_{var}.pdf")

    print("[enter] to exit")
    input()
    




