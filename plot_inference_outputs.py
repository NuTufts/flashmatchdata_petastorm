import os,sys
import ROOT as rt

rt.gStyle.SetOptStat(0)

#filelist = ['corsika_382k','corsika_260k']
#filelist = ['extbnb_685k','extbnb_775k']
#filelist = ['corsika_382k','corsika_52kk']
#filelist = ['corsika_634k']
filelist = ['temp']

variable_list = ['sinkhorn','fracerr','pe_tot','fracerr_remake','pe_tot_remake']

remake_factors = {
    'corsika_382k':1.75,
    'corsika_260k':1.75,
    'corsika_524k':1.5,
    'corsika_634k':1.5,
    'extbnb_685k':1.3,
    'extbnb_775k':1.3,
    'temp':1.0
}

filepaths = {
    'corsika_260k':'output_siren_inference_lemon_snowflake_00260000.root',
    'corsika_382k':'output_siren_inference_mccorsika_lemon_snowflake_00382000.root',
    'corsika_524k':'output_siren_inference_mccorsika_lemon_snowflake_00524000.root',
    'corsika_634k':'output_siren_inference_mccorsika_lemon_snowflake_00634000.root',    
    'extbnb_685k':'output_siren_inference_desert_universe_checkpoint_iteration_00685000.root',
    'extbnb_775k':'output_siren_inference_desert_universe_checkpoint_iteration_00775000.root',
    'temp':'output_siren_inference_temp.root'
}

hist_list = {
    ('siren_sinkhorn',100,0,0.3,";sinkhorn divergence (AU)"),
    ('ub_sinkhorn',   100,0,0.3,";sinkhorn divergence (AU)"),
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
    'ub_fracerr_remake':"(ub_pe_tot*{remake_factor:.1f}-obs_pe_tot)/obs_pe_tot",
    'ub_pe_tot_remake':'ub_pe_tot',
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
            hsiren = hists[(f'siren_{var}',fname)]
            hub    = hists[(f'ub_{var}',fname)]

            hsiren.SetLineColor(rt.kRed)
            hsiren.SetLineWidth(2)
            hub.SetLineColor(rt.kBlue-4)
            hub.SetLineWidth(2)


            tlen = rt.TLegend(0.6,0.7,0.9,0.9)
            tlen.AddEntry(hub,"UB Light Model", "L")
            tlen.AddEntry(hsiren,"Siren Model", "L")

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
                if maxval < h.GetMaximum():
                    maxval = h.GetMaximum()
                    hmax = h
            
            hmax.Draw("hist")
            hub.Draw("histsame")
            hsiren.Draw("histsame")
            if var in ['pe_tot','pe_tot_remake']:
                hobs.Draw("histsame")

            tlen.Draw()
            tlen_v[(var,fname)] = tlen

            c.Update()
            canvs[(var,fname)] = c
            c.SaveAs(f"{plotfolder}/{var}_{fname}.png")

    print("[enter] to exit")
    input()
    




