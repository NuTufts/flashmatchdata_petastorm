import os,sys
import ROOT as rt

rt.gStyle.SetOptStat(0)

filelist = ['corsika','extbnb']
#filelist = ['extbnb']
filepaths = {
    'corsika':'output_siren_inference_lemon_snowflake_00133000.root',
    'extbnb':'output_siren_inference_desert_universe_checkpoint_iteration_00565000.root'
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
    ('ub_pe_tot',   20,0,10e3,";total PE")
}

var_formula = {
    'siren_pe_tot':'siren_pe_tot*3.0',
    'siren_fracerr_remake':f"(siren_pe_tot*3.0-obs_pe_tot)/obs_pe_tot",
    'ub_fracerr_remake':f"(ub_pe_tot*3.0-obs_pe_tot)/obs_pe_tot",
}

def make_histograms(hists,filename,filepath,outfile):
    outfile.cd()

    rfile = rt.TFile( filepath, 'open' )
    ttree = rfile.Get( 'siren_inference' )

    outfile.cd()

    outhists = {}

    for var, nbins, xmin, xmax, title in hists:
        hname = f"h{var}_{filename}"
        h = rt.TH1D(hname,title,nbins,xmin,xmax)
        varform = var
        if var in var_formula:
            varform = var_formula[var]
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
        for var in ['sinkhorn','fracerr','pe_tot','fracerr_remake']:
            c = rt.TCanvas(f"c{var}_{fname}",f"{var}: {fname}",1200,1000) 
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
            if var=='pe_tot':
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
            if var=='pe_tot':
                hobs.Draw("histsame")

            tlen.Draw()
            tlen_v[(var,fname)] = tlen

            c.Update()
            canvs[(var,fname)] = c
            c.SaveAs(f"{plotfolder}/{var}_{fname}.png")

    print("[enter] to exit")
    input()
    




