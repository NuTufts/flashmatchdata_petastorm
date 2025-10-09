import os,sys

import ROOT as rt
rt.gStyle.SetOptStat(0)
rt.gStyle.SetPadRightMargin(0.15)


#finput = "mccorsika_different_yogurt_404k.root"
#label  = "corsika_404k"

finput = "extbnb_devoted_pyramid_371k.root"
label  = "extbnb_devoted_pyramid_371k"


plot_path = "./scan_plots/"
os.system(f'mkdir -p {plot_path}')

rfile = rt.TFile(finput)

temp = rt.TFile("temp.root","recreate")

clist = []
hlist = []
for ix in range(5):
    c = rt.TCanvas(f"c{ix}",f"x[{ix}]",8*600,4*400)
    c.Divide(8,4)
    for ipmt in range(32):
        hfvis = rfile.Get(f"hyz_pmt{ipmt}_x{ix}")
        hdist = rfile.Get(f"hyz_dist_pmt{ipmt}_x{ix}")
        c.cd( int(ipmt/8)*8 + int(ipmt%8) + 1 ).SetLogz(1)
        hfvis.GetZaxis().SetRangeUser(1.0e-5,1.0)
        hfvis.Draw("colz")
        hdist.Draw("cont1same")
        hlist.append( hfvis )
        hlist.append( hdist )
    c.Update()
    c.SaveAs(f"{plot_path}/fvis_{label}_ix{ix}.png")
    clist.append(c)
    
print("[enter] to quit")
input()
