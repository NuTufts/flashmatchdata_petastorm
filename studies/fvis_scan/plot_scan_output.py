import os,sys

import ROOT as rt
rt.gStyle.SetOptStat(0)
rt.gStyle.SetPadRightMargin(0.15)


#finput = "mccorsika_different_yogurt_404k.root"
#label  = "corsika_404k"

#finput = "extbnb_devoted_pyramid_371k.root"
#label  = "extbnb_devoted_pyramid_371k"

#finput = "mccorsika_colorful_feather_151k.root"
#label  = "corsika_151k"

#finput = "extbnb_rare_monkey_169k.root"
#label  = "extbnb_rare_monkey_169k"

#finput = "mccorsika_stoic_hill_136k.root"
#label  = "mccorsika_stoic_hill_136k"

#finput = "good_sun_060k.root"
#label  = "good_sun_060k"

#finput = "colorful_haze_030k.root"
#label  = "colorful_haze_030k"

#finput = "deft_universe_075k.root"
#label  = "deft_universe_075k"

#finput = "deft_universe_745k.root"
#label  = "deft_universe_745k"

#finput = "fearless_tree_032k.root" # siren model, full extbnb data, unbalanced loss, w0=3.0
#label  = "fearless_tree_032k"

#finput = "stellar_dream_044k.root" # siren model, full extbnb data, unbalanced loss, w0=0.3
#label  = "stellar_dream_044k"

finput = "smooth_wildflower_034k.root" # siren model, full extbnb data, unbalanced loss, w0=0.03
label  = "smooth_wildflower_034k"      # siren model, full extbnb data, unbalanced loss, e0=0.03

use_logz = False


plot_path = "./scan_plots/"
os.system(f'mkdir -p {plot_path}')

rfile = rt.TFile(finput)

temp = rt.TFile("temp.root","recreate")

clist = []
hlist = []
max_y = 1.0
for ix in range(5):
    c = rt.TCanvas(f"c{ix}",f"x[{ix}]",1000,1500)
    c.Divide(4,8)
    for ipmt in range(32):
        hfvis = rfile.Get(f"hyz_pmt{ipmt}_x{ix}")
        hdist = rfile.Get(f"hyz_dist_pmt{ipmt}_x{ix}")
        c.cd( int(ipmt/8)*8 + int(ipmt%8) + 1 ).SetLogz(use_logz)
        if use_logz:
            hfvis.GetZaxis().SetRangeUser(1.0e-5,10.0)
        else:
            hfvis.GetZaxis().SetRangeUser(1.0e-5,max_y)
        hfvis.Draw("colz")
        hdist.Draw("cont1same")
        hlist.append( hfvis )
        hlist.append( hdist )
    c.Update()
    c.SaveAs(f"{plot_path}/fvis_{label}_ix{ix}.png")
    clist.append(c)
    max_y *= 0.4
    
print("[enter] to quit")
input()
