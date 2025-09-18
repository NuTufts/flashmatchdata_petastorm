import os,sys
import ROOT as rt
from math import fabs

rt.gStyle.SetOptStat(0)
rt.gStyle.SetPadRightMargin(0.15)

#input_file="out_model_anaysis.root" # cosmic-universe-230
#input_file="images/tmp/out_model_anaysis.root" # 231
input_file="images/northern-sea-231/out_model_analysis_northern_sea_231_iter625000.root"

rfile = rt.TFile(input_file,"open")
hreco  = {"x":rfile.Get("hpesum"),
          "z":rfile.Get("hpesum_z")}
htruth = {"x":rfile.Get("hpesum_true"),
          "z":rfile.Get("hpesum_z_true")}

hmades = []
c_v = []

rebinfactor_x = 4
rebinfactor_y = 2

for var in ["x","z"]:


    cx = rt.TCanvas("c%s"%(var),"PE Sum versus (%s) from cathode"%(var),1800,500)
    
    cx.Divide(3,1)
    cx.cd(1)
    projx_zx_true = htruth[var].Project3D("zx")
    projx_zx_true.RebinX(rebinfactor_x)
    projx_zx_true.RebinY(rebinfactor_y)
    
    projx_zx_true.SetTitle("PE_{sum} vs. q-weighted #bar{%s}: True"%(var))
    true_norm = 1.0/projx_zx_true.Integral()
    projx_zx_true.Scale(true_norm)
    projx_zx_true.Draw("colz")
    projx_zx_true.GetXaxis().SetLabelSize(0.05)
    projx_zx_true.GetXaxis().SetTitleSize(0.05)
    projx_zx_true.GetYaxis().SetLabelSize(0.05)
    projx_zx_true.GetYaxis().SetTitleSize(0.05)

    cx.cd(2)
    projx_zx_reco = hreco[var].Project3D("zx")
    projx_zx_reco.RebinX(rebinfactor_x)
    projx_zx_reco.RebinY(rebinfactor_y)
    
    
    projx_zx_reco.SetTitle("PE_{sum} vs. q-weighted #bar{%s}: Reco"%(var))
    reco_norm = 1.0/projx_zx_reco.Integral()
    projx_zx_reco.Scale(reco_norm)
    projx_zx_reco.Draw("colz")
    projx_zx_reco.GetXaxis().SetLabelSize(0.05)
    projx_zx_reco.GetXaxis().SetTitleSize(0.05)
    projx_zx_reco.GetYaxis().SetLabelSize(0.05)
    projx_zx_reco.GetYaxis().SetTitleSize(0.05)

    cx.cd(3)
    projx_zx_diff = projx_zx_reco.Clone("projx_zx_diff_%s"%(var))
    projx_zx_diff.Add( projx_zx_true, -1.0 )
    projx_zx_diff.SetTitle("Reco-True")
    zmax = projx_zx_diff.GetMaximum()
    zmin = projx_zx_diff.GetMinimum()
    projx_zx_diff.GetXaxis().SetLabelSize(0.05)
    projx_zx_diff.GetXaxis().SetTitleSize(0.05)
    projx_zx_diff.GetYaxis().SetLabelSize(0.05)
    projx_zx_diff.GetYaxis().SetTitleSize(0.05)
    projx_zx_diff.Draw("colz")
    if fabs(zmax)>fabs(zmin):
        projx_zx_diff.GetZaxis().SetRangeUser(-fabs(zmax), fabs(zmax) )
    else:
        projx_zx_diff.GetZaxis().SetRangeUser(-fabs(zmin), fabs(zmin) )
    projx_zx_diff.Draw("colz")

    hmades += [projx_zx_true,projx_zx_reco,projx_zx_diff]
    c_v.append( cx )

    cx.Update()

print("[enter] to continue")
input()
