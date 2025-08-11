import os,sys
import ROOT as rt

rt.gStyle.SetOptStat(0)

input_file = "../test_match.root"

tfile = rt.TFile( input_file )

flashmatch = tfile.Get("flashmatch")

nentries = flashmatch.GetEntries()

out_temp = rt.TFile("temp_vis_flashpredictions.root","recreate")

predicted_factor = 1000.0

c = rt.TCanvas("c","",2000,600)
c.Divide(2,1)

MATCH_TYPE_NAMES = {-1:"undefined match",
    0:"Anode Match",
    1:"Cathode Match",
    2:"CRT Track Match",
    3:"CRT Hit Match",
    4:"Track-to-flash Match"}

for ientry in range(nentries):
    flashmatch.GetEntry(ientry)

    match_type = flashmatch.match_type
    match_name = MATCH_TYPE_NAMES[match_type]

    hobserved  = rt.TH1D("hobserved_entry%d"%(ientry),f"{match_name}",32,0,32)
    hobserved.SetLineWidth(2)
    hobserved.SetLineColor(rt.kBlack)

    hpredicted = rt.TH1D("hpredicted_entry%d"%(ientry),f"{match_name}",32,0,32)
    hpredicted.SetLineWidth(2)
    hpredicted.SetLineColor(rt.kRed)

    hobserved_normed  = hobserved.Clone("hobserved_norm_entry%d"%(ientry))
    hpredicted_normed = hpredicted.Clone("hpredicted_norm_entry%d"%(ientry))

    x_predicted_totpe = flashmatch.predicted_pe_total*predicted_factor

    for ipmt in range(32):

        hobserved.SetBinContent( ipmt+1, flashmatch.opflash_pe_v[ipmt] )
        if flashmatch.opflash_pe_total>0.0:
            hobserved_normed.SetBinContent(ipmt+1,  flashmatch.opflash_pe_v[ipmt]/flashmatch.opflash_pe_total  )
        else:
            hobserved_normed.SetBinContent(ipmt+1, 0.0 )

        x_predicted_pe = flashmatch.predicted_pe_v[ipmt]*predicted_factor
        hpredicted.SetBinContent(ipmt+1,x_predicted_pe)
        if flashmatch.predicted_pe_total>0.0:
            hpredicted_normed.SetBinContent(ipmt+1, x_predicted_pe/x_predicted_totpe )
        else:
            hpredicted_normed.SetBinContent(ipmt+1, 0.0)

    c.cd(1)

    if ( hobserved.GetMaximum()>hpredicted.GetMaximum()):
        hobserved.Draw("histE1")
        hpredicted.Draw("histsame")
    else:
        hpredicted.Draw("hist")
        hobserved.Draw("histE1same")


    c.cd(2)

    if ( hobserved_normed.GetMaximum()>hpredicted_normed.GetMaximum()):
        hmax_norm = hobserved_normed
    else:
        hmax_norm = hpredicted_normed

    hmax_norm.Draw("hist")
    hobserved_normed.Draw("histsame")
    hpredicted_normed.Draw("histsame")

    c.Update()
    c.Draw()

    print("[enter] to continue")
    input()
