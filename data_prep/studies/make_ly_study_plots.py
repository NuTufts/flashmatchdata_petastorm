import os,sys
import ROOT as rt

rt.gStyle.SetOptStat(0)


os.system('mkdir -p ./plots')
samples = ['extbnb','corsika']

files = {
    'extbnb':'out_lyanalysis_run3extbnb.root',
    'corsika':'out_lyanalysis_mccorsika.root'
}

rfiles = {}
ttrees = {}

temp = rt.TFile('temp.root','recreate')

for sample in samples:
    rfiles[sample] = rt.TFile(files[sample],'open')
    ttrees[sample] = rfiles[sample].Get("lyana")

qxcuts = [0,50,100,200]

h_petot_v_qsum_v = {}
c_petot_v_qsum = rt.TCanvas("c_petot_v_qsum","Total PE vs. TPC Charge Sum",500*len(qxcuts),500*len(samples))
c_petot_v_qsum.Divide(len(qxcuts),2)

for isample,sample in enumerate(samples):
    for ix,x in enumerate(qxcuts):
        xmin = 50.0*ix
        xmax = 50.0*(ix+1)
        xcut = f"qx>{xmin:.1f} && qx<{xmax:.1f}"
        c_petot_v_qsum.cd(len(qxcuts)*isample + ix+1)
        hname = f"h_petot_v_qsum_{sample}_{ix}"
        h = rt.TH2D(hname,f"{sample}: {xcut};charge sum/50000;pe total/5000",50,0,3.0,50,0,1.0)
        h_petot_v_qsum_v[(sample,ix)] = h
        ttrees[sample].Draw(f"petot/5000.0:qsum/50000>>{hname}",xcut,"colz")
c_petot_v_qsum.Update()
c_petot_v_qsum.SaveAs("plots/c_pe_tot_v_qsum.png")

print("[enter]")
input()
