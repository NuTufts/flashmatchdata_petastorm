import os,sys

nue_corsika_folder = "/cluster/tufts/wongjiradlab/larbys/data/db/mcc9_v13_bnbnue_corsika/stage1/"

pfind = os.popen("find %s | grep root"%(nue_corsika_folder))
lfind = pfind.readlines()

fset = {} # key fileid, value: dict {"opreco":x.root,"mcinfo":x.root,"larcvtruth":x.root}

for l in lfind:
    l = l.strip()
    info = l.split("/")
    fileid = int(info[-3])*100+int(info[-2])
    if fileid not in fset:
        print("creating fileid key=",fileid)
        fset[fileid] = {}
    bname = os.path.basename(l)
    for ftype in ["opreco","mcinfo","larcvtruth"]:
        if ftype in bname:
            fset[fileid][ftype] = l
            break

fid_v = list(fset.keys())
fid_v.sort()
print("print filesets for ",len(fid_v)," fileid keys")
        
with open('nue_corsika_input_filesets_v2.txt','w') as f:
    for xx,fid in enumerate(fid_v):
        fdict = fset[fid]
        opreco = fdict['opreco']
        mcinfo = fdict['mcinfo']
        lcvtruth = fdict['larcvtruth']
        print("%d %s %s %s"%(fid,lcvtruth,opreco,mcinfo),file=f)



