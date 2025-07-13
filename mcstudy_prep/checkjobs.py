import os,sys

# simple check that jobs finished ok.

# filelist
input_list = "nue_corsika_input_filesets.txt"
fid_mcinput = {}
with open(input_list,'r') as f:
    ll = f.readlines()
    x = 0
    for l in ll:
        info = l.strip().split()
        fileid = int(info[0])
        mcinfoname = os.path.basename(info[-1])
        #print(fileid,": ",mcinfoname)
        fid_mcinput[mcinfoname] = [fileid,x]
        x += 1

def is_in_train_split_nue_corsika(fid):
    if fid<2037:
        # training split
        return True
    else:
        # validation split
        return False

# we look for the ID number
finished_fid = []
dbfolders=["/cluster/tufts/wongjiradlabnu/twongj01/dev_petastorm/datasets/flashmatch_mc_data_v3_training/",
           "/cluster/tufts/wongjiradlabnu/twongj01/dev_petastorm/datasets/flashmatch_mc_data_v3_validation/"]

for dbfolder in dbfolders:
    dbcontents = os.listdir(dbfolder)

    for x in dbcontents:
        x = x.strip()
        if "sourcefile" not in x:
            continue

        sourcefile = x.split("=")[-1].strip()
        #print("x: ",sourcefile)

        if sourcefile in fid_mcinput:
            fid,lineno = fid_mcinput[sourcefile]
            print("finished: ",sourcefile," ",(fid,lineno))        
            #finished.append((fid,lineno))
            finished_fid.append(fid)


print("Number of file IDs finished: ",len(finished_fid))
rerun = []
nmatched = 0
for mcinfo,xx in fid_mcinput.items():
    fid = xx[0]
    lineno = xx[1]
    if fid not in finished_fid:
        #print("rerun ",xx)
        rerun.append(xx)
    else:
        nmatched += 1
        
print("Number of file IDs to rerun: ",len(rerun))
print("nmatched: ",nmatched)

rerun_training = open("rerun_list_training.txt",'w')
rerun_valid    = open("rerun_list_validation.txt",'w')

for (fid,lineno) in rerun:
    if is_in_train_split_nue_corsika( fid ):
        print(lineno," ",fid,file=rerun_training)
    else:
        print(lineno," ",fid,file=rerun_valid)

rerun_training.close()
rerun_valid.close()
