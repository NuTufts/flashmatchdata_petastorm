import os,sys
from datetime import datetime

fbad = open("badlist.txt",'r')
fbadlines = fbad.readlines()

bad_fileids = []
for badfile in fbadlines:
    badfile = badfile.strip()
    #print(badfile)
    badbase = os.path.basename(badfile)
    fileidno = int(badbase.split("-")[0].split("fileid")[-1])
    print(fileidno,": ",badbase)
    bad_fileids.append(fileidno)

now = datetime.now()
datestr = now.strftime("%Y%m%d")
bad_runid_filelist = f"runid_mcc9_v13_bnbnue_corsika_{datestr}.list"
with open(bad_runid_filelist,'w') as f:
    for fileidno in bad_fileids:
        print(fileidno,file=f)

    
