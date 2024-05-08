import os,sys
from pyspark.sql import SparkSession

dbfolder="/cluster/tufts/wongjiradlabnu/twongj01/dev_petastorm/datasets/flashmatch_mc_data_v3_training/"
dataset_url="file://"+dbfolder

spark = SparkSession.builder.config('spark.driver.memory', '2g').master('local[2]').getOrCreate()


df = spark.read.parquet(dataset_url)
df.registerTempTable("opmodel")

sourcefiles = [ r['sourcefile'] for r in df.select("sourcefile").distinct().collect() ]

for s in sourcefiles:
    s = s.strip()
    fid = int(s.split("SubRun")[-1].split(".")[0])

    if fid>=2037:
        print("we want to drop this partition from the training table: ",s)
        print("[Do it?]")
        command = "ALTER TABLE opmodel DROP IF EXISTS PARTITION (sourcefile='%s') PURGE"%(s)
        doit = input()
        if doit in ['y','Y']:
            print("sending: ",command)
            spark.sql(command)
            #spark.sql("ALTER TABLE tmp DROP IF EXISTS PARTITION (month='2015-02-01') PURGE")

        

#print(dataframe.getNumPartitions())
