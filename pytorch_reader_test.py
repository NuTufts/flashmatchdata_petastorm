import os,sys
import torch
from petastorm import make_reader, TransformSpec
from petastorm.pytorch import DataLoader

torch.manual_seed(1)

dataset_folder = 'file:///tmp/test_flash_dataset'

def _transform_row( row ):
    #print(row)
    result = {"coord":row["coord"],
              "feat":row["feat"],
              "flashpe":row["flashpe"],              
              "event":row["event"],
              "matchindex":row["matchindex"]}
    return result

transform = TransformSpec(_transform_row, removed_fields=['sourcefile','run','subrun','ancestorid'])
#transform = TransformSpec(_transform_row, removed_fields=[])
#reader = make_reader( dataset_folder, num_epochs=1, transform_spec=transform, seed=1, shuffle_rows=False )
#for row in reader:
#    print(row)

with DataLoader( make_reader(dataset_folder, num_epochs=1, transform_spec=transform, seed=1, shuffle_rows=True ),
#with DataLoader( make_reader(*dataset_list, num_epochs=2, transform_spec=transform, seed=1, shuffle_rows=False ),
                 batch_size=1 ) as loader:

    for batch_idx, row in enumerate(loader):
        print("BATCH[",batch_idx,"] ==================")
        print(" event: ",row['event']," matchindex: ",row['matchindex'])
        print(" coord: ",row['coord'].shape)
        print(" feat: ",row['feat'].shape)
        print(" flashpe: ",row['flashpe'].shape)

