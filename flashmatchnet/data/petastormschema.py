import numpy as np

import petastorm
from pyspark.sql.types import IntegerType, StringType
from petastorm.codecs import ScalarCodec, CompressedImageCodec, NdarrayCodec
from petastorm.unischema import dict_to_spark_row, Unischema, UnischemaField

"""
This module contains definitions and utilities 
"""

# This defines the table we will write to the database
# It's basically defining the columns of the database table,
#  listing the for each: the name, base type, shape of array, and how to pack/unpack it into bits
#  note the last pool in the tuble defining the column is whether its ok to have a missing value
#  for the column.
FlashMatchSchema = Unischema("FlashMatchSchema",[
    UnischemaField('sourcefile', np.string_, (), ScalarCodec(StringType()),  True),
    UnischemaField('run',        np.int32,   (), ScalarCodec(IntegerType()), False),
    UnischemaField('subrun',     np.int32,   (), ScalarCodec(IntegerType()), False),
    UnischemaField('event',      np.int32,   (), ScalarCodec(IntegerType()), False),
    UnischemaField('matchindex', np.int32,   (), ScalarCodec(IntegerType()), False),
    UnischemaField('ancestorid', np.int32,   (), ScalarCodec(IntegerType()), False),
    UnischemaField('coord',      np.int64,   (None,3), NdarrayCodec(), False),
    UnischemaField('feat',       np.float32, (None,3), NdarrayCodec(), False),    
    UnischemaField('flashpe',    np.float32, (None,32), NdarrayCodec(), False),
])
