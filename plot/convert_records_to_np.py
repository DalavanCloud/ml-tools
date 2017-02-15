'''
python convert_records_to_np.py record.txt
'''
import os
import sys
import numpy as np

if __name__ == '__main__':
  records = []
  fn = sys.argv[1]
  for line in open(fn):
    line = line.strip()
    ss = line.split(' ')
    records.append([int(ss[0]), float(ss[1]), float(ss[2])])
  record_array = np.asarray(records, dtype=np.float32)
  np.save('record', record_array)