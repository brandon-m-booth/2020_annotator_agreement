import os
import sys
import glob
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'util')))
import util
import agreement_metrics as agree

def TimeWarpAnnotations(data_root_path, sample_rate=1):
   max_warp_seconds = 3
   hz_suffix = str(sample_rate)+'hz'

   data_dict = util.LoadData(data_root_path, sample_rate)
   for task in data_dict.keys():
      # Apply DTW
      anno_df = data_dict[task]['raw']['annotations']
      ot_df = data_dict[task]['raw']['objective_truth']
      dtw_anno_df = agree.DTWReference(anno_df.iloc[:,1:], ot_df.iloc[:,1], max_warp_distance=max_warp_seconds*sample_rate)

      # Output DTW annotations
      dtw_output_task_path = os.path.join(data_root_path, task, 'annotations_'+hz_suffix+'_dtw_aligned')
      if not os.path.isdir(dtw_output_task_path):
         os.makedirs(dtw_output_task_path)

      for anno_col in dtw_anno_df.columns:
         output_task_file = os.path.join(dtw_output_task_path, task+'_'+anno_col+'.csv')
         dtw_single_anno_df = pd.concat((anno_df.iloc[:,0], dtw_anno_df[anno_col]), axis=1)
         dtw_single_anno_df.to_csv(output_task_file, index=False, header=True)

      # Plot
      num_annos = len(anno_df.columns)-1
      subplot_rows = int(math.floor(math.sqrt(num_annos)))
      subplot_cols = math.ceil(float(num_annos)/subplot_rows)
      fig, axs = plt.subplots(subplot_rows, subplot_cols)
      for anno_idx in range(len(anno_df.columns)-1):
         anno_col = anno_df.columns[anno_idx+1] # Skip time col
         anno = anno_df[anno_col]
         ot = ot_df.iloc[:,1]
         axs_col = anno_idx%subplot_cols
         axs_row = int(math.floor(anno_idx/subplot_cols))
         axs[axs_row, axs_col].plot(ot_df.iloc[:,0], ot_df.iloc[:,1], 'm-')
         axs[axs_row, axs_col].plot(ot_df.iloc[:,0], dtw_anno_df.iloc[:,anno_idx], 'k-')
         axs[axs_row, axs_col].plot(ot_df.iloc[:,0], anno_df[anno_col].values, 'b--')
         axs[axs_row, axs_col].title.set_text(anno_col)
      fig.suptitle(task + '  DTW')
   plt.show()
         
   return

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--data_root_path', required=True, help='Path to the parent folder of the green intensity data set')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   TimeWarpAnnotations(args.data_root_path)
