# Utility function library
import os
import pdb
import glob
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import pytz
import agreement_metrics as agree

# Find and load the shades of green data
def LoadGreenData(data_root_path, sample_rate):
   hz_suffix = str(sample_rate)+'hz'
   data_dict = {}
   task_dirs = glob.glob(os.path.join(data_root_path, '*'))
   for task_dir in task_dirs:
      task_name = os.path.basename(task_dir)
      data_dict[task_name] = {}

      # Find and load the truth file
      task_ot_file = [x for x in glob.glob(os.path.join(task_dir, 'objective_truth', '*.csv')) if hz_suffix in x][0]
      task_ot_df = pd.read_csv(task_ot_file)
      task_ot_df.columns = (task_ot_df.columns[0], 'Objective Truth')

      # Find and load the annotations
      anno_types = [('raw', 'annotations_'+hz_suffix), ('aligned_shifted', 'annotations_'+hz_suffix+'_shift_aligned'), ('aligned_dtw', 'annotations_'+hz_suffix+'_dtw_aligned')]
      for anno_type in anno_types:
         anno_type_name = anno_type[0]
         anno_type_folder_name = anno_type[1]
         if not os.path.isdir(os.path.join(task_dir, anno_type_folder_name)):
            continue
         task_anno_files = glob.glob(os.path.join(task_dir, anno_type_folder_name, '*.csv'))
         if len(task_anno_files) == 0:
            continue
         task_anno_dfs = []
         for task_anno_file in task_anno_files:
            task_anno_df = pd.read_csv(task_anno_file)
            anno_name = os.path.basename(task_anno_file).split('_')[1].split('.')[0]
            task_anno_df.columns = (task_anno_df.columns[0], anno_name)
            task_anno_dfs.append(task_anno_df)
         task_anno_data_dfs = [df.iloc[:,1] for df in task_anno_dfs]
         task_anno_df = pd.concat(task_anno_data_dfs, axis=1) # Concat the annotation data series
         task_anno_df = pd.concat((task_anno_dfs[0].iloc[:,0], task_anno_df), axis=1) # Concat the time column

         # Clip the annotations to make the objective truth time domain
         min_ot_time = task_ot_df.iloc[:,0].min()
         max_ot_time = task_ot_df.iloc[:,0].max()
         anno_clip_mask = np.logical_and(task_anno_df.iloc[:,0] >= min_ot_time, task_anno_df.iloc[:,0] <= max_ot_time)
         task_anno_df = task_anno_df.loc[anno_clip_mask,:]

         if task_ot_df.shape[0] > task_anno_df.shape[0]: # Time aligned signals need a clipped ot signal
            min_anno_time = task_anno_df.iloc[:,0].min()
            max_anno_time = task_anno_df.iloc[:,0].max()
            ot_clip_mask = np.logical_and(task_ot_df.iloc[:,0] >= min_anno_time, task_ot_df.iloc[:,0] <= max_anno_time)
            task_ot_df = task_ot_df.loc[ot_clip_mask,:]
            
         task_anno_df = task_anno_df.iloc[:,1:] # Remove time column
         data_dict[task_name][anno_type_name] = {'annotations': task_anno_df, 'objective_truth': task_ot_df}
   return data_dict

def LoadRecolaData(data_root_path, cache_file=None):
   data_dict = {}
   if cache_file is not None and os.path.isfile(cache_file):
      with open(cache_file, 'rb') as f:
         data_dict = pickle.load(f)
   else:
      anno_files = glob.glob(os.path.join(data_root_path, '*.csv'))
      for anno_file in anno_files:
         anno_task = os.path.basename(anno_file).split('.')[0]
         df = pd.read_csv(anno_file, sep=';')
         df.index = [datetime.fromtimestamp(t, pytz.utc) for t in df.iloc[:,0]]
         df = df.iloc[:,1:] # Remove the first time column
         dtw_df = agree.DTWReference(df, df.iloc[:,0], max_warp_distance=75)
         df = df.resample('1s').mean().interpolate()
         dtw_df = dtw_df.resample('1s').mean().interpolate()
         data_dict[anno_task] = {'raw': {}, 'dtw': {}}
         data_dict[anno_task]['raw']['annotations'] = df
         data_dict[anno_task]['raw']['objective_truth'] = None
         data_dict[anno_task]['dtw']['annotations'] = dtw_df
         data_dict[anno_task]['dtw']['objective_truth'] = None
   if cache_file is not None and not os.path.isfile(cache_file):
      with open(cache_file, 'wb') as f:
         pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)
   return data_dict

def LoadDEAMData(data_root_path):
   anno_df = data_dict['taska']['dtw']['annotations']
   ot_df = data_dict['taska']['dtw']['objective_truth']
   return data_dict
