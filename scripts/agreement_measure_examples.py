import os
import pdb
import sys
import glob
import math
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tikzplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, spearmanr, kendalltau

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'util')))
import util
import agreement_metrics as agree

def AgreementMeasureExamples(data_root_path, output_path, show_plots=True, sample_rate=1):
   if not os.path.isdir(output_path):
      os.makedirs(output_path)

   data_dict = util.LoadData(data_root_path, sample_rate)
   print('Computing examples of drawbacks to existing measures...')

   # Pearson
   anno = data_dict['TaskA']['aligned_shifted']['annotations'].loc[:,'ann7'].values
   ot = data_dict['TaskA']['aligned_shifted']['objective_truth'].iloc[:,1].values
   poly_feats = PolynomialFeatures(degree=3)
   poly_feats_fit = poly_feats.fit_transform(anno.reshape(-1,1))
   poly_reg = LinearRegression()
   poly_reg.fit(poly_feats_fit, ot)
   ot_warped = poly_reg.predict(poly_feats.fit_transform(ot.reshape(-1,1)))
   pearson_self = pearsonr(ot, ot)[0]
   pearson_distort = pearsonr(ot, ot_warped)[0]
   print('Pearson example, ann7:')
   print('  Self correlation: %f'%(pearson_self))
   print('  Distorted correlation: %f'%(pearson_distort))

   # CCC
   ccc_self = agree._CCCHelper(ot, ot)
   ccc_distort = agree._CCCHelper(ot, ot_warped)
   print('CCC example, ann7:')
   print('  Self correlation: %f'%(ccc_self))
   print('  Distorted correlation: %f'%(ccc_distort))

   # Spearman and Kendall
   anno = 10*[4]+3*[0]+8*[4.1]
   anno_distort = 10*[4]+3*[0]+8*[3.9]
   spearman_self = spearmanr(anno, anno)[0]
   spearman_distort = spearmanr(anno, anno_distort)[0]
   kendall_self = kendalltau(anno, anno)[0]
   kendall_distort = kendalltau(anno, anno_distort)[0]
   print('Spearman example:')
   print('  Self correlation: %f'%(spearman_self))
   print('  Distorted correlation: %f'%(spearman_distort))
   print('Kendall example:')
   print('  Self correlation: %f'%(kendall_self))
   print('  Distorted correlation: %f'%(kendall_distort))
   fig, axs = plt.subplots(1,2)
   axs[0].plot(range(len(anno)), anno, 'b-')
   axs[1].plot(range(len(anno_distort)), anno_distort, 'r-')
   axs[0].axes.set_ylim(-1,5)
   axs[1].axes.set_ylim(-1,5)
   axs[0].title.set_text('Example Annotation')
   axs[1].title.set_text('Similar Annotation')
   tikzplotlib.save(os.path.join(output_path, 'agreement_examples_spearmankendall.tex'), figure=fig, axis_width='\\figureWidth', textsize=12)

   print('Simulating annotations and computing agreement measures....')
   sim_sig, sim_axs = plt.subplots(1,1)
   sim_anno = np.array([8,8,8,7,5.2,3,2.5,2,1.8,1.7,1.6,1.5,1.7,1.9,2.2,3,4,5.5,6.8,7.9,9.5,10.5,11,11.2,11.3,11.3,11.3,11.3,11.2,11,10.7,10.4,10,9,7.5,7.3,7.2,7.2,7.2,7.2])

   sim_anno2 = sim_anno.copy()
   sim_anno2[16:] *= 1.5
   sim_anno2 += 2

   sim_axs.plot(range(len(sim_anno)), sim_anno, 'r-')
   sim_axs.plot(range(len(sim_anno2)), sim_anno2, 'b--')
   sim_axs.title.set_text('Simulated annotations')

   if show_plots:
      plt.show()
         
   return

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--data_root_path', required=True, help='Path to the parent folder of the green intensity data set')
   parser.add_argument('--output_path', required=True, help='Output folder path')
   parser.add_argument('--show_plots', required=False, action='store_true')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   AgreementMeasureExamples(args.data_root_path, args.output_path, args.show_plots)
