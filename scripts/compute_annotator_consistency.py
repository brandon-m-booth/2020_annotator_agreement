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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'util')))
import util

def PlotBackgroundBars(ax, x_vals, y_labels, y_lim=(-1000,1000), y_vals=None):
   true_color = "#C9E0BC"
   false_color = "#C6B8A1"
   line_true_color = "#0077FF"
   line_false_color = "#BF4E59"
   avg_x_width = np.mean(np.diff(x_vals))
   i = 0
   while i < len(x_vals):
      start_idx = i
      while i < len(x_vals) and y_labels[i] == y_labels[start_idx]:
         i += 1
      last_idx = i-1
      left_x = x_vals[start_idx]-(x_vals[start_idx]-x_vals[start_idx-1])/2.0 if start_idx > 0 else x_vals[start_idx]-avg_x_width
      right_x = x_vals[last_idx]+(x_vals[last_idx+1]-x_vals[last_idx])/2.0 if last_idx < len(x_vals)-1 else x_vals[last_idx] + avg_x_width
      color = true_color if y_labels[start_idx] else false_color
      bg_rect = patches.Rectangle((left_x,y_lim[0]), right_x-left_x,y_lim[1]-y_lim[0], linewidth=1, facecolor=color)
      #ax.add_patch(bg_rect)
      if y_vals is not None:
         line_color = line_true_color if y_labels[start_idx] else line_false_color
         clipped_last_idx = last_idx+2 if last_idx+2 < len(x_vals) else len(x_vals)-1
         ax.plot(x_vals[start_idx:clipped_last_idx], y_vals[start_idx:clipped_last_idx], c=line_color, marker='o', markersize=3, zorder=2)
   return

def ComputeAnnotatorConsistency(data_root_path, output_path, show_plots=True, sample_rate=1):
   if not os.path.isdir(output_path):
      os.makedirs(output_path)

   data_dict = util.LoadData(data_root_path, sample_rate)
   for task in data_dict.keys():
      anno_types = data_dict[task].keys()
      print('Processing task: '+task)
      for anno_type in anno_types:
         anno_df = data_dict[task][anno_type]['annotations']
         ot_df = data_dict[task][anno_type]['objective_truth']
         num_annos = len(anno_df.columns)-1
         #subplot_rows = int(math.floor(math.sqrt(num_annos)))
         #subplot_cols = math.ceil(float(num_annos)/subplot_rows)
         subplot_rows = 2
         subplot_cols = 5
         fig, axs = plt.subplots(subplot_rows, subplot_cols, sharex=True, sharey=True, figsize=(16,6))
         fig_diff, axs_diff = plt.subplots(subplot_rows, subplot_cols, sharex=True, sharey=True, figsize=(16,6))
         fig_anno, ax_anno = plt.subplots(subplot_rows, subplot_cols, sharex=True, sharey=True, figsize=(16,6))
         fig_anno_avg, ax_anno_avg = plt.subplots(1,1)
         avg_conf_diff_mask = None
         sorted_anno_cols = sorted(anno_df.columns[1:], key=lambda anno_col: int(anno_col[3:]))
         for anno_idx in range(len(sorted_anno_cols)):
            anno_col = sorted_anno_cols[anno_idx]
            anno = anno_df[anno_col]
            ot = ot_df.iloc[:,1]
            anno_diff = anno.diff()
            anno_diff[np.abs(anno_diff) < 1e-5] = 0
            ot_diff = ot.diff()
            ot_diff[np.abs(ot_diff) < 1e-5] = 0

            # Generate confusion matrices for the four quadrants
            cpal = sns.cubehelix_palette(len(anno))
            cmap = sns.cubehelix_palette(len(anno), as_cmap=True)
            conf_mat = np.zeros((2,2)).astype(int)
            conf_diff_mat  = np.zeros((2,2)).astype(int)
            signed_anno = np.sign(anno.values-0.5)
            signed_ot = np.sign(ot.values-0.5)
            signed_anno_diff = np.sign(anno_diff.values)[1:] # Skip first NaN
            signed_ot_diff = np.sign(ot_diff.values)[1:]
            conf_mat[0,0] = np.sum(np.logical_and(signed_anno > 0, signed_ot > 0))
            conf_mat[1,1] = np.sum(np.logical_and(signed_anno < 0, signed_ot < 0))
            conf_mat[0,1] = np.sum(np.logical_and(signed_anno > 0, signed_ot < 0))
            conf_mat[1,0] = np.sum(np.logical_and(signed_anno < 0, signed_ot > 0))
            conf_diff_mat[0,0] = np.sum(np.logical_and(signed_anno_diff > 0, signed_ot_diff > 0))
            conf_diff_mat[1,1] = np.sum(np.logical_and(signed_anno_diff < 0, signed_ot_diff < 0))
            conf_diff_mat[0,1] = np.sum(np.logical_and(signed_anno_diff > 0, signed_ot_diff < 0))
            conf_diff_mat[1,0] = np.sum(np.logical_and(signed_anno_diff < 0, signed_ot_diff > 0))
            tp_rect = patches.Rectangle((0.5,0.5),0.5,0.5,linewidth=1,facecolor=cpal[conf_mat[0,0]])
            fn_rect = patches.Rectangle((0.5,0.0),0.5,0.5,linewidth=1,facecolor=cpal[conf_mat[1,0]])
            tn_rect = patches.Rectangle((0.0,0.0),0.5,0.5,linewidth=1,facecolor=cpal[conf_mat[1,1]])
            fp_rect = patches.Rectangle((0.0,0.5),0.5,0.5,linewidth=1,facecolor=cpal[conf_mat[0,1]])
            tp_rect_diff = patches.Rectangle((0.0,0.0),0.5,0.5,linewidth=1,facecolor=cpal[conf_diff_mat[0,0]])
            fn_rect_diff = patches.Rectangle((0.0,-0.5),0.5,0.5,linewidth=1,facecolor=cpal[conf_diff_mat[1,0]])
            tn_rect_diff = patches.Rectangle((-0.5,-0.5),0.5,0.5,linewidth=1,facecolor=cpal[conf_diff_mat[1,1]])
            fp_rect_diff = patches.Rectangle((-0.5,0.0),0.5,0.5,linewidth=1,facecolor=cpal[conf_diff_mat[0,1]])
            print('Conf mat for annotator: '+anno_col)
            print(conf_diff_mat/np.sum(np.sum(conf_diff_mat)))
            norm_conf_diff_mat = conf_diff_mat/np.sum(np.sum(conf_diff_mat))
            tp,fp,fn,tn = norm_conf_diff_mat.flatten()
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            f1_score = 2*precision*recall/(precision+recall)
            mcc = (tp*tn - fp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
            print('F1 score: %f'%(f1_score))
            print('Matthews correlation coefficient: %f'%(mcc))

            # Fit a cubic polynomial to the consistency plots
            poly_feats = PolynomialFeatures(degree=3)
            poly_feats_fit = poly_feats.fit_transform(anno.values.reshape(-1,1))
            poly_reg = LinearRegression()
            poly_reg.fit(poly_feats_fit, ot.values)

            # Plotting
            axs_col = anno_idx%subplot_cols
            axs_row = int(math.floor(anno_idx/subplot_cols))

            # Plot annotations
            conf_diff_mask = np.logical_or(np.logical_or(np.logical_and(signed_anno_diff > 0, signed_ot_diff > 0), np.logical_and(signed_anno_diff < 0, signed_ot_diff < 0)), np.logical_and(signed_anno_diff == 0, signed_ot_diff == 0))
            conf_diff_mask = np.insert(conf_diff_mask, 0, True) # Assume the first difference was correct
            if avg_conf_diff_mask is None:
               avg_conf_diff_mask = conf_diff_mask
            else:
               avg_conf_diff_mask = np.hstack((avg_conf_diff_mask, conf_diff_mask))


            #correct_anno = anno[conf_diff_mask]
            #incorrect_anno = anno[~conf_diff_mask]
            PlotBackgroundBars(ax_anno[axs_row, axs_col], ot_df.iloc[:,0], conf_diff_mask, y_lim=(0,1), y_vals=anno)
            #ax_anno[axs_row, axs_col].plot(ot_df.iloc[:,0].values, ot, 'm-')
            #ax_anno[axs_row, axs_col].plot(anno_df.iloc[conf_diff_mask,0].values, correct_anno.values, c='#0077FF', marker='o', markersize=3)
            #ax_anno[axs_row, axs_col].plot(anno_df.iloc[~conf_diff_mask,0].values, incorrect_anno.values, c='#BF4E59', marker='o', markersize=3)
            #fig_anno.text(0.5, 0.04, 'Time (seconds)', ha='center')
            #fig_anno.text(0.04, 0.5, 'Green intensity', va='center', rotation='vertical')
            ax_anno[axs_row, axs_col].set_xlabel('Time (seconds)')
            ax_anno[axs_row, axs_col].set_ylabel('Green intensity')
            ax_anno[axs_row, axs_col].title.set_text(anno_col)

            # Plot consistency
            axs[axs_row, axs_col].axes.set_xlim(0.0,1.0)
            axs[axs_row, axs_col].axes.set_ylim(0.0,1.0)
            #axs[axs_row, axs_col].add_patch(tp_rect)
            #axs[axs_row, axs_col].add_patch(tn_rect)
            #axs[axs_row, axs_col].add_patch(fp_rect)
            #axs[axs_row, axs_col].add_patch(fn_rect)
            im = axs[axs_row, axs_col].scatter(ot, anno, c='#0077FF', zorder=2, s=5)
            im.set_cmap(cmap)
            axs[axs_row, axs_col].plot((0.0,1.0), (0.0,1.0), c='#000C99', linestyle='--', linewidth=2)
            #axs[axs_row, axs_col].plot((0.0,1.0), (0.5,0.5), 'r--')
            poly_reg_anno = np.sort(anno.values)
            axs[axs_row, axs_col].plot(poly_reg.predict(poly_feats.fit_transform(poly_reg_anno.reshape(-1,1))), poly_reg_anno, c="#FF5400", linestyle='-', linewidth=4)
            #fig.subplots_adjust(right=0.8)
            #cax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
            #fig.colorbar(im, cax=cax, orientation='vertical')
            axs[axs_row, axs_col].set_xlabel('True green intensity')
            axs[axs_row, axs_col].set_ylabel('Annotated green intensity')
            #fig.text(0.5, 0.04, 'True green intensity', ha='center')
            #fig.text(0.04, 0.5, 'Annotated green intensity', va='center', rotation='vertical')

            #axs[axs_row, axs_col].plot(consist_df.index, consist_df.iloc[:,0], 'r-')
            axs[axs_row, axs_col].title.set_text(anno_col)

            # Plot consistency derivative
            axs_diff[axs_row, axs_col].axes.set_xlim(-0.5,0.5)
            axs_diff[axs_row, axs_col].axes.set_ylim(-0.5,0.5)
            axs_diff[axs_row, axs_col].add_patch(tp_rect_diff)
            axs_diff[axs_row, axs_col].add_patch(tn_rect_diff)
            axs_diff[axs_row, axs_col].add_patch(fp_rect_diff)
            axs_diff[axs_row, axs_col].add_patch(fn_rect_diff)
            im_diff = axs_diff[axs_row, axs_col].scatter(ot_diff, anno_diff, c='#006099', zorder=2, s=5)
            im_diff.set_cmap(cmap)
            axs_diff[axs_row, axs_col].plot((0,0), (-0.5,0.5), 'b--')
            axs_diff[axs_row, axs_col].plot((-0.5,0.5), (0,0), 'b--')
            #div_diff = make_axes_locatable(axs_diff[axs_row, axs_col])
            #cax_diff = div_diff.append_axes('right', size='5%', pad=0.05)
            #fig.colorbar(im_diff, cax=cax_diff, orientation='vertical')
            fig_diff.subplots_adjust(right=0.8)
            cax_diff = fig_diff.add_axes([0.85, 0.15, 0.005, 0.7])
            fig.colorbar(im_diff, cax=cax_diff, orientation='vertical')
            #fig_diff.text(0.5, 0.04, 'True green intensity derivative', ha='center')
            #fig_diff.text(0.04, 0.5, 'Annotated green intensity derivative', va='center', rotation='vertical')
            axs_diff[axs_row, axs_col].set_xlabel('True green intensity derivative')
            axs_diff[axs_row, axs_col].set_ylabel('Annotated green intensity derivative')
            axs_diff[axs_row, axs_col].title.set_text(anno_col+' derivative')

         # Average annotation plots
         PlotBackgroundBars(ax_anno_avg, ot_df.iloc[:,0], avg_conf_diff_mask, y_lim=(0,1))
         ax_anno_avg.plot(ot_df.iloc[:,0].values, ot, 'm-')
         #fig_anno_avg.suptitle(task + ' ' + anno_type + ' majority vote correct derivative')
         #fig_anno.suptitle(task + ' ' + anno_type + ' correct derivative')
         #fig.suptitle(task + ' ' + anno_type + ' consistency')
         #fig_diff.suptitle(task + ' ' + anno_type + ' derivative consistency')

         tikzplotlib.save(os.path.join(output_path, task+'_'+anno_type+'_consistency.tex'), figure=fig, axis_width='\\figureWidth', textsize=12)
         tikzplotlib.save(os.path.join(output_path, task+'_'+anno_type+'_diff_consistency.tex'), figure=fig_diff, axis_width='\\figureWidth', textsize=12)
         tikzplotlib.save(os.path.join(output_path, task+'_'+anno_type+'_annos.tex'), figure=fig_anno, axis_width='\\figureWidth', textsize=12)
         tikzplotlib.save(os.path.join(output_path, task+'_'+anno_type+'_annos_avg.tex'), figure=fig_anno_avg, axis_width='\\figureWidth', textsize=12)
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
   ComputeAnnotatorConsistency(args.data_root_path, args.output_path, args.show_plots)
