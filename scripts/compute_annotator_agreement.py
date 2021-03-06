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
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as hcluster
from sklearn.cluster import SpectralClustering
from datetime import datetime
import pytz

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'util')))
import util
import agreement_metrics as agree

def GetUpperTri(corr_mat):
   m = corr_mat.shape[0]
   r,c = np.triu_indices(m,1)
   upper_tri_vals = corr_mat[r,c].flatten()
   return upper_tri_vals
   #median = np.quantile(upper_tri_vals, 0.5)
   #q1 = np.quantile(upper_tri_vals, 0.25)
   #q3 = np.quantile(upper_tri_vals, 0.75)
   #min_val = np.min(upper_tri_vals)
   #max_val = np.max(upper_tri_vals)
   #return (min_val, q1, median, q3, max_val)

def GetHierarchyClusters(link, k, method='maxclust'):
   clusters = hcluster.fcluster(link, k, method)
   clusters -= 1 # Zero-based indexing
   return clusters

def ClusterSimilarityMatrix(sim_mat, method='average'):
   n = len(sim_mat)
   flat_dist_mat = ssd.squareform(1.0-sim_mat)
   res_linkage = hcluster.linkage(flat_dist_mat, method=method)
   res_order = hcluster.leaves_list(res_linkage)
   seriated_sim = np.zeros((n,n))
   a,b = np.triu_indices(n,k=1)
   seriated_sim[a,b] = sim_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
   seriated_sim[b,a] = seriated_sim[a,b]
   for i in range(n):
      seriated_sim[i,i] = sim_mat[i,i]

   return seriated_sim, res_order, res_linkage

def ComputeAnnotatorAgreement(data_root_path, data_set_name, output_path, show_plots):
   if not os.path.isdir(output_path):
      os.makedirs(output_path)

   if data_set_name == 'green':
      data_dict = util.LoadGreenData(data_root_path, sample_rate=1)
      project_entry_name = 'Green Experiment'
      range_min = 0
      range_max = 1
   elif data_set_name == 'recola':
      #cache_file = '/tmp/recola_cache_arousal.pkl'
      cache_file = '/tmp/recola_cache_valence.pkl'
      data_dict = util.LoadRecolaData(data_root_path, cache_file)
      #project_entry_name = 'RECOLA arousal'
      project_entry_name = 'RECOLA valence'
      range_min = -1
      range_max = 1
   elif data_set_name == 'deam':
      print("Not implemented")
      return
      data_dict = util.LoadDEAMData(data_root_path)
      project_entry_name = 'DEAM'
   else:
      print("Unknown data set name argument.  Please use one of: 'green', 'recola', 'deam'")
      return

   # Compute agreement
   #for task_name in list(data_dict.keys()):
   # Train8
   #for task_name in list(data_dict.keys())[16:]:
   # Train4
   #for task_name in list(data_dict.keys())[12:]:
   #for task_name in list(data_dict.keys())[9:]:
   for task_name in list(data_dict.keys()):
   #for task_name in list(data_dict.keys())[17:]:
      for anno_type in data_dict[task_name].keys():
         combined_anno_df = data_dict[task_name][anno_type]['annotations']
             
         fig_anno, axs_anno = plt.subplots(1,1, figsize=(11,9))
         fig_corr_box, axs_corr_box = plt.subplots(1,1, figsize=(11,9))
         fig_agree, axs_agree = plt.subplots(1,1, figsize=(11,9), tight_layout=True)
         fig_corr, axs_corr = plt.subplots(2,3, figsize=(11,9))
         fig_agg, axs_agg = plt.subplots(2,3, figsize=(11,9))
         fig_spec, axs_spec= plt.subplots(2,3, figsize=(11,9))

         for anno_col in combined_anno_df.columns:
            anno = combined_anno_df[anno_col].values
            axs_anno.plot(range(len(anno)), anno)
            #axs_anno.plot(range(len(anno)), len(anno)*[np.mean(anno)], linestyle='dashed')

         ### Value-based metrics ###
         # Pearson
         pearson_corr_mat = agree.PearsonCorr(combined_anno_df)

         # Spearman
         spearman_corr_mat = agree.SpearmanCorr(combined_anno_df)

         # Kendall's Tau
         kendall_corr_mat = agree.KendallTauCorr(combined_anno_df)

         # MSE
         mse_mat = agree.MeanSquaredErrorMat(combined_anno_df)

         # CCC
         ccc_corr_mat = agree.ConcordanceCorrelationCoef(combined_anno_df)
         
         # SAGR
         sagr_corr_mat = agree.SAGR(combined_anno_df, range_min, range_max)

         ### Derivative-based methods ###
         norm_diff_df = agree.NormedDiff(combined_anno_df)
         #abs_norm_diff_df = norm_diff_df.abs()
         #accum_norm_diff_df = agree.AccumNormedDiff(combined_anno_df)

         # Cohen's Kappa normed diff
         #cohens_kappa_norm_diff_corr_mat = agree.CohensKappaCorr(norm_diff_df, labels=[-1,0,1])

         # Cohen's Kappa abs normed diff
         #cohens_kappa_abs_norm_diff_corr_mat = agree.CohensKappaCorr(abs_norm_diff_df, labels=[0,1])

         # SDA
         #sda_mat = agree.SDA(norm_diff_df)
         #sda_mat = agree.KSDA(norm_diff_df)
         sda_mat = agree.SDACorrected(norm_diff_df)
         #sda_mat = agree.SDACorrected2(norm_diff_df)

         # TSS absolute diff, T>=N-1
         #abs_norm_sum_delta_mat = agree.NormedSumDelta(abs_norm_diff_df)

         ###############
         # Compute summary agreement measures
         ###############
         # Cronbach's alpha
         cronbachs_alpha = agree.CronbachsAlphaCorr(combined_anno_df)
         
         # Cronbach's alpha normed diff
         #cronbachs_alpha_norm_diff = agree.CronbachsAlphaCorr(norm_diff_df)

         # Cronbach's alpha abs normed diff
         #cronbachs_alpha_abs_norm_diff = agree.CronbachsAlphaCorr(abs_norm_diff_df)

         # Cronbach's alpha abs normed diff
         #cronbachs_alpha_accum_norm_diff = agree.CronbachsAlphaCorr(accum_norm_diff_df)

         # ICC(2,1)
         icc_df = agree.ICC(combined_anno_df)
         #icc21_df = icc_df.loc[icc_df['type'] == 'ICC2',:]
         #icc21 = icc21_df['ICC'].iloc[0]
         icc2 = icc_df.iloc[0,0]

         # SAGR (signed agreement)
         # BB - Doesn't make sense for scales where zero isn't the center

         # Krippendorff's alpha
         krippendorffs_alpha = agree.KrippendorffsAlpha(combined_anno_df)

         # Krippendorff's alpha of normed diff
         #krippendorffs_alpha_norm_diff = agree.KrippendorffsAlpha(norm_diff_df)

         # Krippendorff's alpha of abs normed diff
         #krippendorffs_alpha_abs_norm_diff = agree.KrippendorffsAlpha(abs_norm_diff_df)

         # Accumulated Normed Rank-based Krippendorff's Alpha
         #krippendorffs_alpha_accum_norm_diff = agree.KrippendorffsAlpha(accum_norm_diff_df)

         ###############

         # Put global agreement measures into a dataframe
         #global_agreement_df = pd.DataFrame(data=[[icc2, cronbachs_alpha, cronbachs_alpha_norm_diff, cronbachs_alpha_abs_norm_diff, cronbachs_alpha_accum_norm_diff, krippendorffs_alpha, krippendorffs_alpha_norm_diff, krippendorffs_alpha_abs_norm_diff, krippendorffs_alpha_accum_norm_diff]], columns=['ICC(2)', 'Cronbach\'s Alpha', 'Cronbach\'s Alpha Norm Diff', 'Cronbach\'s Alpha Abs Norm Diff', 'Cronbach\'s Alpha Accum Norm Diff', 'Krippendorff\'s Alpha', 'Krippendorff\'s Alpha Norm Diff', 'Krippendorff\'s Alpha Abs Norm Diff', 'Krippendorff\'s Alpha Accum Norm Diff'])
         global_agreement_df = pd.DataFrame(data=[[icc2, cronbachs_alpha, krippendorffs_alpha]], columns=['ICC(2)', 'Cronbach\'s Alpha', 'Krippendorff\'s Alpha'])

         # Max-normalize the MSE and convert to a correlation-like matrix
         mse_corr_mat = 1.0 - mse_mat/np.max(mse_mat.values)
         np.fill_diagonal(mse_corr_mat.values, 1)

         # Force symmetry for corr matrices and normalize into [0,1] range
         pearson_corr_mat[pd.isna(pearson_corr_mat)] = 0
         np.fill_diagonal(pearson_corr_mat.values, 1)
         pearson_corr_mat.clip(lower=-1.0, upper=1.0, inplace=True)
         #pearson_norm_corr_mat = pearson_corr_mat.abs()
         pearson_norm_corr_mat = 0.5*pearson_corr_mat + 0.5

         spearman_corr_mat[pd.isna(spearman_corr_mat)] = 0
         np.fill_diagonal(spearman_corr_mat.values, 1)
         spearman_corr_mat.clip(lower=-1.0, upper=1.0, inplace=True)
         #spearman_norm_corr_mat = spearman_corr_mat.abs()
         spearman_norm_corr_mat = 0.5*spearman_corr_mat + 0.5

         kendall_corr_mat[pd.isna(kendall_corr_mat)] = 0
         np.fill_diagonal(kendall_corr_mat.values, 1)
         kendall_corr_mat.clip(lower=-1.0, upper=1.0, inplace=True)
         #kendall_norm_corr_mat = kendall_corr_mat.abs()
         kendall_norm_corr_mat = 0.5*kendall_corr_mat+ 0.5

         ccc_corr_mat[pd.isna(ccc_corr_mat)] = 0
         np.fill_diagonal(ccc_corr_mat.values, 1)
         ccc_corr_mat.clip(lower=-1.0, upper=1.0, inplace=True)
         #ccc_norm_corr_mat = ccc_corr_mat.abs()
         ccc_norm_corr_mat = 0.5*ccc_corr_mat + 0.5

         sagr_corr_mat[pd.isna(sagr_corr_mat)] = 0
         np.fill_diagonal(sagr_corr_mat.values, 1)
         sagr_corr_mat.clip(lower=-1.0, upper=1.0, inplace=True)
         sagr_norm_corr_mat = 0.5*sagr_corr_mat + 0.5

         #cohens_kappa_norm_diff_corr_mat[pd.isna(cohens_kappa_norm_diff_corr_mat)] = 0
         #np.fill_diagonal(cohens_kappa_norm_diff_corr_mat.values, 1)
         #cohens_kappa_norm_diff_corr_mat.clip(lower=-1.0, upper=1.0, inplace=True)
         #cohens_kappa_norm_diff_corr_mat = 0.5*cohens_kappa_norm_diff_corr_mat + 0.5

         #cohens_kappa_abs_norm_diff_corr_mat[pd.isna(cohens_kappa_abs_norm_diff_corr_mat)] = 0
         #np.fill_diagonal(cohens_kappa_abs_norm_diff_corr_mat.values, 1)
         #cohens_kappa_abs_norm_diff_corr_mat.clip(lower=-1.0, upper=1.0, inplace=True)
         #cohens_kappa_abs_norm_diff_corr_mat = 0.5*cohens_kappa_abs_norm_diff_corr_mat + 0.5

         np.fill_diagonal(sda_mat.values, 1)
         sda_norm_mat = 0.5*sda_mat + 0.5

         # Print correlation statistics
         pearson_tri = GetUpperTri(pearson_corr_mat.values)
         spearman_tri = GetUpperTri(spearman_corr_mat.values)
         kendall_tau_tri = GetUpperTri(kendall_corr_mat.values)
         ccc_tri = GetUpperTri(ccc_corr_mat.values)
         #mse_tri = GetUpperTri(mse_mat.values)
         sda_tri = GetUpperTri(sda_mat.values)
         combined_tri = np.vstack((pearson_tri, spearman_tri, kendall_tau_tri, ccc_tri, sda_tri)).T
         corr_tri_df = pd.DataFrame(data=combined_tri, columns=['Pearson', 'Spearman', 'Kendall\'s Tau', 'CCC',  'SDA'])

         # Agglomerative clustering
         (agg_pearson_sim, pearson_cluster_order_idx, pearson_agg_link) = ClusterSimilarityMatrix(pearson_norm_corr_mat.values, method='centroid')
         agg_pearson_sim = pd.DataFrame(data=agg_pearson_sim, index=pearson_norm_corr_mat.columns[pearson_cluster_order_idx], columns=pearson_norm_corr_mat.columns[pearson_cluster_order_idx])
         (agg_spearman_sim, spearman_cluster_order_idx, spearman_agg_link) = ClusterSimilarityMatrix(spearman_norm_corr_mat.values, method='centroid')
         agg_spearman_sim = pd.DataFrame(data=agg_spearman_sim, index=spearman_norm_corr_mat.columns[spearman_cluster_order_idx], columns=spearman_norm_corr_mat.columns[spearman_cluster_order_idx])
         (agg_kendall_sim, kendall_cluster_order_idx, kendall_agg_link) = ClusterSimilarityMatrix(kendall_norm_corr_mat.values, method='centroid')
         agg_kendall_sim = pd.DataFrame(data=agg_kendall_sim, index=kendall_norm_corr_mat.columns[kendall_cluster_order_idx], columns=kendall_norm_corr_mat.columns[kendall_cluster_order_idx])
         (agg_mse_sim, mse_cluster_order_idx, mse_agg_link) = ClusterSimilarityMatrix(mse_corr_mat.values, method='centroid')
         agg_mse_sim = pd.DataFrame(data=agg_mse_sim, index=mse_corr_mat.columns[mse_cluster_order_idx], columns=mse_corr_mat.columns[mse_cluster_order_idx])
         (agg_ccc_sim, ccc_cluster_order_idx, ccc_agg_link) = ClusterSimilarityMatrix(ccc_norm_corr_mat.values, method='centroid')
         agg_ccc_sim = pd.DataFrame(data=agg_ccc_sim, index=ccc_corr_mat.columns[ccc_cluster_order_idx], columns=ccc_norm_corr_mat.columns[ccc_cluster_order_idx])
         #(agg_cohens_kappa_norm_diff_sim, cohens_kappa_norm_diff_cluster_order_idx, cohens_kappa_norm_diff_agg_link) = ClusterSimilarityMatrix(cohens_kappa_norm_diff_corr_mat.values, method='centroid')
         #(agg_cohens_kappa_abs_norm_diff_sim, cohens_kappa_abs_norm_diff_cluster_order_idx, cohens_kappa_abs_norm_diff_agg_link) = ClusterSimilarityMatrix(cohens_kappa_abs_norm_diff_corr_mat.values, method='centroid')
         (agg_sda_sim, sda_cluster_order_idx, sda_agg_link) = ClusterSimilarityMatrix(sda_mat.values, method='centroid')
         agg_sda_sim = pd.DataFrame(data=agg_sda_sim, index=sda_mat.columns[sda_cluster_order_idx], columns=sda_mat.columns[sda_cluster_order_idx])
         #(agg_abs_norm_sum_delta_sim, abs_norm_sum_delta_cluster_order_idx, abs_norm_sum_delta_agg_link) = ClusterSimilarityMatrix(abs_norm_sum_delta_mat.values, method='centroid')

         agg_pearson_labels = GetHierarchyClusters(pearson_agg_link, k=2)
         agg_spearman_labels = GetHierarchyClusters(spearman_agg_link, k=2)
         agg_kendall_labels = GetHierarchyClusters(kendall_agg_link, k=2)
         agg_mse_labels = GetHierarchyClusters(mse_agg_link, k=2)
         agg_ccc_labels = GetHierarchyClusters(ccc_agg_link, k=2)
         #agg_cohens_kappa_norm_diff_labels = GetHierarchyClusters(cohens_kappa_norm_diff_agg_link, k=2)
         #agg_cohens_kappa_abs_norm_diff_labels = GetHierarchyClusters(cohens_kappa_abs_norm_diff_agg_link, k=2)
         agg_sda_labels = GetHierarchyClusters(sda_agg_link, k=2)
         #agg_norm_sum_delta_labels = GetHierarchyClusters(abs_norm_sum_delta_agg_link, k=2)


         # Spectral clustering
         spec_pearson_labels = SpectralClustering(n_clusters=2, assign_labels="discretize", affinity='precomputed').fit(pearson_norm_corr_mat.values).labels_
         spec_spearman_labels = SpectralClustering(n_clusters=2, assign_labels="discretize", affinity='precomputed').fit(spearman_norm_corr_mat.values).labels_
         spec_kendall_labels = SpectralClustering(n_clusters=2, assign_labels="discretize", affinity='precomputed').fit(kendall_norm_corr_mat.values).labels_
         spec_mse_labels = SpectralClustering(n_clusters=2, assign_labels="discretize", affinity='precomputed').fit(mse_corr_mat.values).labels_
         spec_ccc_labels = SpectralClustering(n_clusters=2, assign_labels="discretize", affinity='precomputed').fit(ccc_norm_corr_mat.values).labels_
         #spec_cohens_kappa_norm_diff_labels = SpectralClustering(n_clusters=2, assign_labels="discretize", affinity='precomputed').fit(cohens_kappa_norm_diff_corr_mat.values).labels_
         #spec_cohens_kappa_abs_norm_diff_labels = SpectralClustering(n_clusters=2, assign_labels="discretize", affinity='precomputed').fit(cohens_kappa_abs_norm_diff_corr_mat.values).labels_
         spec_sda_labels = SpectralClustering(n_clusters=2, assign_labels="discretize", affinity='precomputed').fit(sda_norm_mat.values).labels_
         #spec_abs_norm_sum_delta_labels = SpectralClustering(n_clusters=2, assign_labels="discretize", affinity='precomputed').fit(abs_norm_sum_delta_mat.values).labels_

         # Output binary clustering results
         agreement_strs = ['Pearson', 'Spearman', 'Kendall\'s Tau', 'CCC', 'MSE', 'SDA']
         cluster_methods = []
         for method in ['Agglomerative', 'Spectral']:
            for i in range(len(agreement_strs)):
               cluster_methods.append(agreement_strs[i] + ' ' + method)
         cluster_labels_mat = np.vstack((agg_pearson_labels, agg_spearman_labels, agg_kendall_labels, agg_ccc_labels, agg_mse_labels, agg_sda_labels, spec_pearson_labels, spec_spearman_labels, spec_kendall_labels, spec_ccc_labels, spec_mse_labels, spec_sda_labels)).T
         cluster_df = pd.DataFrame(data=cluster_labels_mat, index=combined_anno_df.columns, columns=cluster_methods)

         output_file_path = os.path.join(output_path, project_entry_name+'_'+task_name+'_'+anno_type+'_clusters.csv')
         cluster_df.to_csv(output_file_path, index=True, header=True)

         cmap = sns.diverging_palette(220, 10, as_cmap=True)

         # Pairwise correlation and error matrices
         sns.heatmap(pearson_norm_corr_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_corr[0,0])
         sns.heatmap(spearman_norm_corr_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_corr[0,1])
         sns.heatmap(kendall_norm_corr_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_corr[0,2])
         sns.heatmap(mse_corr_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_corr[1,0])
         sns.heatmap(ccc_norm_corr_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_corr[1,1])
         #sns.heatmap(cohens_kappa_norm_diff_corr_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_corr[1,0])
         #sns.heatmap(cohens_kappa_abs_norm_diff_corr_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_corr[1,1])
         sns.heatmap(sda_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_corr[1,2])
         #sns.heatmap(abs_norm_sum_delta_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_corr[1,3])

         # Summary agreement bar graph
         sns.barplot(data=global_agreement_df, ax=axs_agree)
         #for tick in axs_agree.xaxis.get_major_ticks()[1::2]:
         #   tick.set_pad(15)
         for tick in axs_agree.xaxis.get_major_ticks():
            tick.label.set_rotation(90)

         # Agg clustered pairwise correlation
         sns.heatmap(agg_pearson_sim, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_agg[0,0])
         sns.heatmap(agg_spearman_sim, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_agg[0,1])
         sns.heatmap(agg_kendall_sim, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_agg[0,2])
         sns.heatmap(agg_mse_sim, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_agg[1,0])
         sns.heatmap(agg_ccc_sim, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_agg[1,1])
         #sns.heatmap(agg_cohens_kappa_norm_diff_sim, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_agg[1,0])
         #sns.heatmap(agg_cohens_kappa_abs_norm_diff_sim, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_agg[1,1])
         sns.heatmap(agg_sda_sim, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_agg[1,2])
         #sns.heatmap(agg_abs_norm_sum_delta_sim, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_agg[1,3])

         # Spectral clustered pairwise correlation matrices
         pearson_corr_mat_spec = pearson_norm_corr_mat 
         spearman_corr_mat_spec = spearman_norm_corr_mat
         kendall_corr_mat_spec = kendall_norm_corr_mat
         ccc_corr_mat_spec = ccc_norm_corr_mat
         mse_corr_mat_spec = mse_corr_mat
         #cohens_kappa_norm_diff_corr_mat_spec = cohens_kappa_norm_diff_corr_mat
         #cohens_kappa_abs_norm_diff_corr_mat_spec = cohens_kappa_abs_norm_diff_corr_mat
         sda_norm_mat_spec = sda_norm_mat
         #abs_norm_sum_delta_mat_spec = abs_norm_sum_delta_mat

         #spec_corr_mats = [(pearson_corr_mat_spec, spec_pearson_labels), (spearman_corr_mat_spec, spec_spearman_labels), (kendall_corr_mat_spec, spec_kendall_labels), (ccc_corr_mat_spec, spec_ccc_labels), (mse_corr_mat_spec, spec_mse_labels), (cohens_kappa_norm_diff_corr_mat_spec, spec_cohens_kappa_norm_diff_labels), (cohens_kappa_abs_norm_diff_corr_mat_spec, spec_cohens_kappa_abs_norm_diff_labels), (norm_sum_delta_mat_spec, spec_norm_sum_delta_labels), (abs_norm_sum_delta_mat_spec, spec_abs_norm_sum_delta_labels)]
         spec_corr_mats = [(pearson_corr_mat_spec, spec_pearson_labels), (spearman_corr_mat_spec, spec_spearman_labels), (kendall_corr_mat_spec, spec_kendall_labels), (ccc_corr_mat_spec, spec_ccc_labels), (mse_corr_mat_spec, spec_mse_labels), (sda_norm_mat_spec, spec_sda_labels)]
         for spec_corr_mat, spec_labels in spec_corr_mats:
            for i in range(len(spec_corr_mat)):
               if spec_labels[i] == 0:
                  spec_corr_mat.iloc[i,:] *= -1
                  spec_corr_mat.iloc[:,i] *= -1
                  spec_corr_mat.iloc[i,i] *= -1
               spec_corr_mat = 0.5*spec_corr_mat + 0.5
         sns.heatmap(pearson_corr_mat_spec, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_spec[0,0])
         sns.heatmap(spearman_corr_mat_spec, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_spec[0,1])
         sns.heatmap(kendall_corr_mat_spec, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_spec[0,2])
         sns.heatmap(mse_corr_mat_spec, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_spec[1,0])
         sns.heatmap(ccc_corr_mat_spec, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_spec[1,1])
         #sns.heatmap(cohens_kappa_norm_diff_corr_mat_spec, cmap=cmap, vmax=1.0, center=0, square=True, linewidth=0.5, cbar_kws={"shrink" : 0.5}, ax=axs_spec[1,0])
         #sns.heatmap(cohens_kappa_abs_norm_diff_corr_mat_spec, cmap=cmap, vmax=1.0, center=0, square=True, linewidth=0.5, cbar_kws={"shrink" : 0.5}, ax=axs_spec[1,1])
         sns.heatmap(sda_norm_mat_spec, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_spec[1,2])
         #sns.heatmap(abs_norm_sum_delta_mat_spec, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_spec[1,3])

         # Box and whisker plots
         if len(corr_tri_df) > 2:
            sns.boxplot(data=corr_tri_df, ax=axs_corr_box)
         else:
            sns.barplot(data=corr_tri_df, ax=axs_corr_box)
         #sns.heatmap(pearson_norm_corr_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_corr[0,0])

         axs_anno.title.set_text('Raw Annotations')
         axs_anno.legend(combined_anno_df.columns.values)
         axs_agree.title.set_text(project_entry_name+' '+task_name+' '+anno_type+'Global Agreement Metrics')
         axs_corr_box.title.set_text(project_entry_name+' '+task_name+' '+anno_type+' Correlation')
         axs_corr[0,0].title.set_text('Pearson Corr')
         axs_corr[0,1].title.set_text('Spearman Corr')
         axs_corr[0,2].title.set_text('Kendall Tau')
         axs_corr[1,0].title.set_text('MSE Corr')
         axs_corr[1,1].title.set_text('CCC')
         #axs_corr[1,0].title.set_text('Cohens Kappa Normed Diff')
         #axs_corr[1,1].title.set_text('Cohens Kappa Abs Normed Diff')
         axs_corr[1,2].title.set_text('KSDA')
         #axs_corr[1,3].title.set_text('Abs Normed Sum Delta')
         axs_agg[0,0].title.set_text('Pearson Agg')
         axs_agg[0,1].title.set_text('Spearman Agg')
         axs_agg[0,2].title.set_text('Kendall Agg')
         axs_agg[1,0].title.set_text('MSE Agg')
         axs_agg[1,1].title.set_text('CCC Agg')
         #axs_agg[1,0].title.set_text('Cohens Kappa Normed Diff Agg')
         #axs_agg[1,1].title.set_text('Cohens Kappa Abs Normed Diff Agg')
         axs_agg[1,2].title.set_text('KSDA Agg')
         #axs_agg[1,3].title.set_text('Abs Normed Sum Delta Agg')
         axs_spec[0,0].title.set_text('Pearson Spec')
         axs_spec[0,1].title.set_text('Spearman Spec')
         axs_spec[0,2].title.set_text('Kendall Spec')
         axs_spec[1,0].title.set_text('MSE Spec')
         axs_spec[1,1].title.set_text('CCC Spec')
         #axs_spec[1,0].title.set_text('Cohens Kappa Normalized Diff Spec')
         #axs_spec[1,1].title.set_text('Cohens Kappa Abs Normalized Diff Spec')
         axs_spec[1,2].title.set_text('KSDA Spec')
         #axs_spec[1,3].title.set_text('Abs Normed Sum Delta Spec')
         fig_anno.suptitle(project_entry_name+' '+task_name+' '+anno_type+' Annotations')
         fig_agree.suptitle(project_entry_name+' '+task_name+' '+anno_type+' Global Agreement Measures')
         fig_agg.suptitle(project_entry_name+' '+task_name+' '+anno_type+' Agglomerative Clustering of Agreement')
         fig_spec.suptitle(project_entry_name+' '+task_name+' '+anno_type+' Spectral Clustering of Agreement')

         tikzplotlib.save(os.path.join(output_path, project_entry_name.lower().replace(' ','_')+' '+task_name+' '+anno_type+'_annos.tex'), figure=fig_anno, axis_width='\\figureWidth', textsize=12)
         tikzplotlib.save(os.path.join(output_path, project_entry_name.lower().replace(' ','_')+' '+task_name+' '+anno_type+'_corr.tex'), figure=fig_corr, axis_width='\\figureWidth', textsize=12)
         tikzplotlib.save(os.path.join(output_path, project_entry_name.lower().replace(' ','_')+' '+task_name+' '+anno_type+'_agg_cluster.tex'), figure=fig_agg, axis_width='\\figureWidth', textsize=12)
         tikzplotlib.save(os.path.join(output_path, project_entry_name.lower().replace(' ','_')+' '+task_name+' '+anno_type+'_spec_cluster.tex'), figure=fig_spec, axis_width='\\figureWidth', textsize=12)
         tikzplotlib.save(os.path.join(output_path, project_entry_name.lower().replace(' ','_')+' '+task_name+' '+anno_type+'_global_agree.tex'), figure=fig_agree, axis_width='\\figureWidth', textsize=12)
         tikzplotlib.save(os.path.join(output_path, project_entry_name.lower().replace(' ','_')+' '+task_name+' '+anno_type+'_boxplot.tex'), figure=fig_corr_box, axis_width='\\figureWidth', textsize=12)

         if show_plots:
            plt.show()

   return

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--data_root_path', required=True, help='Path to the root folder of the data set')
   parser.add_argument('--data_set_name', required=True, help='One of the following: "green", "recola", "deam"')
   parser.add_argument('--output_path', required=True, help='Output folder path')
   parser.add_argument('--show_plots', required=False, action='store_true')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   ComputeAnnotatorAgreement(args.data_root_path, args.data_set_name, args.output_path, args.show_plots)
