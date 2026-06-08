"""
Exported from Downstream/Presence_Score.ipynb
"""

# %% [cell 1 | notebook cell 0]
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad
from scipy import sparse
# import scarches as sca

# %% [cell 2 | notebook cell 2]
human  = sc.read_h5ad('workspace/hpca/hpca_downstream/julia_concat_objects_new/T1D_T2D_aab_core_2705.h5ad')
mouse = sc.read_h5ad('workspace/hpca/data/Mouse_Atlas/mouse_atlas.h5ad')

# %% [cell 3 | notebook cell 3]
mouse

# %% [cell 4 | notebook cell 4]
human

# %% [cell 5 | notebook cell 5]
pd.set_option('display.max_columns', None)
mouse.obs.head()

# %% [cell 6 | notebook cell 6]
mouse.obs.age_approxDays.value_counts()

# %% [cell 7 | notebook cell 7]
mouse.obs.head()

# %% [cell 8 | notebook cell 8]
human.obs.head()

# %% [cell 9 | notebook cell 9]
#make a df of all unique methuman columns with column name as the methuman and rows as unique values
interesting_cols = ['design', 'ins_high', 'gcg_high','age', 'age_approxDays', 'strain', 'diabetes_model', 'chemical_stress', 'sex', 'disease', 'tissue', 'development_stage',]
methuman= {}
for i in interesting_cols:
    methuman[i] = mouse.obs[i].unique()
methuman_df = pd.DataFrame.from_dict(methuman, orient='index').T
methuman_df

# %% [cell 10 | notebook cell 10]
human.obs['Level_4_extension_diabetes_simplified'] = human.obs.Level_4_extension_diabetes.copy()
human.obs.Level_4_extension_diabetes_simplified = np.where(human.obs.Level_3_extension_diabetes.str.contains('T1D|T2D|aab'), human.obs.Level_3_extension_diabetes, human.obs.Level_4_extension_diabetes)
human.obs.columns[human.obs.columns.str.contains('diabetes')]

# %% [cell 11 | notebook cell 11]
plot_groups= human[human.obs.Level_4_extension_diabetes_simplified.str.contains('T1D|T2D|AAB')].obs.Level_4_extension_diabetes_simplified.unique().tolist()

# %% [cell 12 | notebook cell 12]
sc.set_figure_params(dpi=100)

# %% [cell 13 | notebook cell 13]
mouse.obsm['X_umap'] = mouse.obsm['X_integrated_umap'].copy()
sc.pl.umap(mouse, color='cell_type_reannotatedIntegrated', legend_fontsize=5, frameon=False)
sc.pl.umap(human, color='Level_4_extension_diabetes_simplified', legend_fontsize=5, frameon=False, groups=plot_groups, palette='Set1', size=10)

# %% [cell 14 | notebook cell 14]
print(len(human.obs.Dataset_Source.unique()))
human.obs.Dataset_Source.value_counts()

# %% [cell 15 | notebook cell 15]
human.obs['Dataset_ID'] = human.obs['Dataset_Source'].astype(str) + ' : ' + human.obs['donor_id'].astype(str)
human.obs.groupby('Dataset_Source')['Dataset_ID'].unique()

# %% [cell 16 | notebook cell 16]
pwd

# %% [cell 17 | notebook cell 17]
from typing import Any


def shannon_entropy(x):
    p = x.value_counts(normalize=True)
    return -(p * np.log2(p)).sum()
df = human[human.obs.source == 'core'].obs.copy()
cluster_df = (
    df
    .groupby("Level_4_extension")
    .agg(
        n_cells=("Level_4_extension", "size"),

        # robustness / mixing
        dataset_entropy=("Dataset", shannon_entropy),
        donor_entropy=("Dataset_ID", shannon_entropy),
        suspension_entropy=("suspension_type", shannon_entropy),
        dataset_diversity=("Dataset", "nunique"),
        donor_diversity=("Dataset_ID", "nunique"),
        suspension_diversity=("suspension_type", "nunique"),).reset_index())
def minmax(x):
    return (x - x.min()) / (x.max() - x.min())
print(cluster_df.shape)
print(df.shape)
cluster_df["donor_entropy_scaled"] = minmax(cluster_df["donor_entropy"])
cluster_df["dataset_entropy_scaled"] = minmax(cluster_df["dataset_entropy"])
cluster_df["suspension_entropy_scaled"] = minmax(cluster_df["suspension_entropy"])
cluster_df['donor_entropy_cat'] =  pd.cut(cluster_df["donor_entropy_scaled"], bins=3, labels=["low", "mid", "high"])
cluster_df['dataset_entropy_cat'] =  pd.cut(cluster_df["dataset_entropy_scaled"], bins=3, labels=["low", "mid", "high"] )
cluster_df['suspension_entropy_cat'] =  pd.cut(cluster_df["suspension_entropy_scaled"], bins=3, labels=["low", "mid", "high"])
cluster_df["entropy_score"] = cluster_df[["donor_entropy_scaled", "dataset_entropy_scaled", "suspension_entropy_scaled"]].mean(axis=1)
cluster_df['entropy_cat'] = pd.cut(cluster_df["entropy_score"], bins=3, labels=["low", "mid", "high"])
# add to human.obs
human.obs['entropy_cell_states_cat'] = np.nan
human.obs['entropy_cell_states_cat'] = np.nan
mask_core = human.obs['source'] == 'core'
human.obs.loc[mask_core, 'entropy_cell_states_cat'] = human.obs.loc[mask_core, 'Level_4_extension'].map(
    dict(zip(cluster_df.Level_4_extension, cluster_df['entropy_cat']))
)

mask = human.obs['entropy_cell_states_cat'].isna()
human.obs.loc[mask, 'entropy_cell_states_cat'] = human.obs.loc[mask, 'source']
# human.obs['entropy_cell_states_score'] = human.obs.Level_4_extension_diabetes_simplified.map(dict(zip(cluster_df.Level_4_extension_diabetes_simplified, cluster_df['entropy_score'])))

# %% [cell 18 | notebook cell 18]
pd.set_option('display.max_rows', 100)
cluster_df.sort_values(by='entropy_cat').to_csv('workspace/hpca/hpca_downstream/entropy_df.csv')
# pd.set_option('display.max_rows', 20)
cluster_df.sort_values(by='entropy_cat')

# %% [cell 19 | notebook cell 19]
human[human.obs.Level_4_extension == 'Beta Cell: INS_High, IAPP_Low'].obs.donor_id.value_counts()

# %% [cell 20 | notebook cell 20]
human.obs['entropy_cell_states_cat'].value_counts()

# %% [cell 21 | notebook cell 21]
from pynndescent import NNDescent
from scipy import sparse
from typing import Optional, Union, Mapping, Literal
import warnings

def gaussian_kernel(d, sigma = None):
    if sigma is None:
        sigma = np.max(d) / 3
    gauss = np.exp(-0.5 * np.square(d) / np.square(sigma))
    return gauss

def nn2adj(nn,
           n1 = None,
           n2 = None,
           weight: Literal['unweighted','dist','gaussian_kernel'] = 'unweighted',
           sigma = None
          ):
    if n1 is None:
        n1 = nn[0].shape[0]
    if n2 is None:
        n2 = np.max(nn[0].flatten())
    
    df = pd.DataFrame({'i' : np.repeat(range(nn[0].shape[0]), nn[0].shape[1]),
                       'j' : nn[0].flatten(),
                       'x' : nn[1].flatten()})
    
    if weight == 'unweighted':
        adj = sparse.csr_matrix((np.repeat(1, df.shape[0]), (df['i'], df['j'])), shape=(n1, n2))
    else:
        if weight == 'gaussian_kernel':
            df['x'] = gaussian_kernel(df['x'], sigma)
        adj = sparse.csr_matrix((df['x'], (df['i'], df['j'])), shape=(n1, n2))
    
    return adj

def build_nn(ref,
             query = None,
             k = 100,
             weight: Literal['unweighted','dist','gaussian_kernel'] = 'unweighted',
             sigma = None
            ):
    if query is None:
        query = ref
    
    index = NNDescent(ref)
    knn = index.query(query, k=k)
    adj = nn2adj(knn, n1 = query.shape[0], n2 = ref.shape[0], weight = weight, sigma = sigma)
    return adj

def build_mutual_nn(dat1, dat2 = None, k1 = 100, k2 = None):
    if dat2 is None:
        dat2 = dat1
    if k2 is None:
        k2 = k1
    
    index_1 = NNDescent(dat1)
    index_2 = NNDescent(dat2)
    knn_21 = index_1.query(dat2, k=k1)
    knn_12 = index_2.query(dat1, k=k2)
    adj_21 = nn2adj(knn_21, n1 = dat2.shape[0], n2 = dat1.shape[0])
    adj_12 = nn2adj(knn_12, n1 = dat1.shape[0], n2 = dat2.shape[0])
    
    adj_mnn = adj_12.multiply(adj_21.T)
    return adj_mnn

def get_transition_prob_mat(dat, k = 50, symm = True):
    index = NNDescent(dat)
    knn = index.query(dat, k = k)
    adj = nn2adj(knn, n1 = dat.shape[0], n2 = dat.shape[0])
    if symm:
        adj = ((adj + adj.T) > 0) + 0
    prob = sparse.diags(1 / np.array(adj.sum(1)).flatten()) @ adj.transpose()
    return prob

def random_walk_with_restart(init, transition_prob, alpha = 0.5, num_rounds = 100):
    init = np.array(init).flatten()
    heat = init[:,None]
    for i in range(num_rounds):
        heat = init[:,None] * alpha + (1 - alpha) * (transition_prob.transpose() @ heat)
    return heat

def get_wknn(ref,                                                                                                    # the ref representation to build ref-query neighbor graph
             query,                                                                                                  # the query representation to build ref-query neighbor graph
             ref2 = None,                                                                                            # the ref representation to build ref-ref neighbor graph
             k: int = 100,                                                                                           # number of neighbors per cell
             query2ref: bool = True,                                                                                 # consider query-to-ref neighbors
             ref2query: bool = True,                                                                                 # consider ref-to-query neighbors
             weighting_scheme: Literal['n','top_n','jaccard','jaccard_square','gaussian','dist'] = 'jaccard_square', # how to weight edges in the ref-query neighbor graph
             top_n: Optional[int] = None,
             sigma: Optional[float] = None,
             return_adjs: bool = False
            ):
    adj_q2r = build_nn(ref = ref, query = query, k = k, weight = 'dist' if weighting_scheme in ['gaussian', 'dist'] else 'unweighted')
    
    adj_r2q = None
    if ref2query:
        adj_r2q = build_nn(ref = query, query = ref, k = k, weight = 'dist' if weighting_scheme in ['gaussian', 'dist'] else 'unweighted')
    
    if query2ref and not ref2query:
        adj_knn = adj_q2r.T
    elif ref2query and not query2ref:
        adj_knn = adj_r2q
    elif ref2query and query2ref:
        adj_knn_shared = (adj_r2q > 0).multiply(adj_q2r.T > 0)
        adj_knn = adj_r2q + adj_q2r.T - adj_r2q.multiply(adj_knn_shared)
    else:
        warnings.warn('At least one of query2ref and ref2query should be True. Reset to default with both being True.')
        adj_knn_shared = (adj_r2q > 0).multiply(adj_q2r.T > 0)
        adj_knn = adj_r2q + adj_q2r.T - adj_r2q.multiply(adj_knn_shared)
    
    if weighting_scheme in ['n','top_n','jaccard','jaccard_square']:
        if ref2 is None:
            ref2 = ref
        adj_ref = build_nn(ref = ref2, k=k)
        num_shared_neighbors = adj_q2r @ adj_ref.T
        num_shared_neighbors_nn = num_shared_neighbors.multiply(adj_knn.T)

        wknn = num_shared_neighbors_nn.copy()
        if weighting_scheme == 'top_n':
            if top_n is None:
                top_n = k//4 if k > 4 else 1
            wknn = (wknn > top_n) * 1
        elif weighting_scheme == "jaccard":
            wknn.data = wknn.data / (k+k-wknn.data)
        elif weighting_scheme == "jaccard_square":
            wknn.data = (wknn.data / (k+k-wknn.data)) ** 2
    else:
        wknn = adj_knn.T
        if weighting_scheme == 'gaussian':
            wknn.data = gaussian_kernel(wknn.data, sigma = sigma)
    
    if return_adjs:
        adjs = {'q2r' : adj_q2r,
                'r2q' : adj_r2q,
                'knn' : adj_knn,
                'r2r' : adj_ref}
        return (wknn, adjs)
    else:
        return wknn

# %% [cell 22 | notebook cell 23]
mouse_emb = sc.read_h5ad('workspace/hpca/data/Mouse_Atlas/data_integrated_analysed.h5ad')
human.obsm['X_emb'] = human.obsm['X_scvi'].copy()
mouse_emb = mouse_emb[mouse.obs_names]
mouse_emb
mouse.obsm['X_emb'] = mouse_emb.obsm['X_integrated'].copy()
import gc
del mouse_emb
gc.collect()

# %% [cell 23 | notebook cell 24]
mouse_meta = mouse.obs.copy()
mouse_group_key = "diabetes_model"
mouse_groups = mouse_meta.groupby(mouse_group_key).indices
mouse_groups

# %% [cell 24 | notebook cell 25]
human_meta = human.obs.copy()

human_groups = {
    "control": human_meta.query("source == 'core'").index,
    "T2D": human_meta.query("source == 'T2D_extension'").index,
    "T1D": human_meta.query("source == 'T1D_extension'").index,
    "AAB": human_meta.query("source == 'aab_extension'").index,
}

# %% [cell 25 | notebook cell 26]
cell_types = human_meta["Level_4_extension_diabetes_simplified"].unique()

# %% [cell 26 | notebook cell 27]
import os
os.makedirs('workspace/hpca/hpca_downstream/Presence_Score', exist_ok=True)

# %% [cell 27 | notebook cell 29]
human.obs['Level_4_extension_diabetes_simplified'] = human.obs.Level_4_extension_diabetes.copy()
human.obs.Level_4_extension_diabetes_simplified = np.where(human.obs.Level_3_extension_diabetes.str.contains('T1D|T2D|aab'), human.obs.Level_3_extension_diabetes, human.obs.Level_4_extension_diabetes)

# %% [cell 28 | notebook cell 30]
human.obs.groupby(['source', 'Level_4_extension']).size().unstack()

# %% [cell 29 | notebook cell 31]
human_cell_key = "Level_4_extension"
mouse_cell_key = "cell_type_reannotatedIntegrated"
endocrine_human = human[human.obs[human_cell_key].astype(str).str.contains("Alpha|Beta|Delta|Epsilon|Pancreatic Polypeptide",case=False,na=False)].copy()
mouse_endocrine_labels = ["alpha","beta","delta","gamma","epsilon","alpha+beta","alpha+delta","beta+delta","beta+gamma","delta+gamma","endo. prolif.","E endo."]

endocrine_mouse = mouse[mouse.obs[mouse_cell_key].isin(mouse_endocrine_labels)].copy()

print("Human endocrine cells:", endocrine_human.n_obs)
print("Mouse endocrine cells:", endocrine_mouse.n_obs)

print("\nMouse endocrine composition:")
print(endocrine_mouse.obs[mouse_cell_key].value_counts())

# %% [cell 30 | notebook cell 32]
import os 
os.chdir('workspace/hpca/hpca_downstream/Presence_Score_Results/')

# %% [cell 31 | notebook cell 33]
latent_key = "X_emb" 
wknn_endo = get_wknn(
    ref=endocrine_human.obsm[latent_key],
    query=endocrine_mouse.obsm[latent_key],
    k=100,
    weighting_scheme="jaccard_square"
).tocsr()

import scipy.sparse
scipy.sparse.save_npz("wknn_endo.npz", wknn_endo)
# scipy.sparse.save_npz("workspace/hpca/hpca_downstream/Presence_Score/wknn_endo.npz", wknn_endo)

# %% [cell 32 | notebook cell 34]
# import os
# os.chdir('/')

# %% [cell 33 | notebook cell 35]
# wknn_endo = sparse.load_npz("workspace/hpca/hpca_downstream/Presence_Score/wknn_endo.npz")

# %% [cell 34 | notebook cell 36]
wknn_endo

# %% [cell 35 | notebook cell 37]
latent_key = "X_emb"
mouse_model_key = "diabetes_model"
human_state_key = "Level_4_extension"  
wknn_endo = wknn_endo.tocsr()
row_sums = np.asarray(wknn_endo.sum(axis=1)).ravel()
row_sums[row_sums == 0] = 1

wknn_norm = sparse.diags(1 / row_sums) @ wknn_endo
wknn_norm = wknn_norm.tocsr()

# %% [cell 36 | notebook cell 38]
transition_prob = get_transition_prob_mat(
    endocrine_human.obsm[latent_key],
    k=50,
    symm=True
)

# Save transition_prob to disk
from scipy import sparse
sparse.save_npz("transition_prob.npz", transition_prob)

mouse_groups = endocrine_mouse.obs.groupby(mouse_model_key).indices
mouse_groups = {
    k: v for k, v in mouse_groups.items()
    if pd.notna(k) and str(k).lower() not in ["none", "nan"]
}

print("Mouse models:")
for k, v in mouse_groups.items():
    print(k, len(v))

# %% [cell 37 | notebook cell 39]
def smooth_presence_score(raw_score, transition_prob, alpha=0.5, num_rounds=100):
    """
    raw_score: vector length n_reference_cells
    transition_prob: reference transition probability matrix
    returns normalized smoothed presence score in [0, 1]
    """

    raw_score = np.asarray(raw_score).ravel()
    smooth = random_walk_with_restart(
        init=raw_score,
        transition_prob=transition_prob,
        alpha=alpha,
        num_rounds=num_rounds
    ).ravel()

    # log transform
    smooth = np.log1p(smooth)

    # trim 5th and 95th percentile
    lo, hi = np.nanpercentile(smooth, [5, 95])
    smooth = np.clip(smooth, lo, hi)

    # min-max normalize to [0, 1]
    denom = smooth.max() - smooth.min()
    if denom == 0:
        smooth_norm = np.zeros_like(smooth)
    else:
        smooth_norm = (smooth - smooth.min()) / denom

    return smooth_norm

presence_scores = pd.DataFrame(index=endocrine_human.obs_names)

for model, mouse_idx in mouse_groups.items():

    # raw weighted degree on reference cells
    # sum = frequency of human cell being connected to query cells
    raw_score = np.asarray(wknn_norm[mouse_idx, :].sum(axis=0)).ravel()

    # optional: normalize by number of mouse cells
    # this makes scores comparable across differently sized mouse models
    raw_score = raw_score / len(mouse_idx)

    presence_scores[model] = smooth_presence_score(
        raw_score=raw_score,
        transition_prob=transition_prob,
        alpha=0.5,
        num_rounds=100
    )

presence_scores[human_state_key] = endocrine_human.obs[human_state_key].values
presence_scores.to_csv('presence_scores.csv')

presence_scores.head()

# %% [cell 38 | notebook cell 40]
# presence_scores.to_csv('presence_score_results.csv', index=False)

# %% [cell 39 | notebook cell 41]
model_cols = list(mouse_groups.keys())
presence_scores["max_mouse_model"] = presence_scores[model_cols].idxmax(axis=1)
presence_scores["max_presence_score"] = presence_scores[model_cols].max(axis=1)
human.obs["mouse_max_presence_score"] = np.nan
human.obs["mouse_max_model"] = pd.Series(index=human.obs_names, dtype="object")
human.obs.loc[endocrine_human.obs_names, "mouse_max_presence_score"] = presence_scores["max_presence_score"].values
human.obs.loc[endocrine_human.obs_names, "mouse_max_model"] = presence_scores["max_mouse_model"].values
human.obs["mouse_max_model"] = human.obs["mouse_max_model"].astype("category")

# %% [cell 40 | notebook cell 42]
sc.set_figure_params(dpi_save=300, dpi=300, frameon=False, figsize=(5,5))

# %% [cell 41 | notebook cell 43]
pwd

# %% [cell 42 | notebook cell 44]
sc.pl.umap(
    human,
    color="mouse_max_presence_score",
    size=5,
    sort_order=True,
    groups=None,
    add_outline=True,
    frameon=False,
    outline_width=(0.1, 0.05),
    outline_color=['white', 'black'],
    na_color='lightgray',
    alpha=0.75,
    na_in_legend=False,
    cmap="viridis",
    save="_presence_score_umap.svg"
)

sc.pl.umap(
    human,
    color="mouse_max_model",
    size=5,
    sort_order=True,
    groups=None,
    add_outline=True,
    frameon=False,
    outline_width=(0.1, 0.05),
    outline_color=['white', 'black'],
    na_color='lightgray',
    alpha=0.75,
    na_in_legend=False,
    save="_mouse_max_model_umap.svg"
)

# %% [cell 43 | notebook cell 45]
human[human.obs.Level_4_extension.str.contains('Alpha|Beta|Delta|Epsilon|Pancreatic Polypeptide', case=False, na=False)].obs.groupby(['mouse_max_model','Level_4_extension']).size().unstack().T

# %% [cell 44 | notebook cell 46]

# Individual model maps
plot_models = [
    "T1D_NOD",
    "T1D_NOD_prediabetic",
    "T2D_db/db",
    "T2D_db/db-treated_VSG",
    "T2D_mSTZ",
]

for model in plot_models:
    if model not in presence_scores.columns:
        continue

    col = f"presence_hnoca_{model}"
    human.obs[col] = np.nan
    human.obs.loc[endocrine_human.obs_names, col] = presence_scores[model].values

sc.pl.umap(
    human,
    color=[f"presence_hnoca_{m}" for m in plot_models if f"presence_hnoca_{m}" in human.obs.columns],
    cmap="viridis",
    na_color="lightgrey",
    frameon=False,
    size=5,
    ncols=3,
    vmin=0,
    vmax=1,
)

# %% [cell 45 | notebook cell 47]
state_mean = (
    presence_scores
    .groupby(human_state_key)[model_cols]
    .mean()
    .T
)

# top states by max average presence
top_states = (
    state_mean
    .max(axis=0)
    .sort_values(ascending=False)
    .head(30)
    .index
)

plot_mat = state_mean[top_states]

# row-z for profile visualization
plot_mat_z = plot_mat.apply(
    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0,
    axis=1
)

plt.figure(figsize=(18, 7))
sns.heatmap(
    plot_mat_z,
    cmap="CMRmap",
    center=0,
    cbar_kws={"label": "Row z-scored HNOCA-style presence"}
)
plt.title("Mouse model presence profiles across HPCA Level_4_extension states")
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.show()

# %% [cell 46 | notebook cell 48]
plot_mat_z

# %% [cell 47 | notebook cell 49]
state_max = (
    presence_scores
    .groupby(human_state_key)["max_presence_score"]
    .mean()
    .sort_values(ascending=False)
)
plt.figure(figsize=(6, 8))
state_max.head(30).sort_values().plot(kind="barh")
plt.xlabel("Mean max presence score")
plt.ylabel("")
plt.title("HPCA endocrine states most represented across mouse models")
plt.tight_layout()
plt.show()

# %% [cell 48 | notebook cell 50]
low_entropy_ct = human[human.obs.entropy_cell_states_cat == 'low'].obs.Level_4_extension.unique().tolist()

# %% [cell 49 | notebook cell 51]
low_entropy_ct

# %% [cell 50 | notebook cell 52]
presence_scores.shape

# %% [cell 51 | notebook cell 53]
presence_scores[~(presence_scores.Level_4_extension.isin(low_entropy_ct))].shape

# %% [cell 52 | notebook cell 54]
state_max = (
    presence_scores[~(presence_scores.Level_4_extension.isin(low_entropy_ct))]
    .groupby(human_state_key)["max_presence_score"]
    .mean()
    .sort_values(ascending=False)
)
plt.figure(figsize=(15, 12), dpi=300)
state_max.head(30).sort_values().plot(kind="barh")
plt.xlabel("Mean max presence score")
plt.ylabel("")
plt.title("HPCA endocrine states most represented across mouse models")
plt.tight_layout()
plt.grid(False)
plt.savefig("hpca_endocrine_states_presence.svg", format="svg", dpi=300)
plt.show()

# %% [cell 53 | notebook cell 55]
state_max

# %% [cell 54 | notebook cell 56]
# Filter out 'low' Entropy_Cat from df before proceeding
filtered_states = [i for i in low_entropy_ct if i in presence_scores.Level_4_extension.unique()]

# Build a filtered presence_scores DataFrame, dropping rows whose Level_4_extension is in low_entropy_ct
presence_scores_filtered = presence_scores[~presence_scores.Level_4_extension.isin(filtered_states)]

# Calculate the mean presence scores by group and transpose
state_mean_filtered = (
    presence_scores_filtered
    .groupby(human_state_key)[model_cols]
    .mean()
    .T
)

# top states by max average presence (from filtered)
top_states = (
    state_mean_filtered
    .max(axis=0)
    .sort_values(ascending=False)
    .head(30)
    .index
)

plot_mat = state_mean_filtered[top_states]

# row-z for profile visualization
plot_mat_z = plot_mat.apply(
    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0,
    axis=1
)

plt.figure(figsize=(18, 12), dpi=300)
sns_plot = sns.heatmap(
    plot_mat_z,
    cmap="CMRmap",
    center=0,
    cbar_kws={"label": "Row z-scored HNOCA-style presence"}
)
plt.title("Mouse model presence profiles across HPCA endocrine states")
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.grid(False)
plt.savefig("mouse_model_presence_hpca_states.svg", format="svg", dpi=300)
plt.show()

# %% [cell 55 | notebook cell 57]
pd.set_option('display.max_columns', 50)
plot_mat_z

# %% [cell 56 | notebook cell 58]
endocrine_human.obs.source.unique()

# %% [cell 57 | notebook cell 59]
endocrine_human.obs['Disease'] = endocrine_human.obs.source.astype(str).copy()
endocrine_human.obs['Disease'] = endocrine_human.obs['Disease'].map({
    "T2D_extension": "Type 2 Diabetes",
    "T1D_extension": "Type 1 Diabetes",
    "AAB": "Autoantibody Positive",
    "core": "Healthy Controls",
    "aab_extension": "Autoantibody Positive",   # In case there are extension types too
    "GSE167880_extension": "Other"
}).fillna(endocrine_human.obs['Disease'])

# %% [cell 58 | notebook cell 60]
endocrine_human.obs['Disease'].value_counts()

# %% [cell 59 | notebook cell 61]
# collapse_map = {
#     # Alpha identity
#     "Alpha Cell: GCG_High": "Alpha | Identity",
#     "Alpha Cell: GCG_Mid": "Alpha | Identity",
#     "Alpha Cell: GCG_Low": "Alpha | GCG_Low",
#     "Alpha Cell: Mature Identity": "Alpha | Identity",

#     # Alpha remodeling
#     "Alpha Cell: High OxPhos": "Alpha | OxPhos",
#     "Alpha Cell: Oxidative Stress": "Alpha | Oxidative Stress",
#     "Alpha Cell: ER Stress": "Alpha | ER Stress",
#     "Alpha Cell: Interferon Response": "Alpha | Interferon",
#     "Alpha Cell: Dedifferentiation-like": "Alpha | Dedifferentiation",
#     "Alpha Cell: Proliferating": "Alpha | Proliferation",

#     # Beta identity / function
#     "Beta Cell: INS_High": "Beta | Identity",
#     "Beta Cell: Mature Identity": "Beta | Identity",
#     "Beta Cell: Secretory Granules": "Beta | Secretory",
#     "Beta Cell: INS_Mid, IAPP_High": "Beta | Secretory / IAPP-high",
#     "Beta Cell: INS_Low": "Beta | INS-low",

#     # Beta remodeling
#     "Beta Cell: High OxPhos": "Beta | OxPhos",
#     "Beta Cell: Oxidative Stress": "Beta | Oxidative Stress",
#     "Beta Cell: ER Stress": "Beta | ER Stress",
#     "Beta Cell: Interferon Response": "Beta | Interferon",
#     "Beta Cell: Dedifferentiation-like": "Beta | Dedifferentiation",
#     "Beta Cell: Proliferating": "Beta | Proliferation",

#     # Optional but useful
#     "Alpha-Beta-Delta Cell": "Polyhormonal | Alpha-Beta-Delta",
# }

# # Add collapsed program
# presence_scores["collapsed_program"] = presence_scores["Level_4_extension"].map(collapse_map)

# # Filter
# filtered_presence_scores = presence_scores[
#     presence_scores["collapsed_program"].notna()
#     & ~presence_scores["Level_4_extension"].isin(low_entropy_ct)
# ].copy()

# # Build disease-context label
# filtered_presence_scores["collapsed_context"] = (
#     filtered_presence_scores["collapsed_program"].astype(str)
#     + " | "
#     + filtered_presence_scores["disease_context"].astype(str)
# )

# print(filtered_presence_scores.shape)

# # Aggregate
# state_mean = (
#     filtered_presence_scores
#     .groupby("collapsed_context")[model_cols]
#     .mean()
#     .T
# )

# # Minimum cell count per collapsed context
# ctx_counts = filtered_presence_scores.groupby("collapsed_context").size()
# keep_ctx = ctx_counts[ctx_counts >= 100].index

# state_mean = state_mean.loc[:, keep_ctx]

# # Split alpha/beta
# alpha_cols = [c for c in state_mean.columns if c.startswith("Alpha")]
# beta_cols  = [c for c in state_mean.columns if c.startswith("Beta")]
# poly_cols  = [c for c in state_mean.columns if c.startswith("Polyhormonal")]

# # Row z-score
# plot_mat_z = state_mean.apply(
#     lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0,
#     axis=1
# )

# # Plot beta
# plt.figure(figsize=(18, 8))
# sns.heatmap(
#     plot_mat_z[beta_cols],
#     cmap="CMRmap",
#     center=0,
#     cbar_kws={"label": "Row z-scored HNOCA-style presence"}
# )
# plt.title("Mouse model presence profiles across beta-cell HPCA programs stratified by disease context")
# plt.xlabel("Beta-cell program | Human disease context")
# plt.ylabel("Mouse model / condition")
# plt.tight_layout()
# plt.show()

# # Plot alpha
# plt.figure(figsize=(18, 8))
# sns.heatmap(
#     plot_mat_z[alpha_cols],
#     cmap="CMRmap",
#     center=0,
#     cbar_kws={"label": "Row z-scored HNOCA-style presence"}
# )
# plt.title("Mouse model presence profiles across alpha-cell HPCA programs stratified by disease context")
# plt.xlabel("Alpha-cell program | Human disease context")
# plt.ylabel("Mouse model / condition")
# plt.tight_layout()
# plt.show()

# %% [cell 60 | notebook cell 62]
collapse_map = {
    # Alpha identity
    "Alpha Cell: GCG_High": "Alpha | Identity",
    "Alpha Cell: GCG_Mid": "Alpha | Identity",
    "Alpha Cell: GCG_Low": "Alpha | GCG_Low",
    "Alpha Cell: Mature Identity": "Alpha | Identity",

    # Alpha remodeling
    "Alpha Cell: High OxPhos": "Alpha | OxPhos",
    "Alpha Cell: Oxidative Stress": "Alpha | Oxidative Stress",
    "Alpha Cell: ER Stress": "Alpha | ER Stress",
    "Alpha Cell: Interferon Response": "Alpha | Interferon",
    "Alpha Cell: Dedifferentiation-like": "Alpha | Dedifferentiation",
    "Alpha Cell: Proliferating": "Alpha | Proliferation",

    # Beta identity / function
    "Beta Cell: INS_High": "Beta | Identity",
    "Beta Cell: Mature Identity": "Beta | Identity",
    "Beta Cell: Secretory Granules": "Beta | Secretory",
    "Beta Cell: INS_Mid, IAPP_High": "Beta | Secretory / IAPP-high",
    "Beta Cell: INS_Low": "Beta | INS-low",

    # Beta remodeling
    "Beta Cell: High OxPhos": "Beta | OxPhos",
    "Beta Cell: Oxidative Stress": "Beta | Oxidative Stress",
    "Beta Cell: ER Stress": "Beta | ER Stress",
    "Beta Cell: Interferon Response": "Beta | Interferon",
    "Beta Cell: Dedifferentiation-like": "Beta | Dedifferentiation",
    "Beta Cell: Proliferating": "Beta | Proliferation",

    # Optional but useful
    "Alpha-Beta-Delta Cell": "Polyhormonal | Alpha-Beta-Delta",
}

# %% [cell 61 | notebook cell 63]
presence_scores['disease_context'] = presence_scores.index.map(dict(zip(endocrine_human.obs_names, endocrine_human.obs.Disease)))

# %% [cell 62 | notebook cell 64]
presence_scores.head()

# %% [cell 63 | notebook cell 65]
presence_scores.disease_context = presence_scores.disease_context.replace('Autoantibody Positive', 'Auto-antibody positive')

# %% [cell 64 | notebook cell 66]
presence_scores.disease_context = presence_scores.disease_context.replace('Auto-antibody positive', 'Auto-antibody Positive')

# %% [cell 65 | notebook cell 67]
presence_scores.disease_context.value_counts()

# %% [cell 66 | notebook cell 68]
df_ctx = presence_scores.copy()

df_ctx["collapsed_program"] = df_ctx["Level_4_extension"].map(collapse_map)

df_ctx = df_ctx[
    df_ctx["collapsed_program"].notna()
    & ~df_ctx["Level_4_extension"].isin(low_entropy_ct)
].copy()

# keep only main disease contexts
disease_order = [
    "Healthy Controls",
    "Type 1 Diabetes",
    "Type 2 Diabetes",
    "Auto-antibody Positive",
]

df_ctx = df_ctx[df_ctx["disease_context"].isin(disease_order)].copy()

# optional minimum group size
group_counts = (
    df_ctx
    .groupby(["collapsed_program", "disease_context"])
    .size()
    .reset_index(name="n_cells")
)

valid_groups = group_counts[group_counts["n_cells"] >= 100][
    ["collapsed_program", "disease_context"]
]

df_ctx = df_ctx.merge(
    valid_groups,
    on=["collapsed_program", "disease_context"],
    how="inner"
)
df_ctx

# %% [cell 67 | notebook cell 69]

# ============================================================
# 2. Mean presence per mouse model x collapsed program x disease context
# ============================================================

mean_ctx = (
    df_ctx
    .groupby(["collapsed_program", "disease_context"])[model_cols]
    .mean()
    .reset_index()
)

long_ctx = mean_ctx.melt(
    id_vars=["collapsed_program", "disease_context"],
    value_vars=model_cols,
    var_name="mouse_model",
    value_name="presence_score"
)

# %% [cell 68 | notebook cell 70]
mean_ctx.head()

# %% [cell 69 | notebook cell 71]
long_ctx.head()

# %% [cell 70 | notebook cell 72]

# ============================================================
# 3. Context preference within each collapsed program
#    For each mouse_model + collapsed_program:
#    context_preference = disease-context score / sum(context scores)
# ============================================================

long_ctx["context_preference"] = (
    long_ctx
    .groupby(["mouse_model", "collapsed_program"])["presence_score"]
    .transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
)

# also useful: z-score across disease contexts within each model/program
long_ctx["context_z"] = (
    long_ctx
    .groupby(["mouse_model", "collapsed_program"])["presence_score"]
    .transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)
)

# %% [cell 71 | notebook cell 73]
long_ctx.head()

# %% [cell 72 | notebook cell 74]

# ============================================================
# 4. Build clean labels
# ============================================================

long_ctx["program_context"] = (
    long_ctx["collapsed_program"].astype(str)
    + " | "
    + long_ctx["disease_context"].astype(str)
)

# %% [cell 73 | notebook cell 75]
long_ctx

# %% [cell 74 | notebook cell 76]

# ============================================================
# 5. Plot beta context preference (dpi=300, svg, figsize 18x12)
# ============================================================

beta_programs = [
    p for p in sorted(long_ctx["collapsed_program"].unique())
    if p.startswith("Beta")
]

beta_df = long_ctx[long_ctx["collapsed_program"].isin(beta_programs)].copy()

# order columns by program, then disease context
beta_df["program_context"] = pd.Categorical(
    beta_df["program_context"],
    categories=[
        f"{p} | {d}"
        for p in beta_programs
        for d in disease_order
        if f"{p} | {d}" in beta_df["program_context"].values
    ],
    ordered=True
)

beta_pivot = beta_df.pivot_table(
    index="mouse_model",
    columns="program_context",
    values="context_preference",
    fill_value=0,
    observed=False
)

# %% [cell 75 | notebook cell 77]
beta_pivot #.head()

# %% [cell 76 | notebook cell 78]

fig1 = plt.figure(figsize=(18, 12), dpi=300)
sns.heatmap(
    beta_pivot,
    cmap="CMRmap",
    vmin=0,
    vmax=1,
    cbar_kws={"label": "Disease-context preference within program"}
)
plt.title("Disease-context preference of mouse models across HPCA beta-cell programs")
plt.xlabel("Beta-cell program | Human disease context")
plt.ylabel("Mouse model / condition")
plt.xticks(rotation=90)
plt.tight_layout()
plt.grid(False)
plt.savefig("beta_context_preference.svg", format="svg", dpi=300)
plt.show()

# ============================================================
# 6. Plot alpha context preference (dpi=300, svg, figsize 18x12)
# ============================================================

alpha_programs = [
    p for p in sorted(long_ctx["collapsed_program"].unique())
    if p.startswith("Alpha")
]

alpha_df = long_ctx[long_ctx["collapsed_program"].isin(alpha_programs)].copy()

alpha_df["program_context"] = pd.Categorical(
    alpha_df["program_context"],
    categories=[
        f"{p} | {d}"
        for p in alpha_programs
        for d in disease_order
        if f"{p} | {d}" in alpha_df["program_context"].values
    ],
    ordered=True
)

alpha_pivot = alpha_df.pivot_table(
    index="mouse_model",
    columns="program_context",
    values="context_preference",
    fill_value=0,
    observed=False
)

fig2 = plt.figure(figsize=(18, 12), dpi=300)
sns.heatmap(
    alpha_pivot,
    cmap="CMRmap",
    vmin=0,
    vmax=1,
    cbar_kws={"label": "Disease-context preference within program"}
)
plt.title("Disease-context preference of mouse models across HPCA alpha-cell programs")
plt.xlabel("Alpha-cell program | Human disease context")
plt.ylabel("Mouse model / condition")
plt.xticks(rotation=90)
plt.tight_layout()
plt.grid(False)
plt.savefig("alpha_context_preference.svg", format="svg", dpi=300)
plt.show()

# ============================================================
# 7. Disease-context summary per mouse model
#    Across all alpha/beta programs
# ============================================================

summary_ctx = (
    long_ctx
    .groupby(["mouse_model", "disease_context"])["presence_score"]
    .mean()
    .reset_index()
)

summary_ctx["context_preference_overall"] = (
    summary_ctx
    .groupby("mouse_model")["presence_score"]
    .transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
)

summary_pivot = summary_ctx.pivot_table(
    index="mouse_model",
    columns="disease_context",
    values="context_preference_overall",
    fill_value=0
)[disease_order]

fig3 = plt.figure(figsize=(12, 10), dpi=300)
sns.heatmap(
    summary_pivot,
    cmap="CMRmap",
    annot=False,
    fmt=".2f",
    vmin=0,
    vmax=1,
    cbar_kws={"label": "Overall disease-context preference"}
)
plt.title("Overall human disease-context preference of mouse models")
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.grid(False)

# Make xtick labels bold
ax = plt.gca()
plt.setp(ax.get_xticklabels(), fontweight="bold", rotation=0, fontsize=10)

plt.savefig("overall_disease_context_preference.svg", format="svg", dpi=300)
plt.show()

# ============================================================
# 8. Optional: identify top context preferences per mouse model
# ============================================================

top_context_hits = (
    long_ctx
    .sort_values("context_preference", ascending=False)
    .groupby("mouse_model")
    .head(10)
    .sort_values(["mouse_model", "context_preference"], ascending=[True, False])
)

top_context_hits[
    ["mouse_model", "collapsed_program", "disease_context", "presence_score", "context_preference", "context_z"]
]

# %% [cell 77 | notebook cell 79]
other_programs = [
    p for p in sorted(long_ctx["collapsed_program"].unique())
    if any(p.startswith(cell) for cell in ["Alpha", "Beta"])
]

other_df = long_ctx[long_ctx["collapsed_program"].isin(other_programs)].copy()

other_df["program_context"] = pd.Categorical(
    other_df["program_context"],
    categories=[
        f"{p} | {d}"
        for p in other_programs
        for d in disease_order
        if f"{p} | {d}" in other_df["program_context"].values
    ],
    ordered=True
)

# Calculate mean context preference per program_context (column)
context_pref_means = (
    other_df.groupby("program_context")["context_preference"]
    .mean()
    .sort_values(ascending=False)
)

# Select only the top 15 columns
top_n = 30
sorted_columns = [col for col in context_pref_means.index if col in other_df["program_context"].cat.categories][:top_n]

other_df["program_context"] = pd.Categorical(
    other_df["program_context"],
    categories=sorted_columns,
    ordered=True
)

other_pivot = other_df.pivot_table(
    index="mouse_model",
    columns="program_context",
    values="context_preference",
    fill_value=0,
    observed=False
)

# Reorder columns explicitly in the pivot table for the heatmap (top 15 only)
other_pivot = other_pivot[sorted_columns]

fig3 = plt.figure(figsize=(18, 10), dpi=300)
sns.heatmap(
    other_pivot,
    cmap="Reds",
    vmin=0,
    vmax=1,
    cbar_kws={"label": "Disease-context preference within program"}
)
plt.title(f"Disease-context preference of mouse models across Alpha & Beta Cells (Top {top_n})")
plt.xlabel("Cell program | Human disease context")
plt.ylabel("Mouse model / condition")

# Set xticks fontsize to 8 and make them bold
ax = plt.gca()
plt.setp(ax.get_xticklabels(), fontweight="bold", rotation=90, fontsize=12)

plt.tight_layout()
plt.grid(False)
plt.savefig("alpha_beta_context_preference_top15.svg", format="svg", dpi=300)
plt.show()

# %% [cell 78 | notebook cell 80]
other_pivot

# %% [cell 79 | notebook cell 81]
fig3 = plt.figure(figsize=(4, 5), dpi=300)
sns.heatmap(
    summary_pivot,
    cmap="CMRmap",
    annot=False,
    fmt=".2f",
    vmin=0,
    vmax=1,
    cbar_kws={"label": "Overall disease-context preference"}
)
plt.title("Overall human disease-context preference of mouse models")
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.grid(False)

# Make xtick labels bold
ax = plt.gca()
plt.setp(ax.get_xticklabels(), fontweight="bold", rotation=90, fontsize=8)
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

plt.savefig("overall_disease_context_preference.svg", format="svg", dpi=300)
plt.show()

# %% [cell 80 | notebook cell 82]
summary_pivot

# %% [cell 81 | notebook cell 83]
top_context_hits[
    ["mouse_model", "collapsed_program", "disease_context", "presence_score", "context_preference", "context_z"]
]

