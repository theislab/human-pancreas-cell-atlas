import os
import scanpy as sc
import torch
import scarches as sca
import matplotlib.pyplot as plt
import numpy as np
import gdown
import pandas as pd
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)

adata = sc.read_h5ad('/lustre/groups/ml01/workspace/hpca/hpca_downstream/2026_final_object_healthy.h5ad')
adata_epi = adata[adata.obs.Level_4.str.contains('Acinar|Ductal')]
batch_key = 'batch_covar_split'
global_hvg = sc.pp.highly_variable_genes(adata_epi, flavor="seurat", n_top_genes=500, batch_key=batch_key, inplace=False)
hvg_all_L3 = global_hvg[global_hvg['highly_variable']].index.tolist()
print(f"Global HVGs for scanvi: {len(hvg_all_L3)}")
for ct in adata_epi.obs.Level_3.unique():
    print(f"Level_3: {ct}")
    subset = adata_epi[adata_epi.obs.Level_3 == ct]
    hvg_df = sc.pp.highly_variable_genes(subset, flavor="seurat", n_top_genes=500, batch_key=batch_key, inplace=False)
    hvgs = hvg_df[hvg_df['highly_variable']].index.tolist()
    print(f"  HVGs for scanvi: {len(hvgs)}")
    hvg_all_L3.extend(hvgs)

hvg_all_L3 = list(set(hvg_all_L3))
print(f"Total unique HVGs across Level_3: {len(hvg_all_L3)}")
gene_programs = {
"acinar_digestive_secreting":["PRSS1","PRSS2","CPA1","CTRB1","CTRB2","CPB1","AMY2A","CELA2B","CLPS"],
"acinar_idling":["INSR","RBPJL","FOXP2","MAP3K5","KLF9","FOSL2"],
"acinar_reg_injury":["REG3A","REG3G","REG4","LCN2","S100A10","S100A11","CFD","LRIG1"],
"acinar_plasticity_adm":["SOX9","KRT19","MMP7","SPP1","KRT17","CLDN4","ANXA13"],
"ductal_transport_cftr":["CFTR","SLC4A4","CA2","AQP1","CLDN4","SLC26A6"],
"ductal_mucinous":["MUC1","MUC5B","TFF1","TFF2","AGR2","SPDEF","KLF5"],
"ductal_reactive_injury":["SPP1","MMP7","KRT17","LGALS3","ANXA3","S100A10","LCN2"],
"ductal_emt_transitional":["VIM","ITGA6","SNAI2","ZEB1","TAGLN2","EMP1"],
"ductal_ionocyte_like":["FOXI1","CFTR","ASCL3","ATP6V1B1","ATP6V0D2","BSND"],
}


state_genes = set()
for prog, genes in gene_programs.items():
        state_genes |= set(genes)
state_genes = [g for g in state_genes if g in adata_epi.var_names]
hvg_all_L3 = state_genes + hvg_all_L3
hvg_all_L3 = list(set(hvg_all_L3))
ambient_genes = ['PRSS1','REG1A','REG1B','CPA1','CPA2','CTRB1','CTRB2','CTRC','CELA3A','CELA3B','AMY2A','AMY2B', "INS","IAPP","GCG","SST","PPY",]
hvg_all_L3 = list(set(hvg_all_L3) - set(ambient_genes))
df = pd.DataFrame(hvg_all_L3, columns=['gene'])
print(f"Total HVGs for scanvi (union of Level_3 HVGs + state_genes): {len(hvg_all_L3)}")
df.to_csv('/lustre/groups/ml01/workspace/hpca/hpca_downstream/Epithelial_PDAC_Extension/Epi_HVGs_Extended_Atlas_500.csv', index=False)

adata_hvg_epi = adata_epi[:, adata_epi.var_names.isin(hvg_all_L3)].copy()
batch_key = 'batch_covar_split'
celltype_key = 'Level_4'
adata_hvg_epi = adata_hvg_epi[:, ~adata_hvg_epi.var_names.isin(ambient_genes)].copy()
adata_hvg_epi.obs[batch_key] = adata_hvg_epi.obs[batch_key].astype(str).astype('category')
sca.models.SCVI.setup_anndata(adata_hvg_epi, layer='counts', batch_key=batch_key, labels_key=celltype_key)
vae = sca.models.SCVI(
    adata_hvg_epi,
    n_layers=4,
    encode_covariates=True,
    use_layer_norm="both",
    deeply_inject_covariates=False,
    use_batch_norm="none", n_latent=20,
)
vae.train(max_epochs=50, check_val_every_n_epoch=5, early_stopping=True, early_stopping_monitor='elbo_validation', early_stopping_patience=10)
scanvae = sca.models.SCANVI.from_scvi_model(vae, unlabeled_category = "Unknown")
scanvae.train(max_epochs=40)
# new adata with only emb
adata_hvg_epi.obsm['X_emb_epi'] = scanvae.get_latent_representation(adata_hvg_epi)
sc.pp.neighbors(adata_hvg_epi, use_rep='X_emb_epi', metric='cosine', n_neighbors=50, key_added='scanvi_epi_neighbors')
sc.tl.umap(adata_hvg_epi, neighbors_key='scanvi_epi_neighbors', min_dist=0.75)
scanvae.save('/lustre/groups/ml01/workspace/hpca/hpca_downstream/Epithelial_PDAC_Extension/scanvi_model_epithelial_extended_atlas_500', overwrite=True)
adata_hvg_epi.write('/lustre/groups/ml01/workspace/hpca/hpca_downstream/Epithelial_PDAC_Extension/healthy_exocrine_integration_extended_atlas_500.h5ad')
