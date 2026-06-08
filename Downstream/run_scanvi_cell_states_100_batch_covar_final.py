import scanpy as sc
import scarches as sca
from scarches.dataset.trvae.data_handling import remove_sparsity
import traceback
import os
import anndata as ad
# from scatlastb_utils.io import read_anndata
import matplotlib.pyplot as plt
import warnings
import pandas as pd

warnings.filterwarnings("ignore")
    
adata = sc.read_h5ad('/lustre/groups/ml01/workspace/hpca/hpca_downstream/2026_final_object_healthy.h5ad')
batch_key = 'batch_covar_split'
# Global + Lineage
global_hvg = sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=500, batch_key=batch_key, inplace=False)
hvg_all_L3 = global_hvg[global_hvg['highly_variable']].index.tolist()
print(f"Global HVGs for scanvi: {len(hvg_all_L3)}")
for ct in adata.obs.Level_3.unique():
    print(f"Level_3: {ct}")
    subset = adata[adata.obs.Level_3 == ct]
    hvg_df = sc.pp.highly_variable_genes(subset, flavor="seurat", n_top_genes=100, batch_key=batch_key, inplace=False)
    hvgs = hvg_df[hvg_df['highly_variable']].index.tolist()
    print(f"  HVGs for scanvi: {len(hvgs)}")
    hvg_all_L3.extend(hvgs)
# save hvg_all_L3 as a df

hvg_all_L3 = list(set(hvg_all_L3))
print(f"Total unique HVGs across Level_3: {len(hvg_all_L3)}")
gene_programs = {
"alpha_identity_mature": ["ARX","MAFB","IRX2","IRX1", "POU6F2", "GCG", "TTR", "CRYBA2"], #"PAX6"
"er_stress_upr": ["HSPA5","HSP90B1","XBP1","DDIT3","DNAJB9","HERPUD1","PDIA4"],  
"inflammation_ifn": ["ISG15","IFIT1","IFIT2","MX1","OAS1","STAT1","IRF1"],
"oxidative_stress": ["TXNIP","HMOX1","NQO1","SOD2","SQSTM1","PRDX1"],
"cell_cycle": ["MKI67","TOP2A","CDK1","CCNB1","UBE2C","MCM5"],
"dedifferentiation_immature": ["HES1","ID1","ID3","SOX4","VIM","ALDH1A3"],
"oxphos_metabolic_state": ["NDUFS2","UQCRC2","COX5B","ATP5F1A","ATP5F1B","ATP5MC1","SDHB"],
"beta_identity_mature": ["MAFA","PDX1","NKX6-1","RFX6","NEUROD1","UCN3","IAPP"],
"excitability_channels": ["KCNJ11","ABCC8","CACNA1D","RIMS2","SLC30A8","SYT7"],
"er_stress_upr": ["HSPA5","HSP90B1","XBP1","DDIT3","DNAJB9","HERPUD1","PDIA4"],
"ieg_activation": ["FOS","FOSB","JUN","JUND","EGR1","NR4A1","DUSP1"],
"secretory_granule": ["CHGA","CHGB","SCG5","CPE", "PCSK1","PCSK2"],
"dedifferentiation_immature": ["SOX9","HES1","ID1","ID3","VIM","ALDH1A3","SLC16A1","LDHA"],
"oxphos_metabolic_state": ["NDUFS2","UQCRC2","COX5B","ATP5F1A","ATP5F1B","ATP5MC1","SDHB"],
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
state_genes = [g for g in state_genes if g in adata.var_names]
hvg_all_L3 = state_genes + hvg_all_L3
hvg_all_L3 = list(set(hvg_all_L3))
ambient_genes = ['PRSS1','REG1A','REG1B','CPA1','CPA2','CTRB1','CTRB2','CTRC','CELA3A','CELA3B','AMY2A','AMY2B', "INS","IAPP","GCG","SST","PPY",]
hvg_all_L3 = list(set(hvg_all_L3) - set(ambient_genes))
df = pd.DataFrame(hvg_all_L3, columns=['gene'])
os.makedirs('/lustre/groups/ml01/workspace/hpca/hpca_downstream/Final_Embedding_Healthy', exist_ok=True)
df.to_csv('/lustre/groups/ml01/workspace/hpca/hpca_downstream/Final_Embedding_Healthy/HVGs.csv', index=False)
print(f"Total HVGs for scanvi (union of Level_3 HVGs + state_genes): {len(hvg_all_L3)}")

adata_hvg = adata[:, adata.var_names.isin(hvg_all_L3)].copy()
batch_key = 'batch_covar_split'
celltype_key = 'Level_4'
adata_hvg = adata_hvg[:, ~adata_hvg.var_names.isin(ambient_genes)].copy()
adata_hvg.obs[batch_key] = adata_hvg.obs[batch_key].astype(str).astype('category')
sca.models.SCVI.setup_anndata(adata_hvg, layer='counts', batch_key=batch_key, labels_key=celltype_key)
vae = sca.models.SCVI(
    adata_hvg,
    n_layers=4,
    encode_covariates=True,
    use_layer_norm="both",
    deeply_inject_covariates=False,
    use_batch_norm="none", n_latent=20,
)
vae.train(max_epochs=100, check_val_every_n_epoch=5, early_stopping=True, early_stopping_monitor='elbo_validation', early_stopping_patience=10)
scanvae = sca.models.SCANVI.from_scvi_model(vae, unlabeled_category = "Unknown")
scanvae.train(max_epochs=80)
# new adata with only emb
adata_emb = ad.AnnData(X=scanvae.get_latent_representation(adata_hvg))
adata_emb.obs = adata_hvg.obs.copy()
# adata.obsm['scanvi_emb_global_lineage'] = scanvae.get_latent_representation(adata_hvg)
sc.pp.neighbors(adata_emb, use_rep='X', metric='cosine', n_neighbors=100, key_added='scanvi_neighbors_global_lineage')
sc.tl.umap(adata_emb, neighbors_key='scanvi_neighbors_global_lineage', key_added='UMAP_scanvi_global_lineage', min_dist=0.75)

adata_emb.write('/lustre/groups/ml01/workspace/hpca/hpca_downstream/Final_Embedding_Healthy/final_embedding_healthy.h5ad')
scanvae.save('/lustre/groups/ml01/workspace/hpca/hpca_downstream/Final_Embedding_Healthy/scanvi_model_healthy', overwrite=True)

