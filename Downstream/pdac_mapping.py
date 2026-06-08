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

# MAP PDAC TO EXTENDED ATLAS
print("Mapping PDAC to Extended Atlas")

pdac = sc.read_h5ad('/lustre/groups/ml01/workspace/shrey.parikh/PDAC_Work_Dir/PDAC_Final/Human_Atlas_Harmonised_genes_filtered.h5ad')
epi_pdac = pdac[pdac.obs.Level_4.str.contains('Acinar|Ductal|Malignant', na=False)]
adata_hvg_epi = sc.read_h5ad('/lustre/groups/ml01/workspace/hpca/hpca_downstream/Epithelial_PDAC_Extension/healthy_exocrine_integration_extended_atlas_500.h5ad')

epi_pdac_hvg = epi_pdac[:, epi_pdac.var_names.isin(adata_hvg_epi.var_names)]
del pdac
import gc
gc.collect()

missing = [g for g in adata_hvg_epi.var_names if g not in epi_pdac_hvg.var_names]
if missing:
    print(f"Missing genes in epi_pdac_hvg: {missing}")
    X_missing = sp.csr_matrix((epi_pdac_hvg.n_obs, len(missing)))
    X_new = sp.hstack([epi_pdac_hvg.X, X_missing], format="csr")
    var_missing = pd.DataFrame(index=missing, columns=epi_pdac_hvg.var.columns)
    var_new = pd.concat([epi_pdac_hvg.var.copy(), var_missing], axis=0)
    epi_pdac_hvg = ad.AnnData(
        X=X_new,
        obs=epi_pdac_hvg.obs.copy(),
        var=var_new
    )

epi_pdac_hvg = epi_pdac_hvg[:, adata_hvg_epi.var_names].copy()
epi_pdac_hvg.layers['counts'] = epi_pdac_hvg.X.copy()
epi_pdac_hvg.obs['batch_covar_split'] = epi_pdac_hvg.obs['Dataset_ID'].astype(str).values
epi_pdac_hvg.obs.rename(columns={'Level_4': 'Level_4_PDAC'}, inplace=True)


model = sca.models.SCANVI.load(dir_path="/lustre/groups/ml01/workspace/hpca/hpca_downstream/Epithelial_PDAC_Extension/scanvi_model_epithelial_extended_atlas_500", adata=adata_hvg_epi)
model = sca.models.SCANVI.load_query_data(epi_pdac_hvg, '/lustre/groups/ml01/workspace/hpca/hpca_downstream/Epithelial_PDAC_Extension/scanvi_model_epithelial_extended_atlas_500', 
                                          freeze_dropout = True)
model._unlabeled_indices = np.arange(epi_pdac_hvg.n_obs)
model._labeled_indices = []
print("Labelled Indices: ", len(model._labeled_indices))
print("Unlabelled Indices: ", len(model._unlabeled_indices))
model.train(max_epochs=40, plan_kwargs=dict(weight_decay=0.0), check_val_every_n_epoch=5)
adata_concat = adata_hvg_epi.concatenate(epi_pdac_hvg, batch_key="condition",batch_categories=["Healthy", "PDAC"])
adata_emb = ad.AnnData(X=model.get_latent_representation(adata_concat))
adata_emb.obs = adata_concat.obs.copy()


try:
    adata_emb.obs["predictions"] = model.predict()
    print("Successfully computed predictions and latent on concatenated object")
except Exception as e:
    print(f"Falling back to PDAC-only prediction/latent: {e}")
    pdac_predictions = model.predict()
    adata_emb.obs["predictions"] = "Reference"
    pdac_mask = adata_emb.obs["condition"] == "PDAC"
    adata_emb.obs.loc[pdac_mask, "predictions"] = pdac_predictions
    query_latent = model.get_latent_representation(epi_pdac_hvg)
    ref_model = sca.models.SCANVI.load("/lustre/groups/ml01/workspace/hpca/hpca_downstream/Epithelial_PDAC_Extension/scanvi_model_epithelial_extended_atlas_500",adata=adata_hvg_epi,)
    ref_latent = ref_model.get_latent_representation(adata_hvg_epi)
    adata_emb.obsm["X_scanvi"] = np.vstack([ref_latent, query_latent])
    print("Fallback completed successfully")

sc.pp.neighbors(adata_emb, use_rep='X', n_neighbors=100, metric='cosine')
sc.tl.umap(adata_emb, min_dist=0.75)
adata_emb.write('/lustre/groups/ml01/workspace/hpca/hpca_downstream/Epithelial_PDAC_Extension/healthy_exocrine_pdac_extension_embedding_500.h5ad')
model.save('/lustre/groups/ml01/workspace/hpca/hpca_downstream/Epithelial_PDAC_Extension/scanvi_model_epithelial_pdac_extension_extended_atlas_500', overwrite=True)