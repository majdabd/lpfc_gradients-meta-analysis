# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Import Packages

# %%
# %matplotlib inline
from typing import Callable, Iterable
import nilearn
from nilearn import plotting
import nilearn.datasets as datasets
import nilearn.image as image
from nilearn import surface
from sklearn.utils import shuffle
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap

from brainspace.gradient import GradientMaps
from brainspace.utils.parcellation import map_to_labels


from neurolang.frontend import NeurolangPDL, ExplicitVBR
import data_utils
from data_utils import vox2mm, merge_difumo_network_label, truncate_colormap, create_coordinate_dataset

# %% [markdown]
# ## Call NeuroLang's Probablistic Frontend

# %%
nl = NeurolangPDL()

# %% [markdown]
# ## Load Grey Matter Mask and Add it as Tuple List (i.e. Table) in NeuroLang

# %%
# Grey Matter Mask
mni_mask = image.resample_img(
    
    nib.load(datasets.fetch_icbm152_2009()["gm"]), np.eye(3)*3, interpolation="nearest"
)

plotting.plot_stat_map(mni_mask, title='Grey Matter Mask')

# Extract GM voxel coordinates (probability > 0.25) and store them in table "GreyMatterVoxel"
gm_matrix_coordinates = np.array(np.where(mni_mask.get_fdata() > 0.25)).T

gm_xyz_coordinates = pd.DataFrame(vox2mm(gm_matrix_coordinates, affine=mni_mask.affine), 
                                  columns=["x", "y", "z"])

nl.add_tuple_set(gm_xyz_coordinates, name="GreyMatterVoxel")

# %% [markdown]
# ## Load LPFC mask and Add it as a Table in NeuroLang

# %%
# Lateral Prefrontal Cortex Mask
lpfc_mask=image.resample_to_img(source_img=nib.load('lpfc_mask.nii'), target_img=mni_mask,
                                        interpolation='nearest')

plotting.plot_roi(lpfc_mask, cut_coords=(45,30,30), title='Lateral Prefrontal Cortex Mask')



# Extract LPFC voxel coordinates and store them in table "LateralPFCVoxel"
lpfc_matrix_coordinates=np.array(np.where(lpfc_mask.get_fdata() != 0)).T 

lpfc_xyz_coordinates = pd.DataFrame(vox2mm(lpfc_matrix_coordinates, affine=mni_mask.affine),
                                   columns = ["x", "y", "z"])

nl.add_tuple_set(lpfc_xyz_coordinates, name="LPFCVoxel")

# %% [markdown]
# ## Load Neurosynth Data and Add them to NeuroLang

# %%
# Neurosynth Data
term_in_study, peak_reported, study_ids = data_utils.fetch_neurosynth(
    tfidf_threshold=0.001,
)

# Store Peak Activation Coordinates in a "PeakReported" table
nl.add_tuple_set(peak_reported, name="PeakReported")
# Store Study IDs in a "Study" table
nl.add_tuple_set(study_ids.drop_duplicates(), name="Study")
# Add uniform weights for study selection in the meta-analysis and store them in "SelectedStudy" table
nl.add_uniform_probabilistic_choice_over_set(study_ids.drop_duplicates(), name="SelectedStudy")

# %% [markdown]
# ## Load Difumo-1024 Atlas Data and Add them to NeuroLang

# %%
#Difumo 1024 Regions Atlas
region_voxels, difumo_meta = data_utils.fetch_difumo(
    mask=mni_mask,
    coord_type="xyz",
    n_components=1024,
)


nl.add_tuple_set(region_voxels.drop_duplicates(), name="RegionVoxel")

# %% [markdown]
# ## Define Functions to be used in NeuroLang Programs

# %%
EPS = 1e-20
@nl.add_symbol
def log_odds(p, p0):
    a = min(p/(1-p) + EPS, 1.)
    b = min(p0/(1-p0) + EPS, 1.)
    logodds = np.log10(a/b)
    return logodds

@nl.add_symbol
def weighted_center(coords: Iterable[int], w: Iterable[float]) -> int:
    return int(np.sum(np.multiply(w, coords)) / np.sum(w))


@nl.add_symbol
def agg_int_to_str(a,b):
    d = min(a, b)
    e = max(a,b)
    return str("%d-%d"%(d, e))

@nl.add_symbol
def Euclidean_Dist(a, b, c, d, e, f):
    x = np.array((a, b, c))
    y = np.array((d, e, f))
    temp = x-y
    dist = np.sqrt(np.dot(temp.T, temp))
    return dist

@nl.add_symbol
def maximum(w: Iterable[float]) -> float:
    return np.max(w)


# %% [markdown]
# ## Infer the VoxelReported table 

# %%
with nl.scope as e:
    voxel_reported=nl.execute_datalog_program("""
    
    VoxelReported(x, y, z, study) :- GreyMatterVoxel(x, y, z) & PeakReported(x2, y2, z2, study) & (distance == EUCLIDEAN(x, y, z, x2, y2, z2)) & (distance < 10)
    
    ans(x, y, z, study) :- VoxelReported(x, y, z, study) 
    """)

nl.add_tuple_set(voxel_reported.as_pandas_dataframe().drop_duplicates(), name="VoxelReported")

# %% [markdown]
# ## Finding Regions within the LPFC mask

# %%
with nl.scope as e:
    lpfc_regions = nl.execute_datalog_program("""
    
    RegionVolume(r, count(x, y, z)) :- RegionVoxel(r, x, y, z, w)
    
    VolumeOfOverlapWithMask(r, count(x, y, z)) :- RegionVoxel(r, x, y, z, w) & LPFCVoxel(x, y, z)
    
    LPFCRegion(r) :- RegionVolume(r, v0) & VolumeOfOverlapWithMask(r, v) & (v/v0 > 0.5)
    
    ans(r) :- LPFCRegion(r) 
    """)
nl.add_tuple_set(lpfc_regions.as_pandas_dataframe().drop_duplicates(), name="LPFCRegion")    

# %% [markdown]
# ## Inferring MetaAnalytic Connectivity Matrix

# %%
with nl.scope as e:
    connectivity_matrix = nl.execute_datalog_program("""
    
    RegionMaxWeight(r, max(w)) :- RegionVoxel(r, x, y, z, w) 
    
    RegionVoxelNormalizedWeight(r, x, y, z) :: w/W :- RegionVoxel(r, x, y, z, w) & RegionMaxWeight(r, W)
    
    LPFCRegionActive(r, study) :- RegionVoxelNormalizedWeight(r, x, y, z) & VoxelReported(x, y, z, study) & LPFCRegion(r)
    
    LPFCRegionNotActive(r, study) :- LPFCRegion(r) & ~LPFCRegionActive(r, study) & Study(study) 
    
    BrainRegionActive(r, study) :- VoxelReported(x, y, z, study) & RegionVoxelNormalizedWeight(r, x, y, z)
    
    ProbabilityOfCoactivation(r, r2, PROB) :- BrainRegionActive(r, study) // (LPFCRegionActive(r2, study) & SelectedStudy(study))
    
    ProbabilityOfNoCoactivation(r, r2, PROB) :- BrainRegionActive(r, study) // (LPFCRegionNotActive(r2, study) & SelectedStudy(study))
    
    MetaAnalyticConnectivityMatrix(r2, r, max(p1, p0)) :- ProbabilityOfCoactivation(r, r2, p1) & ProbabilityOfNoCoactivation(r, r2, p0)
    
    ans(r2, r, LOR) :-  MetaAnalyticConnectivityMatrix(r2, r, LOR)
    """)


# %% [markdown]
# ## Estimating the Gradients Using Diffusion Embedding

# %%
#compute similarity matrix using eta-squared metric and then derive gradients using PCA or DifussionEmbedding
def eta2(X): 
    S = np.zeros((X.shape[0],X.shape[0]))
    for i in range(0,X.shape[0]):
        for j in range(i,X.shape[0]):
            mi = np.mean([X[i,:],X[j,:]],0) 
            mm = np.mean(mi)
            ssw = np.sum(np.square(X[i,:]-mi) + np.square(X[j,:]-mi))
            sst = np.sum(np.square(X[i,:]-mm) + np.square(X[j,:]-mm))
            S[i,j] = 1-ssw/sst
    
    S += S.T 
    S -= np.eye(S.shape[0])    
    return S

similarity_mat = eta2(connectivity_matrix.fillna(0).values)
gradient_maps = GradientMaps(n_components=20, approach="dm")
gradient_maps.fit(similarity_mat, sparsity=0.0)

grads = pd.DataFrame(gm.gradients_, index=coactivation_mat.index)
grads_join = grads.join(region_voxels.set_index('region'))
i, j, k = np.round(
    nib.affines.apply_affine(
        np.linalg.inv(mni_mask.affine), grads_join[["x", "y", "z"]].values
)).T.astype(int)

gradient = np.zeros(mni_mask.shape)
gradient[i, j, k] = pd.qcut(grads_join[0], 5, labels=np.arange(1, 6)) 
lpfc_mask = np.zeros_like(gradient)
lpfc_mask[i, j, k] = 1
gradient = nib.Nifti1Image(gradient, affine=mni_mask.affine)
lpfc_mask = nib.Nifti1Image(lpfc_mask, affine=mni_mask.affine)

plotting.plot_img_on_surf(gradient, mask_img=lpfc_mask, threshold=1, symmetric=False, cmap="gnuplot")
plt.show()

plt.plot(gm.lambdas_ / gm.lambdas_.sum()); plt.show()
