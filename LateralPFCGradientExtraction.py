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

# %%
# %matplotlib inline
from typing import Callable, Iterable
from nilearn import plotting
import nilearn.datasets as datasets
import nilearn.image as image
from nilearn import surface
import nilearn
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
import seaborn as sns
from scipy import special
from brainspace.gradient import GradientMaps
from brainspace.utils.parcellation import map_to_labels
from scipy import special
from scipy.spatial import distance
from neurolang.frontend import NeurolangPDL, ExplicitVBR
import data_utils
from sklearn.utils import shuffle
from data_utils import merge_difumo_network_label, truncate_colormap, create_coordinate_dataset
# %% [markdown]
# ### Load Grey Matter Mask and Lateral Prefrontal Cortex Mask

# %%
# MNI Grey Matter Mask
mni_mask = image.resample_img(
    
    nib.load(datasets.fetch_icbm152_2009()["gm"]), np.eye(3)*3, interpolation="nearest"
)

gm_voxels = pd.DataFrame(np.array(np.where(mni_mask.get_fdata() > 0.25)).T, columns=["i", "j", "k"])
xyz_coordinates = nib.affines.apply_affine(mni_mask.affine, gm_voxels[["i", "j", "k"]].values)
gm_voxels.insert(loc=3, column="x", value=xyz_coordinates[:, 0])
gm_voxels.insert(loc=4, column="y", value=xyz_coordinates[:, 1])
gm_voxels.insert(loc=5, column="z", value=xyz_coordinates[:, 2])

plotting.plot_stat_map(mni_mask)

# Lateral Frontal Cortex Mask
lfc_mask=image.resample_to_img(source_img=nib.load('lpfc_mask.nii'), target_img=mni_mask,
                                        interpolation='nearest')
lfc_voxels=pd.DataFrame(np.array(np.where(lfc_mask.get_fdata() != 0)).T, columns=["i", "j", "k"])
xyz_coordinates = nib.affines.apply_affine(mni_mask.affine, lfc_voxels[["i", "j", "k"]].values)
lfc_voxels.insert(loc=3, column="x", value=xyz_coordinates[:, 0])
lfc_voxels.insert(loc=4, column="y", value=xyz_coordinates[:, 1])
lfc_voxels.insert(loc=5, column="z", value=xyz_coordinates[:, 2])
plotting.plot_roi(lfc_mask)

# %% [markdown]
# ## Load Neurosynth Database and the Difumo Brain Atlas

# %%
# Neurosynth Data
term_in_study, peak_reported, study_ids = data_utils.fetch_neurosynth(
    tfidf_threshold=0.001,
)

study_ids = study_ids.drop_duplicates()

# %%
shuf = peak_reported[["x","y","z"]].values
for i in range(100):
    shuf = shuffle(shuf)
peak_shuffle = pd.DataFrame(np.hstack([shuf]), columns=["x", "y", "z"])
peak_shuffle["study_id"] = peak_reported["study_id"].copy().values

# %%
#Difumo 1024 Regions Atlas
region_voxels, difumo_meta = data_utils.fetch_difumo(
    mask=mni_mask,
    coord_type="xyz",
    n_components=1024,
)

# %%
region_voxels = region_voxels.rename(columns={"label":"region"})

# %% [markdown]
#
# ## Initialize NeuroLang Probabilistic Frontend and Add Useful Functions

# %%
nl = NeurolangPDL()

# %%
nl.add_symbol(
        lambda it: int(sum(it)),
        name="agg_sum",
        type_=Callable[[Iterable], int],
    )

@nl.add_symbol
def agg_count(*iterables) -> int:
    return len(next(iter(iterables)))

EPS = 1e-20

@nl.add_symbol
def log_odds(p, p0):
    a = min(p/(1-p) + EPS, 1.)
    b = min(p0/(1-p0) + EPS, 1.)
    logodds = np.log10(a/b)
    return logodds

@nl.add_symbol
def Bayes_factor(p, p0):
    a = p/(1-p)
    b = p0/(1-p0)
    BF= a/b
    return BF

@nl.add_symbol
def weighted_center(coords: Iterable[int], w: Iterable[float]) -> int:
    return int(np.sum(np.multiply(w, coords)) / np.sum(w))

@nl.add_symbol
def agg_noisy_or(probabilities: Iterable[float]) -> float:
    return 1 - np.prod(1 - probabilities)

@nl.add_symbol
def startswith(prefix: str, s: str) -> bool:
    return s.startswith(prefix)

@nl.add_symbol
def not_nan(c):
    return ~np.isnan(c)

@nl.add_symbol
def agg_create_region_overlay(
    i: Iterable, j: Iterable, k: Iterable,
) -> ExplicitVBR:
    voxels = np.c_[i, j, k]
    return ExplicitVBR(
        voxels, mni_mask.affine, image_dim=mni_mask.shape
    )

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


# %% [markdown]
# ## Define Relations and Tuples

# %%
PeakReported = nl.add_tuple_set(peak_reported[["x","y","z", "study_id"]], name="PeakReported")
Study = nl.add_tuple_set(study_ids, name="Study")
SelectedStudy = nl.add_uniform_probabilistic_choice_over_set(
    study_ids, name="SelectedStudy"
)


# %%
Region = nl.add_tuple_set({(row["Difumo_names"],) for _, row in difumo_meta.iterrows()}, name="Region")
RegionVoxel     = nl.add_tuple_set(region_voxels, name="RegionVoxel")
LfcVoxel       = nl.add_tuple_set(lfc_voxels[["x", "y", "z"]], name="LpfcVoxel")
GreyMatterVoxel = nl.add_tuple_set(gm_voxels[["x", "y", "z"]], name="GreyMatterVoxel")

# %% [markdown]
# ## Select Difumo Regions that fall in an Intersection Lateral PFC Mask 
#
# ### Regions Can be Right, Left, or Bilateral

# %%
with nl.scope as e:
    # Compute center of mass (COM)
    e.RegionCenter_x[e.r, e.weighted_center(e.x, e.w)] = e.RegionVoxel(e.r, e.x, e.y, e.z, e.w)
    e.RegionCenter_y[e.r, e.weighted_center(e.y, e.w)] = e.RegionVoxel(e.r, e.x, e.y, e.z, e.w)
    e.RegionCenter_z[e.r, e.weighted_center(e.z, e.w)] = e.RegionVoxel(e.r, e.x, e.y, e.z, e.w)
    
    e.RegionCenter[e.r, e.x, e.y, e.z] = (
          e.RegionCenter_x[e.r, e.x]
        & e.RegionCenter_y[e.r, e.y]
        & e.RegionCenter_z[e.r, e.z]
    )
    
    # Left LPFC Region 
    e.LeftRegion[e.r, e.x, e.y, e.z, e.w] = (
          RegionVoxel(e.r, e.x, e.y, e.z, e.w)
        & e.RegionCenter(e.r, e.x1, e.y1, e.z1)
        & (e.x1 < 0)
    )
    
    # Right LPFC Region
    e.RightRegion[e.r, e.x, e.y, e.z, e.w] = (
        RegionVoxel(e.r, e.x, e.y, e.z, e.w)
        & e.RegionCenter(e.r, e.x1, e.y1, e.z1)
        & (e.x1 > 0)
    )
        
    e.RegionVolume[e.r, e.agg_count(e.x, e.y, e.z)] = RegionVoxel(
        e.r, e.x, e.y, e.z
    )

    e.RegionOverlap[e.r, e.agg_count(e.x, e.y, e.z)]= (
      e.RegionVoxel(e.r, e.x, e.y, e.z)
    & LfcVoxel(e.x, e.y, e.z)
    )
    
    # Assess total overlap with mask
    e.LpfcRegionOverlap[e.r, e.x, e.y, e.z, e.w] = (
         e.RightRegion(e.r, e.x, e.y, e.z, e.w)
        & e.RegionVolume(e.r, e.v)
        & e.RegionOverlap(e.r, e.vo)
        & (e.vo > 0.5*e.v)
    )

    # Queries
    lpfc_region = nl.query(
        (e.r, e.x, e.y, e.z, e.w),
         e.LpfcRegionOverlap(e.r, e.x, e.y, e.z, e.w)
         )

lpf=lpfc_region.as_pandas_dataframe()
LateralPFCRegionVoxel = nl.add_tuple_set(lpf[['r', 'x', 'y', 'z', 'w']], name='LateralPFCRegionVoxel')
LateralPFCRegion = nl.add_tuple_set(lpf[['r']].drop_duplicates(), name='LateralPFCRegion')
np.unique(lpfc_region.as_pandas_dataframe()['r'].values).shape


# %%
lpf.to_csv("lpfc_regions_LH.csv", index=False)

# %%
with nl.scope as e:
    # Compute center of mass (COM)
    e.RegionCenter_x[e.r, e.weighted_center(e.x, e.w)] = e.RegionVoxel(e.r, e.x, e.y, e.z, e.w)
    e.RegionCenter_y[e.r, e.weighted_center(e.y, e.w)] = e.RegionVoxel(e.r, e.x, e.y, e.z, e.w)
    e.RegionCenter_z[e.r, e.weighted_center(e.z, e.w)] = e.RegionVoxel(e.r, e.x, e.y, e.z, e.w)
    
    e.RegionCenter[e.r, e.x, e.y, e.z] = (
          e.RegionCenter_x[e.r, e.x]
        & e.RegionCenter_y[e.r, e.y]
        & e.RegionCenter_z[e.r, e.z]
    )
    
    sol = nl.query((e.r, e.x, e.y, e.z), e.RegionCenter[e.r, e.x, e.y, e.z])

RegionCenter = nl.add_tuple_set(sol.as_pandas_dataframe(), name="RegionCenter")

# %%
with nl.scope as e: 
    e.RegionVolume[e.r, e.agg_count(e.x, e.y, e.z)] = RegionVoxel(
        e.r, e.x, e.y, e.z
    )

    e.RegionOverlap[e.r, e.agg_count(e.x, e.y, e.z)]= (
      e.RegionVoxel(e.r, e.x, e.y, e.z)
      & GreyMatterVoxel(e.x, e.y, e.z)
    )
        # Assess total overlap with mask
    e.BrainRegion[e.r, e.x, e.y, e.z, e.w] = (
           e.RegionVoxel(e.r, e.x, e.y, e.z, e.w)
        & e.RegionVolume(e.r, e.v)
        & e.RegionOverlap(e.r, e.vo)
        & (e.vo > 0.001*e.v)
    )
        # Queries
    brain_region = nl.query(
        (e.r, e.x, e.y, e.z, e.w),
         e.BrainRegion(e.r, e.x, e.y, e.z, e.w)
         )

# %%
BrainVoxel=nl.add_tuple_set(brain_region.as_pandas_dataframe()[['r', 'x', 'y', 'z', 'w']], name="BrainVoxel")
BrainRegion = nl.add_tuple_set(brain_region.as_pandas_dataframe()[['r']].drop_duplicates(), name="BrainRegion")

# %% [markdown]
# ## Estimate the Co-activation Matrix
#

# %%
with nl.scope as e:
    
        e.VoxelReported[e.x, e.y, e.z, e.s] = (
            e.PeakReported(e.x2, e.y2, e.z2, e.s) 
            & e.GreyMatterVoxel(e.x, e.y, e.z) 
            & (e.d == e.EUCLIDEAN(e.x, e.y, e.z, e.x2, e.y2, e.z2))  
            & (e.d < 10)
        )
        
        e.FrontalRegionVoxelReported[e.fr, e.x, e.y, e.z, e.w, e.s] = (
            e.LateralPFCRegionVoxel(e.fr, e.x, e.y, e.z, e.w)
            & e.VoxelReported[e.x, e.y, e.z, e.s]
        )
        
        (e.FrontalRegionReported @e.max(e.w))[e.fr, e.s] = (
            e.FrontalRegionVoxelReported[e.fr, e.x, e.y, e.z, e.w, e.s]
        )
        
        e.FrontalRegionNotReported[e.fr, e.s] = (
            ~e.FrontalRegionReported[e.fr, e.s]
            & e.LateralPFCRegion(e.fr)
            & e.Study(e.s)
        ) 
            
        e.BrainRegionVoxelReported[e.r, e.x, e.y, e.z, e.w, e.s] = (
            e.BrainVoxel(e.r, e.x, e.y, e.z, e.w)
            & e.VoxelReported[e.x, e.y, e.z, e.s]
        )
        
        (e.BrainRegionReported @e.max(e.w))[e.r, e.s]=(
            e.BrainRegionVoxelReported(e.r, e.x, e.y, e.z, e.w, e.s)
        )
        
        e.Distance[e.fr, e.r, e.d] = (
          e.LateralPFCRegion(e.fr) 
        & e.BrainRegion(e.r)
        & e.RegionCenter(e.fr, e.x, e.y, e.z)
        & e.RegionCenter(e.r, e.x2, e.y2, e.z2)
        & (e.d == e.Euclidean_Dist(e.x, e.y, e.z, e.x2, e.y2, e.z2)) 
        )
        
        
        e.ProbRegionGivenFrontal[e.r, e.fr, e.PROB(e.r, e.fr)]=(
            e.BrainRegionReported(e.r, e.s)) // (e.FrontalRegionReported(e.fr, e.s)
                                                     & SelectedStudy(e.s))                
        
        e.ProbRegionGivenNotFrontal[e.r, e.fr, e.PROB(e.r, e.fr)]=(
            e.BrainRegionReported(e.r, e.s)) // (e.FrontalRegionNotReported(e.fr, e.s)
                                                     & SelectedStudy(e.s))  
        

        e.Query[e.r, e.fr, e.logodds]= (
            e.ProbRegionGivenFrontal(e.r, e.fr, e.p)
            & e.ProbRegionGivenNotFrontal(e.r, e.fr, e.p0) 
            & (e.r != e.fr) 
            & (e.logodds == e.log_odds(e.p, e.p0))  
        )
        
        e.Query20mm[e.r, e.fr, e.logodds] = (
        e.Query[e.r, e.fr, e.logodds]
        & e.Distance[e.fr, e.r, e.d] 
        & (e.d > 20)
        )
        
        e.Query40mm[e.r, e.fr, e.logodds] = (
        e.Query[e.r, e.fr, e.logodds]
        & e.Distance[e.fr, e.r, e.d] 
        & (e.d > 40)
        )
            
        e.Query60mm[e.r, e.fr, e.logodds] = (
        e.Query[e.r, e.fr, e.logodds]
        & e.Distance[e.fr, e.r, e.d] 
        & (e.d > 60)
        )
        
        e.QueryNoFrontal[e.r, e.fr, e.logodds] = (
        e.Query[e.r, e.fr, e.logodds]
        & ~e.LateralPFCRegion(e.r)
        )
        
        
        sol20 = nl.query(
                (e.r, e.fr, e.logodds), 
               e.Query20mm[e.r, e.fr, e.logodds] 
            ) 
        
                
        sol40 = nl.query(
                (e.r, e.fr, e.logodds), 
               e.Query40mm[e.r, e.fr, e.logodds] 
            )  
        
                
        sol60 = nl.query(
                (e.r, e.fr, e.logodds), 
               e.Query60mm[e.r, e.fr, e.logodds] 
            )  
        
                
        solNo = nl.query(
                (e.r, e.fr, e.logodds), 
               e.QueryNoFrontal[e.r, e.fr, e.logodds] 
            )  

# %%
solution = pd.read_csv("SingleSubjects/RH/coactivation_data_population.csv")
# solution = solution.loc[:, ~solution.columns.str.contains('^Unnamed')]

# %%
solution = solNo.as_pandas_dataframe()
# solution.to_csv("Gradients/coactivation_data_60mm_LH.csv")

# %%
coactivation_mat = solution.pivot(index='fr', columns='r', values='logodds')
sns.heatmap(coactivation_mat.fillna(0))
coactivation_mat.shape

# %% [markdown]
# ## Estimate Gradients from the Co-activation Matrix

# %%

labels_idx = np.where(region_voxels['region'].values[:, None] == coactivation_mat.index.values)

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

S = eta2(coactivation_mat.fillna(0).values)
gm = GradientMaps(n_components=20, approach="dm")
gm.fit(S, sparsity=0.0)

# map gradient to brain regions

grad2regions = map_to_labels(-gm.gradients_[:, 0], region_voxels.iloc[labels_idx[0]]['region'].values, 
                     mask=region_voxels.iloc[labels_idx[0]]['region'].values != 0, fill=np.nan)

#Make bins 
bins =  pd.qcut(-gm.gradients_[:, 0], 5, labels=np.arange(0, 5))
bins =  bins.astype('float') + 1

#Map bins and groups to brain regions
bins2regions = map_to_labels(bins, region_voxels.iloc[labels_idx[0]]['region'].values, 
                     mask=region_voxels.iloc[labels_idx[0]]['region'].values != 0,fill=np.nan)

# %%
texture_margulies.shape

# %%
sns.heatmap(S, cmap='jet', vmin=0, vmax=1)

# %%
# Save results
Hemisphere= "RH"
np.save('principal_grad2regions_%s.npy'%Hemisphere, grad2regions)
np.save('principal_bins_%s.npy'%Hemisphere, bins)
np.save('principal_bins2regions_%s.npy'%Hemisphere, bins2regions)

# %%
pd.DataFrame(100*gm.lambdas_/sum(gm.lambdas_).T, index=np.arange(1,21))

# %%
#Plot spectral plot
fig=plt.figure(dpi=300)
plt.plot(100*gm.lambdas_/sum(gm.lambdas_), figure=fig, color='black', marker='o', linewidth=2)
plt.xticks(ticks=np.arange(0,20), labels=np.arange(1,21), fontsize=10, weight="bold")
plt.yticks(fontsize=12, weight="bold")
plt.xlabel('Component', weight="bold")
plt.ylabel('Explained Variance (%)',weight="bold")
plt.show()
# fig.savefig('Gradients/RH/spectral_plot_RH.jpg')

# %% [markdown]
# ## Plot Gradients

# %%
labels_idx[0].shape

# %%
lpfc_regions=region_voxels.iloc[labels_idx[0]]
lpfc_regions.insert(4, "Gradient", grad2regions, True) 
lpfc_regions.insert(0,"Bins", bins2regions, True)
ijk_positions = np.round(
    nib.affines.apply_affine(
        np.linalg.inv(mni_mask.affine),
        lpfc_regions[["x", "y", "z"]].values.astype(int),
    )
).astype(int)

lpfc_regions.insert(loc=1, column="i", value=ijk_positions[:, 0])
lpfc_regions.insert(loc=2, column="j", value=ijk_positions[:, 1])
lpfc_regions.insert(loc=3, column="k", value=ijk_positions[:, 2])

farray=np.zeros_like(mni_mask.get_fdata())
for a,b,c,d in zip(lpfc_regions.i, lpfc_regions.j, lpfc_regions.k, lpfc_regions.Gradient):
    farray[a,b,c]=d

mni_1mm = image.resample_img(    
    nib.load(datasets.fetch_icbm152_2009()["gm"]), np.eye(3) * 3
)

# bn=4
# farray[np.where(farray != bn)] = 0
# farray[np.where(farray == bn)] = 100

grad_img=image.resample_to_img(source_img=nib.Nifti1Image(dataobj=farray, affine=mni_mask.affine), 
                                       target_img=mni_1mm, interpolation = "nearest")

figure=plt.figure(figsize=(10,10))
fsaverage = datasets.fetch_surf_fsaverage()
hemi= "right"
texture = surface.vol_to_surf(grad_img, fsaverage['pial_' + hemi],radius=0, n_samples=1,
                              interpolation="nearest")
plotting.plot_surf_stat_map(fsaverage['infl_' + hemi], stat_map=texture, alpha=1, bg_map=fsaverage['sulc_left'],
                       hemi=hemi, view='lateral',figure=figure, cmap="Blues", bg_on_data=False,  colorbar=False,
                       darkness=0.6, threshold=0.0001)
plt.show()

# figure.savefig("Coactivation/RH/bin%d.jpg"%bn)

# %%
lpfc_regions

# %%
figure=plt.figure(figsize=(5,5), dpi=300)
fsaverage = datasets.fetch_surf_fsaverage()
hemi="right"
view = "dorsal"
lpfc_regions=region_voxels.iloc[labels_idx[0]]
lpfc_regions.insert(4, "Gradient", grad2regions, True) 
lpfc_regions.insert(0,"Bins", bins2regions, True)
ijk_positions = np.round(
    nib.affines.apply_affine(
        np.linalg.inv(mni_mask.affine),
        lpfc_regions[["x", "y", "z"]].values.astype(int),
    )
).astype(int)

lpfc_regions.insert(loc=1, column="i", value=ijk_positions[:, 0])
lpfc_regions.insert(loc=2, column="j", value=ijk_positions[:, 1])
lpfc_regions.insert(loc=3, column="k", value=ijk_positions[:, 2])

farray=np.zeros_like(mni_mask.get_fdata())
for a,b,c,d in zip(lpfc_regions.i, lpfc_regions.j, lpfc_regions.k, lpfc_regions.Bins):
    farray[a,b,c]=d

mni_1mm = image.resample_img(    
    nib.load(datasets.fetch_icbm152_2009()["gm"]), np.eye(3) * 3
)


grad_img=image.resample_to_img(source_img=nib.Nifti1Image(dataobj=farray, affine=mni_mask.affine), 
                                       target_img=mni_1mm, interpolation = "nearest")

texture = surface.vol_to_surf(grad_img, fsaverage['pial_' + hemi], radius=0, n_samples=1,
                              interpolation="nearest")

plotting.plot_surf_stat_map(fsaverage['infl_' + hemi], stat_map=texture,
                            hemi=hemi, view=view, colorbar=False, cmap = "gnuplot",
                            bg_map=fsaverage['sulc_left'], bg_on_data=False,
                            threshold=0.0001, 
                            figure=figure, darkness=0.6)

figure.savefig('Gradients/RH/Principal_Gradient_60mm_%s_%s.jpg'%(hemi, view))

# %%
Hemisphere= "RH"
bins = np.load('principal_bins2regions_%s.npy'%Hemisphere).astype('int')
bins_labels=np.load('principal_bins_%s.npy'%Hemisphere).astype('int')
lpfc_voxel = lpfc_region.as_pandas_dataframe()
lpfc_voxel.insert(0,"Bin", bins, True)
LpfcRegionVoxel = nl.add_tuple_set(lpfc_voxel, name='LpfcRegionVoxel')
BinRegion=nl.add_tuple_set(lpfc_voxel[["Bin", "r"]].drop_duplicates(), name='BinRegion')
Bin = nl.add_tuple_set(lpfc_voxel[["Bin"]].drop_duplicates(), name='Bin')

# %%
with nl.scope as e:
    
    e.VoxelReported[e.x, e.y, e.z, e.s] = (
    e.PeakReported(e.x2, e.y2, e.z2, e.s) 
    & e.GreyMatterVoxel(e.x, e.y, e.z) 
    & (e.d == e.EUCLIDEAN(e.x, e.y, e.z, e.x2, e.y2, e.z2)) 
    & (e.d < 10)
    )
   
    e.BinReported[e.b, e.s] = (
      e.PeakReported(e.x2, e.y2, e.z2, e.s) 
    & LpfcRegionVoxel(e.b, e.r, e.x, e.y, e.z)
    & (e.d == e.EUCLIDEAN(e.x, e.y, e.z, e.x2, e.y2, e.z2)) 
    & (e.d < 3)
    )
    
    e.BinNotReported[e.b, e.s] = (
        Study(e.s)
        & ~e.BinReported(e.b, e.s)
        & Bin(e.b)
    )

    e.BrainRegionReported[e.r, e.x, e.y, e.z, e.s, e.w] = (
        e.BrainVoxel(e.r, e.x, e.y, e.z, e.w)
     &  e.VoxelReported(e.x, e.y, e.z, e.s)
    )
    
    (e.ProbBrainRegionReported @e.max(e.w))[e.r, e.s]=(
        e.BrainRegionReported(e.r, e.x, e.y, e.z, e.s, e.w)
    )

    e.ProbActivationGivenBinActivation[e.r, e.b, e.PROB(e.r, e.b)] = (
        e.ProbBrainRegionReported(e.r, e.s)
    ) // (e.BinReported[e.b, e.s] & SelectedStudy(e.s))
    
    e.ProbActivationGivenNotBinActivation[e.r, e.b, e.PROB(e.r, e.b)] = (
        e.ProbBrainRegionReported(e.r, e.s) 
    ) // (e.BinNotReported[e.b, e.s] & SelectedStudy(e.s)
    )
       
    e.Query[e.r, e.b, e.logodds]= (
     e.ProbActivationGivenBinActivation(e.r, e.b, e.p)
    & e.ProbActivationGivenNotBinActivation(e.r, e.b, e.p0)  
    & (e.logodds == e.log_odds(e.p, e.p0))  
    )    
    
    sol = nl.query(
        (e.region, e.bin, e.logodds), 
      e.Query[e.region, e.bin, e.logodds]
    )  

# %%
from data_utils import mm2vox

ijk = mm2vox(region_voxels[["x", "y", "z"]].values, affine=mni_mask.affine)

# %%
region_voxels[["i", "j", "k"]] = ijk


# %%
def plot_BF_coactivations(
    d: pd.DataFrame,
    mni_mask: nib.Nifti1Image,
    region_voxels: pd.DataFrame,
):
    mni_mask = image.resample_img(
        nib.load(datasets.fetch_icbm152_2009()["gm"]),
        np.eye(3) * 3,
    )
    for b, dg in d.groupby("bin"):
        img_data = np.zeros(mni_mask.shape, dtype=float)
        for r, dgg in dg.groupby("region"):
            if dgg.iloc[0]["logodds"] > np.log10(3):
                idxs = region_voxels.loc[region_voxels.region == r][
                        ["i", "j", "k"]
                    ]
                img_data[tuple(idxs.values.T)] = dgg.iloc[0]["logodds"]
        img = image.threshold_img(nib.Nifti1Image(img_data, mni_mask.affine), threshold="0%")
        figure = plt.figure(figsize=(5,5))
        fsaverage = datasets.fetch_surf_fsaverage()
        hemi="right"
        view = "lateral"
        texture = surface.vol_to_surf(img,fsaverage['pial_' + hemi],interpolation="nearest", kind = 'ball')
        plotting.plot_surf_roi(fsaverage['infl_' + hemi], roi_map=texture,
                                    hemi=hemi, view=view, colorbar=True, vmin=0.3,
                                    bg_map=fsaverage['sulc_' + hemi], cmap="autumn", 
                                    figure=figure, darkness=0.5)
        plt.show()
        figure.savefig('Coactivation/RH/%d_%s_%s.jpg'%(b, hemi, view))


# %%
plot_BF_coactivations(sol.as_pandas_dataframe(), mni_mask, region_voxels)

# %% [markdown]
# # Plot Position of Bins

# %%
df=merge_difumo_network_label(d=sol.as_pandas_dataframe() , difumo_meta=difumo_meta)
df = df[df.network_yeo17 != "No network found"]
nl.add_tuple_set(df[["region", "bin", "network_yeo17"]], name="NetworkName")
with nl.scope as e:
    
    # Compute center of mass (COM)
    e.RegionCenter_x[e.r, e.weighted_center(e.x, e.w)] = e.RegionVoxel(e.r, e.x, e.y, e.z, e.w)
    e.RegionCenter_y[e.r, e.weighted_center(e.y, e.w)] = e.RegionVoxel(e.r, e.x, e.y, e.z, e.w)
    e.RegionCenter_z[e.r, e.weighted_center(e.z, e.w)] = e.RegionVoxel(e.r, e.x, e.y, e.z, e.w)
    
    e.RegionCenter[e.r, e.x, e.y, e.z] = (
          e.RegionCenter_x[e.r, e.x]
        & e.RegionCenter_y[e.r, e.y]
        & e.RegionCenter_z[e.r, e.z]
    )
    
    e.BinCenters[e.b, e.r, e.cy, e.Network] = (
      LpfcRegionVoxel(e.b, e.r, e.x, e.y, e.z, e.w)
    & e.RegionCenter_y(e.r, e.cy)
    & e.RegionCenter_x(e.r, e.cx)
    & e.RegionCenter_z(e.r, e.cz)
    & e.NetworkName(e.r, e.n, e.Network)
    )
    
    solpos = nl.query((e.b, e.r, e.cy, e.Network), e.BinCenters[e.b, e.r, e.cy, e.Network])

# %%
fig = plt.figure()

arr = np.linspace(0, 50, 100).reshape((10, 10))

cmap = plt.get_cmap('gnuplot')
new_cmap = truncate_colormap(cmap, 0.49,1)

plt.cm.register_cmap("newcmap2", new_cmap)
norder=np.array(['SomMotA', 'SomMotB','VisCent', 'VisPeri',  'DorsAttnA', 'SalVentAttnA', 'DorsAttnB', 
                 'TempPar','SalVentAttnB','ContC', 'ContA', 'DefaultC', 'LimbicB', 'LimbicA', 'ContB', 
                 'DefaultA', 'DefaultB', 
                ])
sns.boxplot(data = solpos.as_pandas_dataframe(), x='b', y='cy', palette="newcmap2", showmeans=True,
                       meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                       "markersize":"10"})

ax=sns.stripplot(data = solpos.as_pandas_dataframe(), x='b', y='cy', 
                 hue="Network", hue_order=norder, jitter=True, s = 10)

plt.xlabel("Quintile Along the Gradient", fontsize=14)
plt.ylabel("Posterior-Anterior Position", fontsize=14)
plt.xticks(np.arange(0,5, dtype="int"), fontsize=12, weight="bold")
plt.yticks(fontsize=12, weight="bold")
plt.setp(ax.get_legend().get_texts(), fontsize='16') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='18') # for legend title
plt.legend([],[], frameon=False)
plt.title("Position of Quintile Bins along the Posterior-Anterior (Y) Axis")
fig.savefig('Gradients/LH/Yposition_LH.jpg')

# %%
cols17 = ((255,255,255),
        (70 ,130, 180 ),
        (42, 204, 164  ),
        (255,   0,   0 ),
        (120,  18, 134 ),
        (74 ,155 , 60  ),
        (196 , 58, 250 ),
        (0 ,  118,  14 ),
        (12  ,48, 255  ),
        (255 ,152, 213 ),
        (119 ,140 ,176 ),
        (230 ,148,  34 ),
        (0 ,  0,   130 ),
        (220 ,248, 164 ),
        (122, 135 , 50 ),
        (135,  50 , 74 ),
        (255, 255,   0 ),
        (205 , 62 , 78 )
         )

cols = np.asarray(cols17, dtype=np.float64)/255.
yeoCols = ListedColormap(cols,name='colormapYeo')
colorlist=[]
for i in range(18):
    colorlist.append('#%02x%02x%02x' % cols17[i])
sns.axes_style("white")
sns.set_palette(cols[1:])


did = df
did=did[did.logodds > np.log10(3.0)]

norder=np.array(['SomMotA', 'SomMotB','VisCent', 'VisPeri',  'DorsAttnA', 'SalVentAttnA', 'DorsAttnB', 
                 'TempPar','SalVentAttnB','ContC', 'ContA', 'DefaultC', 'LimbicB', 'LimbicA', 'ContB', 
                 'DefaultA', 'DefaultB', 
                ])
grid = sns.FacetGrid(data=did, col="bin", hue='network_yeo17', hue_order=norder, 
                     col_wrap=5, height=8, aspect=1, sharex=True, sharey=True)
grid.map(sns.countplot,"network_yeo17", order=norder)
grid.set(xticks=[])
grid.set(xlabel="Yeo 17-Networks")
grid.set(ylabel="Number of Regions")
# grid.add_legend()
# leg = grid._legend
# leg.set_title("Networks")

for ax in grid.axes.flat:
    # This only works for the left ylabels
    ax.set_xlabel(ax.get_xlabel(), size=50)
    ax.set_ylabel(ax.get_ylabel(), size=50)
    ax.set_yticklabels(range(0, 41, 10), size = 45)

grid.savefig("Coactivation/RH/facet_plot_histograms.png")

# %%
cols17

# %%
de = did.groupby(["bin","network_yeo17"])["region"].count().reset_index()

norder=np.array(['VisCent', 'VisPeri', 'SomMotA', 'SomMotB', 'DorsAttnA', 'DorsAttnB', 
                 'SalVentAttnA', 'SalVentAttnB', 'LimbicA', 'LimbicB', 'ContC', 'ContA', 'ContB', 
                 'TempPar', 'DefaultC', 'DefaultA', 'DefaultB', 
                ])



cols17 = ((255,255,255),
        (120,  18, 134 ),
        (255,   0,   0 ),
        (70 ,130, 180  ),
        (42, 204, 164  ),
        (74 ,155 , 60  ),
        (0 ,118,  14  ),
        (196 , 58, 250 ),
        (255 ,152, 213 ),
        (220 ,248, 164 ),
        (122, 135 , 50 ),
        (119 ,140 ,176 ),
        (230 ,148,  34 ),
        (135,  50 , 74 ),
        (12  ,48, 255  ),
        (0 ,  0, 130  ),
        (255, 255,   0 ),
        (205 , 62 , 78 ))

cols = np.asarray(cols17, dtype=np.float64)/255.
yeoCols = ListedColormap(cols,name='colormapYeo')
colorlist=[]
for i in range(18):
    colorlist.append('#%02x%02x%02x' % cols17[i])

all_maps= []
ds =  difumo_meta.groupby("Yeo_networks17")["Difumo_names"].count().reset_index()
ds  = ds[ds.Yeo_networks17 != "No network found"]
all_alpha = ds["Difumo_names"].values
for b, dg in de.groupby("bin"):
    cs_alpha = []
    for network, dgg in dg.groupby("network_yeo17"):
        ix = np.where(ds["Yeo_networks17"].values==network)[0][0]
        cs_alpha.append(dgg["region"].values[0]/dg["region"].values.max())
    max_alpha = np.max(cs_alpha)
    cmaps = colorlist[1:].copy()
    for i, (n, c) in enumerate(zip(norder, colorlist[1:])):
            if n in dg["network_yeo17"].values:
                ix = np.where(dg["network_yeo17"].values==n)[0][0]
                cc = alpha_cmap(c, name='', alpha_min=cs_alpha[ix], alpha_max=cs_alpha[ix])
                cmaps[i] = cc(0)
            else:
                cc = alpha_cmap(c, name='', alpha_min=0, alpha_max=0)
                cmaps[i] = cc(0)
#     cc = alpha_cmap(colorlist[0], name='', alpha_min=1, alpha_max=1)
#     cmaps.insert(0,cc(0))
    colors = np.vstack(cmaps)
    mymap = _colors.LinearSegmentedColormap.from_list('bin_colormap', colors)
    all_maps.append(mymap)

# %%
from nilearn import datasets

atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011()
atlas_yeo = atlas_yeo_2011.thick_17
fig= plt.figure(dpi=300)
# Let's now plot it
from nilearn import plotting
bn=5
plotting.plot_roi(atlas_yeo, title=None, display_mode='xz', alpha= 1, figure= fig,cut_coords=(50,50),
                  colorbar=False, cmap=all_maps[bn-1], draw_cross=False, annotate=False)

fig.savefig('Results/LH/coact_noseg/netoverlap_bin%d_lateral'%bn)

# %%
thousand_splits = pd.read_csv("Gradients/results_1k_splits.csv")

# %%

# %%
thousand_splits

# %%
np.unique(grads_join.index)

# %%
thousand_splits = pd.read_csv("Gradients/results_1k_splits.csv")
thousand_splits = thousand_splits.loc[:, ~thousand_splits.columns.str.contains('^Unnamed')]
gdata=[]
sdata =[]
for group_name, df_group in thousand_splits.groupby("row_id"):
    coactivation_mat = df_group.pivot(index='fr', columns='r', values='logodds')
    S = eta2(coactivation_mat.fillna(0).values)
    gm = GradientMaps(n_components=20, approach="dm")
    gm.fit(S, sparsity=0.0)
    grads = pd.DataFrame(gm.gradients_[:,0], index=coactivation_mat.index, columns=["Gradient"])
    grads_join = grads.join(region_voxels.set_index('region'))
    if grads_join.loc["Ventrolateral prefrontal cortex RH"].Gradient.values[0] < 0:
        grads = pd.DataFrame(-gm.gradients_[:,0], index=coactivation_mat.index, columns=["Gradient"])
#         grads_join = grads.join(region_voxels.set_index('region'))
    #Make bins 
    bins =  pd.DataFrame(pd.qcut(grads.Gradient, 5, labels=np.arange(1, 6)))
    bins=bins.rename(columns={"Gradient": "bins_%d"%group_name})
    gdata.append(bins["bins_%d"%group_name])
    sdata.append(pd.DataFrame(100*gm.lambdas_/sum(gm.lambdas_).T, index=np.arange(1,21), 
                              columns=["split_%d"%group_name]))
    

grads_data=pd.concat(gdata, axis=1)
spect_data = pd.concat(sdata, axis=1)

# %%
spect_data= pd.read_csv("Gradients/spect_data.csv")


# %%
spect_data=spect_data.reset_index().rename(columns={"index" : "id"})

# %%
spect_long=spect_data.melt(id_vars="id")

# %%

# %%
import seaborn as sns
plt.figure(figsize=(20,15))
sns.set_theme(style="white")
boxprops = {'edgecolor': 'k', 'linewidth': 2, 'facecolor': 'w'}
lineprops = {'color': 'k', 'linewidth': 2}
boxplot_kwargs = dict({'boxprops': boxprops, 'medianprops': lineprops,
                       'whiskerprops': lineprops, 'capprops': lineprops,
                       'width': 0.75})
stripplot_kwargs = dict({'linewidth': 0.4, 'size': 6, 'alpha': 0.5},
                      )

ax=sns.stripplot(x='id', y='value',data=spect_long,
    split=True, jitter=0.3, **stripplot_kwargs)

ax=sns.boxplot(x="id", y="value", data=spect_long, color="black")
ax=sns.lineplot(x='id',
                y='value',
                data=spect_long,
                ax=ax,
                color= "black",
                linewidth=4)
plt.xticks(np.arange(0,20),np.arange(1,21),fontsize=35)
plt.xlabel("Component", fontsize=40)
plt.yticks(np.arange(1, 50, 5.0), fontsize=35)
plt.ylabel("Variance Explained (%)", fontsize=40)
plt.ylim(0,50)
plt.savefig("Gradients/LH/Spectral_Plot_5ksplits_LH.jpg")

# %%
grads_data= pd.read_csv("Gradients/grads_data_RH.csv")

# %%
import pandas as pd
import numpy as np
import scipy.stats as st
summary=grads_data.agg(['median', 'count', 'std'], axis=1)
ci95_hi = []
ci95_lo = []
typ = "ci95_lo"
for i in summary.index:
    m, c, s = summary.loc[i]
    ci95_hi.append(st.t.interval(alpha=0.95, df=c,
              loc=m,
              scale=s)[1])
    ci95_lo.append(st.t.interval(alpha=0.95, df=c,
              loc=m,
              scale=s)[0])


summary['ci95_hi'] = ci95_hi
summary['ci95_lo'] = ci95_lo

grad2regions = map_to_labels(summary[typ].values, region_voxels.iloc[labels_idx[0]]['region'].values, 
                     mask=region_voxels.iloc[labels_idx[0]]['region'].values != 0, fill=np.nan)

#Make bins 
bins =  pd.qcut(summary[typ].values, 5, labels=np.arange(0, 5))
bins =  bins.astype('float') + 1

#Map bins and groups to brain regions
bins2regions = map_to_labels(bins, region_voxels.iloc[labels_idx[0]]['region'].values, 
                     mask=region_voxels.iloc[labels_idx[0]]['region'].values != 0,fill=np.nan)

# %%
bins_data = []
for i in range(5000):
    #Make bins 
    bins =  pd.qcut(-grads_data.iloc[:, i].values, 5, labels=np.arange(0, 5))
    bins =  bins.astype('float') + 1

    #Map bins and groups to brain regions
    bins2regions = map_to_labels(bins, region_voxels.iloc[labels_idx[0]]['region'].values, 
                         mask=region_voxels.iloc[labels_idx[0]]['region'].values != 0,fill=np.nan)
    
    bins_data.append(bins2regions)

# %%
bins_df = pd.DataFrame(np.vstack(bins_data).T, index=region_voxels.iloc[labels_idx[0]]['region'].values)

# %%
bins_df=bins_df.reset_index().rename(columns={"index":"regions"})

# %%
grouped = bins_df.groupby('regions')
num_counts=[]
ranks=[]
for name, group in grouped:
    cnt = np.zeros((5,1))
    for j in np.arange(1,6):
        cnt[j-1]=(group.iloc[0, 1:].values == j).sum()
    df1 = pd.DataFrame(cnt.T, index=[name], columns=np.arange(1,6)) 
    ranks.append(np.average(np.array(range(0,len(df1.iloc[:, 0:].values[0]))) + 1, weights=df1.iloc[:, 0:].values[0]))
    num_counts.append(df1)

Order = np.argsort(ranks)
dn = pd.concat(num_counts)
dn=dn.reindex(dn.index[Order])

# %%
cmap = plt.get_cmap('gnuplot')
new_cmap = truncate_colormap(cmap, 0.5,1)
plt.cm.register_cmap("newcmap2", new_cmap)
dn.plot.bar(rot=90, stacked=True, figsize=(20,13.5), cmap="newcmap2")
plt.legend(fontsize=16, title='Quintile Bins', title_fontsize=18, loc='center', bbox_to_anchor=(0.85, 1),
          ncol=3, fancybox=True, shadow=True)
plt.ylim(0, 5500)
plt.xticks(fontsize=20)
plt.yticks(fontsize=25)
plt.xlabel("DiFuMo-1024 LPFC Regions", fontsize=25)
plt.ylabel("Counts", fontsize=30)
plt.tight_layout()
plt.savefig("Gradients/gradients_change_RH.jpg", dpi=300)

# %%
Order

# %%
figure=plt.figure(figsize=(5,5), dpi=300)
fsaverage = datasets.fetch_surf_fsaverage()
hemi="left"
view = "dorsal"
lpfc_regions=region_voxels.iloc[labels_idx[0]]
lpfc_regions.insert(4, "Gradient", grad2regions, True) 
lpfc_regions.insert(0,"Bins", bins2regions, True)
ijk_positions = np.round(
    nib.affines.apply_affine(
        np.linalg.inv(mni_mask.affine),
        lpfc_regions[["x", "y", "z"]].values.astype(int),
    )
).astype(int)

lpfc_regions.insert(loc=1, column="i", value=ijk_positions[:, 0])
lpfc_regions.insert(loc=2, column="j", value=ijk_positions[:, 1])
lpfc_regions.insert(loc=3, column="k", value=ijk_positions[:, 2])

farray=np.zeros_like(mni_mask.get_fdata())
for a,b,c,d in zip(lpfc_regions.i, lpfc_regions.j, lpfc_regions.k, lpfc_regions.Bins):
    farray[a,b,c]=d

mni_1mm = image.resample_img(    
    nib.load(datasets.fetch_icbm152_2009()["gm"]), np.eye(3) * 3
)


grad_img=image.resample_to_img(source_img=nib.Nifti1Image(dataobj=farray, affine=mni_mask.affine), 
                                       target_img=mni_1mm, interpolation = "nearest")

texture = surface.vol_to_surf(grad_img, fsaverage['pial_' + hemi], kind="auto", n_samples=1,
                              interpolation="nearest", radius=0)

plotting.plot_surf_stat_map(fsaverage['infl_' + hemi], stat_map=texture,
                            hemi=hemi, view=view, colorbar=False, cmap = "gnuplot",
                            bg_map=fsaverage['sulc_left'], bg_on_data=False,
                            threshold=0.0001, 
                            figure=figure, darkness=0.6)

figure.savefig('Gradients/LH/Principal_Gradient_%s_%s_%s.jpg'%(hemi, view, typ))

# %%
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

arr = np.linspace(0, 50, 100).reshape((10, 10))
fig, ax = plt.subplots(ncols=2)

cmap = plt.get_cmap('gnuplot')
new_cmap = truncate_colormap(cmap, 0.5, 1)
ax[0].imshow(arr, interpolation='nearest', cmap=cmap)
ax[1].imshow(arr, interpolation='nearest', cmap=new_cmap)
plt.show()

# %%
texture[np.argwhere(texture)].shape

# %%
from nilearn.image import iter_img
hemi = "right"                                    
margulies_1 = image.resample_to_img(source_img=nib.load("volume.cort.2.nii.gz"), target_img=mni_mask, 
                                    interpolation="nearest")

texture_margulies = surface.vol_to_surf(margulies_1, fsaverage['pial_' + hemi], 
                              interpolation="nearest")

df = pd.DataFrame(np.hstack([texture[np.argwhere(texture)], texture_margulies[np.argwhere(texture)]]),
                  columns=["Meta-Analytic Gradient", "Resting-State Gradient 1"])

f, axx = plt.subplots(figsize=(15,15))
sns.regplot(data=df, x="Meta-Analytic Gradient", y="Resting-State Gradient 1", ci=95, ax=axx,
          marker="o", color="Black", scatter_kws={'s':40})
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel("Meta-Analytic LPFC Gradient",fontsize=30)
plt.ylabel("Third Resting-State Gradient",fontsize=30)
plt.text(-0.336, 2.735, "r = -0.26 ***", horizontalalignment='left', size=30, color='black', weight='bold')
plt.tight_layout()
plt.savefig("Gradients/LH/spatial_correlation_gradient3_RH.jpg")

# %%
np.corrcoef(texture[np.argwhere(texture)].T, texture_margulies[np.argwhere(texture)].T)

# %%
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

X = scaler.fit_transform(texture[np.argwhere(texture)])
y = scaler.fit_transform(texture_margulies[np.argwhere(texture)])

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


# %%
figure=plt.figure(figsize=(10,10))
fsaverage = datasets.fetch_surf_fsaverage()
big_fsaverage = datasets.fetch_surf_fsaverage()
hemi= "left"
texture = surface.vol_to_surf(gh, big_fsaverage['pial_' + hemi],radius=0, n_samples=1,
                              interpolation="nearest")
plotting.plot_surf_stat_map(big_fsaverage['infl_' + hemi], stat_map=texture, alpha=1, 
                            bg_map=fsaverage['sulc_left'],
                       hemi=hemi, view='lateral',figure=figure, cmap= "gnuplot", bg_on_data=False,  colorbar=True,
                       darkness=0.6, threshold=0.0001)
plt.show()

# %%
plt.figure(figsize=(15,15))
plotting.plot_img_on_surf(margulies_1)

# %%
lpfc_regions

# %%
np.vstack([texture.T[np.argwhere(texture.T)], texture_margulies.T[np.argwhere(texture.T)]])

# %%
marg_voxels = pd.DataFrame(np.array(np.where(gh.get_fdata() != 0)).T, columns=["i", "j", "k"])
xyz_coordinates = nib.affines.apply_affine(mni_mask.affine, marg_voxels[["i", "j", "k"]].values)
marg_voxels.insert(loc=3, column="x", value=xyz_coordinates[:, 0])
marg_voxels.insert(loc=4, column="y", value=xyz_coordinates[:, 1])
marg_voxels.insert(loc=5, column="z", value=xyz_coordinates[:, 2])

# %%
margulies_1 = image.smooth_img(nib.load("volume.cort.2.nii.gz"), fwhm=3)
maps_img = datasets.fetch_atlas_difumo(dimension=1024, resolution_mm=3).maps

maps_masker = NiftiMapsMasker(maps_img=maps_img, verbose=1)

signals = maps_masker.fit_transform(margulies_1)

compressed_maps = maps_masker.inverse_transform(signals)

# %%
from nilearn.input_data import NiftiMapsMasker

hemi="right"
view = "anterior"

texture_margulies = surface.vol_to_surf(compressed_maps, fsaverage['pial_' + hemi], kind="ball", 
                                       mask_img=grad_img)

texture_margulies = pd.DataFrame(texture_margulies, columns=['Gradient'])
bins_mg =  pd.qcut(texture_margulies["Gradient"].rank(method='first'), 5, labels=np.arange(0, 5))
bins_mg =  bins_mg.astype('float') + 1

plotting.plot_surf_stat_map(fsaverage['infl_' + hemi], stat_map=bins_mg.values, 
                            bg_map=fsaverage['sulc_' + hemi],
                            hemi=hemi, view=view,figure=figure, cmap="gnuplot", colorbar=False)

plotting.show()

figure.savefig('Third_Intrinsic_%s_%s.jpg'%(view,hemi))

# %%
0.6*14371

# %%
