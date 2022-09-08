import datetime
import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import neurolang
import nibabel
import nilearn.datasets
import nilearn.datasets.utils
import nilearn.image
import nilearn.input_data
import numpy as np
import pandas as pd
import scipy.sparse
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from neurolang.frontend.neurosynth_utils import fetch_neurosynth_peak_data, fetch_feature_data, fetch_study_metadata

DATA_DIR = Path(neurolang.__path__[0]) / "data"


DIFUMO_N_COMPONENTS_TO_DOWNLOAD_ID = {
    128: "wjvd5",
    256: "3vrct",
    512: "9b76y",
    1024: "34792",
}


def tal2mni(coords):
    """Convert coordinates from Talairach space to MNI space.
    .. versionchanged:: 0.0.8
        * [ENH] This function was part of `nimare.transforms` in previous versions (0.0.3-0.0.7)
    Parameters
    ----------
    coords : (X, 3) :obj:`numpy.ndarray`
        Coordinates in Talairach space to convert.
        Each row is a coordinate, with three columns.
    Returns
    -------
    coords : (X, 3) :obj:`numpy.ndarray`
        Coordinates in MNI space.
        Each row is a coordinate, with three columns.
    Notes
    -----
    Python version of BrainMap's tal2icbm_other.m.
    This function converts coordinates from Talairach space to MNI
    space (normalized using templates other than those contained
    in SPM and FSL) using the tal2icbm transform developed and
    validated by Jack Lancaster at the Research Imaging Center in
    San Antonio, Texas.
    http://www3.interscience.wiley.com/cgi-bin/abstract/114104479/ABSTRACT
    """

    coords = coords.transpose()
    # Transformation matrices, different for each software package
    icbm_other = np.array(
        [
            [0.9357, 0.0029, -0.0072, -1.0423],
            [-0.0065, 0.9396, -0.0726, -1.3940],
            [0.0103, 0.0752, 0.8967, 3.6475],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ]
    )

    # Invert the transformation matrix
    icbm_other = np.linalg.inv(icbm_other)

    # Apply the transformation matrix
    coords = np.concatenate((coords, np.ones((1, coords.shape[1]))))
    coords = np.dot(icbm_other, coords)

    # Format the output, transpose if necessary
    out_coords = coords[:3, :]

    out_coords = out_coords.transpose()
    return out_coords.astype("int")


def rescale(data):
    """Rescale the data to be within the range [new_min, new_max]"""
    return (data - data.min()) / (data.max() - data.min()) 


def coords_to_voxels(
    coords: np.array,
    mask: nilearn.input_data.NiftiMasker,
) -> np.array:
    affine = mask.affine
    coords = np.atleast_2d(coords)
    coords = np.hstack([coords, np.ones((len(coords), 1))])
    voxels = np.linalg.pinv(affine).dot(coords.T)[:-1].T
    voxels = voxels[(voxels >= 0).all(axis=1)]
    voxels = voxels[(voxels < mask.shape[:3]).all(axis=1)]
    voxels = np.floor(voxels)
    return voxels.astype("int")


def fetch_neurosynth(
    tfidf_threshold: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = fetch_feature_data(data_dir="/Users/majdabdallah/Desktop/Work/")
    metadata = fetch_study_metadata(data_dir="/Users/majdabdallah/Desktop/Work/")
    features["study_id"] = metadata["id"].tolist()
    features = features.reset_index()
    term_data = pd.melt(
        features,   
        id_vars="study_id",
        value_name="tfidf",
    )
    term_data=term_data.rename(columns={"variable": "term", "value": "tfidf"})
    if tfidf_threshold is not None:
        term_data = term_data.query("tfidf > {}".format(tfidf_threshold))[
            ["term", "study_id"]
        ]
    else:
        term_data = term_data.query("tfidf > 0")[["term", "tfidf", "study_id"]]
        
    
    activations = fetch_neurosynth_peak_data(data_dir="/Users/majdabdallah/Desktop/Work/")
    mni_peaks = activations.loc[activations.space == "MNI"][
        ["x", "y", "z", "id"]
    ].rename(columns={"id": "study_id"})
    non_mni_peaks = activations.loc[activations.space != "MNI"][
        ["x", "y", "z", "id"]
    ].rename(columns={"id": "study_id"})

    projected= tal2mni(non_mni_peaks[["x", "y", "z"]].values)
    projected_df = pd.DataFrame(
        np.hstack([projected, non_mni_peaks[["study_id"]].values]),
        columns=["x", "y", "z", "study_id"],
        dtype=int,
    )
    peak_data = pd.concat([projected_df, mni_peaks]).astype(int)
    study_ids = peak_data[["study_id"]].drop_duplicates()
    return term_data, peak_data, study_ids


def fetch_neuroquery(
    mask: nilearn.input_data.NiftiMasker,
    data_dir: Path = DATA_DIR,
    tfidf_threshold: Optional[float] = None,
    coord_type: str = "xyz",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_url = "https://github.com/neuroquery/neuroquery_data/"
    tfidf_url = base_url + "tree/main/neuroquery_model/corpus_tfidf.npz"
    coordinates_url = base_url + "tree/main/data/data-neuroquery_version-1_coordinates.tsv.gz"
    feature_names_url = base_url + "tree/main/neuroquery_model/vocabulary.csv"
    study_ids_url = base_url + "tree/main/neuroquery_model/corpus_metadata.csv"
    out_dir = data_dir / "neuroquery"
    opts = {'unzip' : True}
    os.makedirs(out_dir, exist_ok=True)
    (
        tfidf_fn,
        coordinates_fn,
        feature_names_fn,
        study_ids_fn,
    ) = nilearn.datasets.utils._fetch_files(
        out_dir,
        [
            ("corpus_tfidf.npz", tfidf_url, dict()),
            ("data-neuroquery_version-1_coordinates.tsv.gz", coordinates_url, dict()),
            ("vocabulary.csv", feature_names_url, dict()),
            ("corpus_metadata.csv", study_ids_url, dict()),
        ],
    )
    tfidf = np.load(tfidf_fn)
    coordinates = pd.read_csv(coordinates_fn, sep='\t')
    assert coord_type in ("xyz", "ijk")
    coord_cols = list(coord_type)
    to_concat = list()
    for pmid, dfg in coordinates.groupby("id"):
        coords = dfg[["x", "y", "z"]]
        if coord_type == "ijk":
            coords = coords_to_voxels(coords, mask=mask)
        df = pd.DataFrame(coords, columns=coord_cols)
        df["id"] = pmid
        to_concat.append(df[coord_cols + ["id"]])
    coordinates = pd.concat(to_concat)
    peak_reported = coordinates.rename(columns={"id": "study_id"})
    feature_names = pd.read_csv(feature_names_fn, header=None)
    study_ids = pd.read_csv(study_ids_fn, header=None)
    study_ids.rename(columns={0: "study_id"}, inplace=True)
    tfidf = pd.DataFrame(tfidf.todense(), columns=feature_names[0])
    tfidf["study_id"] = study_ids.iloc[:, 0]
    if tfidf_threshold is None:
        term_data = pd.melt(
            tfidf,
            var_name="term",
            id_vars="study_id",
            value_name="tfidf",
        ).query("tfidf > 0")[["term", "tfidf", "study_id"]]
    else:
        term_data = pd.melt(
            tfidf,
            var_name="term",
            id_vars="study_id",
            value_name="tfidf",
        ).query(f"tfidf > {tfidf_threshold}")[["term", "study_id"]]
    return term_data, peak_reported, study_ids


def fetch_difumo_meta(
    data_dir: Path = DATA_DIR,
    n_components: int = 256,
) -> pd.DataFrame:
    out_dir = data_dir / "difumo"
    download_id = DIFUMO_N_COMPONENTS_TO_DOWNLOAD_ID[n_components]
    url = f"https://osf.io/{download_id}/download"
    labels_path = os.path.join(
        str(n_components), f"labels_{n_components}_dictionary.csv"
    )
    files = [
        (labels_path, url, {"uncompress": True}),
    ]
    files = nilearn.datasets.utils._fetch_files(out_dir, files, verbose=2)
    labels = pd.read_csv(files[0])
    return labels


def fetch_difumo(
    mask: nilearn.input_data.NiftiMasker,
    component_filter_fun: Callable = lambda _: True,
    data_dir: Path = DATA_DIR,
    coord_type: str = "xyz",
    n_components: int = 256,
    resolution: str = "3mm"
) -> Tuple[pd.DataFrame, nibabel.Nifti1Image]:
    out_dir = data_dir / "difumo"
    download_id = DIFUMO_N_COMPONENTS_TO_DOWNLOAD_ID[n_components]
    url = f"https://osf.io/{download_id}/download"
    csv_file = os.path.join(
        str(n_components), f"labels_{n_components}_dictionary.csv"
    )
    nifti_file = os.path.join(str(n_components), "%s/maps.nii.gz"%resolution)
    files = [
        (csv_file, url, {"uncompress": True}),
        (nifti_file, url, {"uncompress": True}),
    ]
    files = nilearn.datasets.utils._fetch_files(out_dir, files, verbose=2)
    labels = pd.read_csv(files[0])
    img = nilearn.image.load_img(files[1])
    img = nilearn.image.resample_to_img(
        img,
        target_img=mask,
        interpolation="nearest",
    )
    img_data = img.get_fdata()
    to_concat = list()
    for i, label in enumerate(
        labels.loc[labels.apply(component_filter_fun, axis=1)].Difumo_names
    ):  
        imj = img_data[:, :, :, i]
        coordinates = np.where(imj > 0)
        dataimg = imj[coordinates].flatten()
        if coord_type == "xyz":
            coordinates = nibabel.affines.apply_affine(img.affine, np.vstack(coordinates).T.astype("int"))
            region_data = pd.DataFrame(np.vstack(coordinates), columns=list(coord_type))
        else:
            assert coord_type == "ijk"
            region_data = pd.DataFrame(np.vstack(coordinates).T, columns=list(coord_type))
            
        region_data["region"] = label
        region_data["weights"] = dataimg
        to_concat.append(region_data[["region"] + list(coord_type) + ["weights"]])
    region_voxels = pd.concat(to_concat)
    return region_voxels, labels


def fetch_schaefer(
    mask: nilearn.input_data.NiftiMasker,
    component_filter_fun: Callable = lambda _: True,
    data_dir: Path = DATA_DIR,
    coord_type: str = "xyz",
    n_components: int = 1000,
) -> Tuple[pd.DataFrame, nibabel.Nifti1Image]:
    files = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=n_components)
    labels = files["labels"]
    img = nib.load(files.maps)
    img = nilearn.image.resample_img(
        img,
        target_affine=mask.affine,
        interpolation="nearest",
    )
    img_data = img.get_fdata()
    to_concat = list()
    for i, label in enumerate(
        labels
    ):  
        coordinates = np.argwhere(img_data[:, :, :] == i)
        if coord_type == "xyz":
            coordinates = nibabel.affines.apply_affine(img.affine, coordinates)
        else:
            assert coord_type == "ijk"
        region_data = pd.DataFrame(coordinates, columns=list(coord_type))
        region_data["label"] = label
        to_concat.append(region_data[["label"] + list(coord_type)])
    region_voxels = pd.concat(to_concat)
    return region_voxels, labels    


def fetch_yeo(
    mask: nilearn.input_data.NiftiMasker,
    component_filter_fun: Callable = lambda _: True,
    data_dir: Path = DATA_DIR,
    coord_type: str = "xyz",
) -> Tuple[pd.DataFrame, nibabel.Nifti1Image]:
    files = nilearn.datasets.fetch_atlas_yeo_2011()["thick_7"]
    labels = ["Visual", "SomatoSensory", "Dorsal Attention", "Ventral Attention", 
              "Limbic", "Executive Control", "Default Mode"]
    img = nib.load(files)
    img = nilearn.image.resample_img(
        img,
        target_affine=mask.affine,
        interpolation="nearest",
    )
    img_data = img.get_fdata()
    to_concat = list()
    for i, label in enumerate(
        labels
    ):  
        if i>0:
            coordinates = np.argwhere(img_data[:, :, :] == i)
            if coord_type == "xyz":
                coordinates = nibabel.affines.apply_affine(img.affine, coordinates)
            else:
                assert coord_type == "ijk"
            region_data = pd.DataFrame(coordinates, columns=list(coord_type))
            region_data["label"] = label
            to_concat.append(region_data[["label"] + list(coord_type)])
        else: 
            continue
    region_voxels = pd.concat(to_concat)
    return region_voxels, labels 


def save_to_hdf(d: pd.DataFrame, dst_path: Path) -> None:
    print(f"saving to {dst_path}")
    with pd.HDFStore(
        dst_path, mode="w", complib="blosc:lz4", complevel=9
    ) as hdf_store:
        hdf_store["data"] = d


def get_exp_dir(exp_name: str):
    module_dir = Path(__file__).parent
    exp_dir = module_dir / exp_name
    if not exp_dir.is_dir():
        raise FileNotFoundError(
            f"Unknown exp: exp dir {exp_dir} does not exist"
        )
    return exp_dir


def load_results(
    exp_name: str,
    out_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    exp_dir = get_exp_dir(exp_name)
    result_dir = exp_dir / "_results"
    if not result_dir.is_dir():
        raise FileNotFoundError(
            f"Results not available for exp {exp_name}. "
            f"Directory {exp_dir} does not exist"
        )
    cache_paths = list(result_dir.glob(f"{exp_name}-results*.h5"))
    if cache_paths:
        return pd.read_hdf(result_dir / next(iter(cache_paths)), "data")
    if out_path is not None:
        if isinstance(out_path, str):
            out_path = Path(out_path)
    to_concat = list()
    for path in result_dir.glob("*.h5"):
        to_concat.append(pd.read_hdf(result_dir / path, "data"))
    results = pd.concat(to_concat)
    datestr = datetime.date.today().isoformat()
    cache_path = result_dir / f"{exp_name}-results-{datestr}.h5"
    if cache_path.is_file():
        cache_path.unlink()
    save_to_hdf(results, cache_path)
    if out_path is not None:
        save_to_hdf(results, out_path)
    return results


def load_aggregated_results(exp_name: str) -> pd.DataFrame:
    exp_dir = get_exp_dir(exp_name)
    result_dir = exp_dir / "_results"
    if not result_dir.is_dir():
        raise FileNotFoundError(
            f"Results not available for exp {exp_name}. "
            f"Directory {exp_dir} does not exist"
        )
    cache_paths = list(result_dir.glob(f"{exp_name}-aggregated-results*.h5"))
    if not cache_paths:
        raise FileNotFoundError(
            f"Aggregated results not available in {result_dir}"
        )
    print("loading cached aggregated results from")
    print(next(iter(cache_paths)))
    return pd.read_hdf(result_dir / next(iter(cache_paths)), "data")


def save_aggregated_results(exp_name: str, results: pd.DataFrame) -> None:
    exp_dir = get_exp_dir(exp_name)
    result_dir = exp_dir / "_results"
    if not result_dir.is_dir():
        raise FileNotFoundError(f"Directory {exp_dir} does not exist")
    datestr = datetime.date.today().isoformat()
    filename = f"{exp_name}-aggregated-results-{datestr}.h5"
    save_to_hdf(results, result_dir / filename)
    print("saved aggregated results to")
    print(result_dir / filename)


def load_cognitive_terms(filename: str) -> pd.Series:
    if filename is None:
        path = Path(__file__).parent / "cognitive_terms.txt"
    else:
        path = Path(__file__).parent / f"{filename}.txt"
    return pd.read_csv(path, header=None, names=["term"]).drop_duplicates()


# +
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align.metrics import CCMetric

def transform_to_symmetric_template(coords):
    moving_img = image.resample_img(nib.load('mni_icbm152_gm_tal_nlin_asym_09a.nii'), 
                                    np.eye(3)*3, interpolation="nearest")
    
    template_img = image.resample_img(nib.load('mni_icbm152_gm_tal_nlin_sym_09a.nii'), 
                                    np.eye(3)*3, interpolation="nearest")
    
    moving_data = moving_img.get_fdata()
    moving_affine = moving_img.affine
    template_data = template_img.get_fdata()
    template_affine = template_img.affine

    # The mismatch metric
    metric = CCMetric(3)
    # The optimization strategy:
    level_iters = [10, 10, 5]
    # Registration object
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)

    mapping = sdr.optimize(static=template_data, moving=moving_data,
                           static_grid2world=template_affine,
                           moving_grid2world=moving_affine)
    to_concat = list()
    for study_id, row in coords.groupby("study_id"):
        ijk_positions = np.round(
            nib.affines.apply_affine(
                np.linalg.pinv(moving_img.affine),
                row[["x", "y", "z"]].values.astype("int"),
            )
        ).astype(int)
        data_obj= np.zeros_like(moving_img.get_fdata())
        data_obj[tuple(ijk_positions.T)] = 1
        warped_moving = mapping.transform(data_obj, interpolation="nearest")
        ijk_warped = np.where(warped_moving==1)        
        xyz_coordinates = nib.affines.apply_affine(template_img.affine, np.c_[ijk_warped])
        df = pd.DataFrame(xyz_coordinates, columns = ["x", "y", "z"])
        df["study_id"] = study_id
        to_concat.append(df)
    sym_coords = pd.concat(to_concat)
    return sym_coords.reset_index(drop=True)


# +
import random
from sklearn.utils import shuffle
def create_coordinate_dataset(foci):
    """Generate study specific foci.
    .. versionadded:: 0.0.4
    Parameters
    ----------
    foci : :obj:`int` or :obj:`list`
        The number of foci to be generated per study or the
        x,y,z coordinates of the ground truth foci.
    foci_percentage : :obj:`float`
        Percentage of studies where the foci appear.
    fwhm : :obj:`float`
        Full width at half maximum (fwhm) to define the probability
        spread of the foci.
    n_studies : :obj:`int`
        Number of n_studies to generate.
    n_noise_foci : :obj:`int`
        Number of foci considered to be noise in each study.
    rng : :class:`numpy.random.RandomState`
        Random state to reproducibly initialize random numbers.
    space : :obj:`str`
        The template space the coordinates are reported in.
    Returns
    -------
    ground_truth_foci : :obj:`list`
        List of 3-item tuples containing x, y, z coordinates
        of the ground truth foci or an empty list if
        there are no ground_truth_foci.
    foci_dict : :obj:`dict`
        Dictionary with keys representing the study, and
        whose values represent the study specific foci.
    """

    template_img = nilearn.image.resample_img(nib.load("MNI152_T1_2mm_brain_mask.nii.gz"), 
                                              np.eye(3)*3, interpolation="nearest")

    template_data = template_img.get_fdata()
    possible_ijks = np.argwhere(template_data)

    num_foci = foci.groupby('study_id').size().tolist()
    ground_truth_foci_ijks = []
    for n in num_foci:
            foci_idxs = np.random.choice(range(len(possible_ijks)), n, replace=False)
            ground_truth_foci_ijks.append(possible_ijks[foci_idxs])
            
    ground_truth_foci_ijks = np.vstack(ground_truth_foci_ijks)
    
      
    ground_truth_foci_xyz = vox2mm(ground_truth_foci_ijks, template_img.affine)

    
    xyz_coords_df = pd.DataFrame(ground_truth_foci_xyz, columns=["x", "y", "z"])
    
    return xyz_coords_df

def mm2vox(xyz, affine):
    """Convert coordinates to matrix subscripts.
    .. versionchanged:: 0.0.8
        * [ENH] This function was part of `nimare.transforms` in previous versions (0.0.3-0.0.7)
    Parameters
    ----------
    xyz : (X, 3) :obj:`numpy.ndarray`
        Coordinates in image-space.
        One row for each coordinate, with three columns: x, y, and z.
    affine : (4, 4) :obj:`numpy.ndarray`
        Affine matrix from image.
    Returns
    -------
    ijk : (X, 3) :obj:`numpy.ndarray`
        Matrix subscripts for coordinates being transformed.
    Notes
    -----
    From here:
    http://blog.chrisgorgolewski.org/2014/12/how-to-convert-between-voxel-and-mm.html
    """
    ijk = nib.affines.apply_affine(np.linalg.inv(affine), xyz).astype(int)
    return ijk

def vox2mm(ijk, affine):
    """Convert matrix subscripts to coordinates.
    .. versionchanged:: 0.0.8
        * [ENH] This function was part of `nimare.transforms` in previous versions (0.0.3-0.0.7)
    Parameters
    ----------
    ijk : (X, 3) :obj:`numpy.ndarray`
        Matrix subscripts for coordinates being transformed.
        One row for each coordinate, with three columns: i, j, and k.
    affine : (4, 4) :obj:`numpy.ndarray`
        Affine matrix from image.
    Returns
    -------
    xyz : (X, 3) :obj:`numpy.ndarray`
        Coordinates in image-space.
    Notes
    -----
    From here:
    http://blog.chrisgorgolewski.org/2014/12/how-to-convert-between-voxel-and-mm.html
    """
    xyz = nib.affines.apply_affine(affine, ijk)
    return xyz


# -

import numpy as np
from matplotlib import cm 
from matplotlib import colors as _colors
from matplotlib import rcParams 
def alpha_cmap(color, name='', alpha_min=0.2, alpha_max=0.5):
    """ Return a colormap with the given color, and alpha going between two values.
    Parameters
    ----------
    color : (r, g, b), or a string
        A triplet of floats ranging from 0 to 1, or a matplotlib
        color string.
    name : string, optional
        Name of the colormap. Default=''.
    alpha_min : Float, optional
        Minimum value for alpha. Default=0.5.
    alpha_max : Float, optional
        Maximum value for alpha. Default=1.0.
    """
    red, green, blue = _colors.colorConverter.to_rgb(color)
    if name == '' and hasattr(color, 'startswith'):
        name = color
    cmapspec = [(red, green, blue, 1.),
                (red, green, blue, 1.),
               ]
    cmap = _colors.LinearSegmentedColormap.from_list(
        '%s_transparent' % name, cmapspec, rcParams['image.lut'])
    cmap._init()
    cmap._lut[:, -1] = np.linspace(alpha_min, alpha_max, cmap._lut.shape[0])
    cmap._lut[-1, -1] = 0
    return cmap


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def merge_difumo_network_label(
    d: pd.DataFrame,
    difumo_meta: pd.DataFrame,
    dst_col: str = "network_yeo17",
    network_col: str = "Yeo_networks17",
) -> pd.DataFrame:
    if network_col not in difumo_meta.columns:
        raise ValueError(f"Unknown network column name: {network_col}")
    meta = difumo_meta[["Difumo_names", network_col]]
    meta = meta.rename(
        columns={"Difumo_names": "region", network_col: dst_col},
    )
    get_network_fn = dict(meta[["region", dst_col]].values).get
    d[dst_col] = d['region'].apply(get_network_fn)
    return d
