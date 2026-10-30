"""Notebook-first BiVO4/water proton-transfer and free-energy analysis."""
from .selection import PTSelection, by_z, notebook_bi_water
from .io import Frame, as_box, iter_xyz, xyz_symbols, iter_mdanalysis, read_colvar, align_colvar, check_cv_alignment
from .geometry import PTResult, analyze, frame_paths, all_path_samples, PATH_COLUMNS, choose_path
from .reweight import (opes_logweights, normalized_weights, weight_diagnostics,
                       original_walls, upper_wall, lower_wall, populations,
                       weighted_mean, basin_delta_f, block_populations)
from .fes import FES, compute_fes, read_fes, average_fes
from .mep import String2D, minimax_path
from .coordination import coordination, protonated_site
__version__ = '0.1.2'
from .geometry import analyze_chunks
from .io import write_colvar
