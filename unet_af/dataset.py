# Maps a physics modality to the subdirectory (relative to the Div2K-Deblur
# base root) where its generated HDF5 dataset lives.
DATASET_DIRS = {
    "circular": "Padding=Circular",
    "valid": "Padding=Valid",
    "valid_v2": "Padding=Valid_v2",
    "denoising": "Modality=Denoising",
}
