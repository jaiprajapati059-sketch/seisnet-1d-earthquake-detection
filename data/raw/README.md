# STEAD (Stanford Earthquake Dataset) Data Directory

This folder is designated for authentic STEAD dataset chunks downloaded from the official Stanford Earthquake Dataset repository.

### Dataset Overview
STEAD is a global, high-quality dataset of seismic events and ambient noise recorded by seismic instruments around the world.

- **Primary Repository**: [Stanford Earthquake Dataset (STEAD) on GitHub](https://github.com/smousavi05/STEAD)
- **Data Archive**: [Stanford Geospatial Data Repository](https://earthquake.usgs.gov/earthquakes/) / Zenodo / AWS Open Data

### File Placement
To train and evaluate SeisNet-1D on authentic STEAD data:
1. Download a single chunk (e.g., `chunk1.hdf5` and `chunk1.csv`).
2. Place both files directly into this directory:
   ```
   data/raw/
   ├── chunk1.hdf5
   └── chunk1.csv
   ```
3. Run the pipeline with the `--mode stead` flag:
   ```bash
   python src/train.py --mode stead
   python src/evaluate.py --mode stead
   python src/visualize.py --mode stead
   ```

### Default Demo Mode
If you do not place STEAD chunk files here, `SeisNet-1D` automatically defaults to `--mode demo`. In demo mode, `src/generate_demo_chunk.py` synthesizes a deterministic, physically realistic 1,200-trace sample chunk (`data/sample/chunk_demo.hdf5` and `data/sample/chunk_demo.csv`) containing 800 local earthquakes (with vertical P-onsets, horizontal S-shear packets, and coda decay) and 400 colored noise traces, enabling immediate zero-download execution.
