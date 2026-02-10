# Evaluation Metrics Algorithm: Baseline Finder vs. ML Finder

This algorithm is designed to compare the the track reconstruction performance of the **Baseline Finder** and **ML Finder (with and without fitting)**.

## Environment Setup & Usage Instructions

### Prerequisite

Install **Anaconda/Miniconda** first. Download link: [https://www.anaconda.com/download](https://www.anaconda.com/download)

### 1. Create and Activate a Conda Virtual Environment

```bash
# Create a virtual environment named DCMetrics with Python 3.9
conda create -n DCMetrics python=3.9 -y

# Activate the environment
conda activate DCMetrics

```

### 2. Upgrade pip (Recommended)

```bash
# Upgrade pip to the latest version to avoid installation errors
pip install --upgrade pip
```

### 3. Install Dependencies with Specified Versions

```bash
# Install core required packages
pip install numpy==1.26 particle==0.25.4 uproot==5.6.0 matplotlib==3.9.4 pybes3==0.3.3 joblib==1.4.2
conda install -c conda-forge root=6.28.0
```


### 4. Run the Evaluation Algorithm

1. Prepare input data files (format `.rec`), which should include the output results of Baseline Finder and ML Finder (with/without fitting) respectively.

2. Execute the evaluation script:

```bash
# For detailed parameter settings, please refer to the file: ./src/performance/parser.py
python3 main.py \
    --data_type [DATA_TYPE] \
    --particle_type [PARTICLE_TYPE] \
    --particle_q [PARTICLE_CHARGE] \
    --root_dir [ROOT_DATA_DIRECTORY] \
    --trad_dir [TRADITIONAL_METHOD_DIRECTORY] \
    --ml_dir [ML_METHOD_DIRECTORY] \
    --trad_mdc_dir [TRADITIONAL_MDC_DIRECTORY] \
    --ml_mdc_dir [ML_MDC_DIRECTORY] \
    --cache_path [CACHE_FILE_PATH] \
    --output_pdf [OUTPUT_PDF_PATH] \
    --batch_size [BATCH_SIZE]
```



