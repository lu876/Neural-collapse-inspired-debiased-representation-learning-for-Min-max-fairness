# Neural-collapse-inspired-debiased-representation-learning-for-Min-max-fairness
This repository contains the implementation code for our work presented at the KDD24 conference. Follow the steps below to set up the environment, prepare the data, and reproduce our experiments.

## Environment Setup

**Create and activate a virtual environment using Conda:**

   ```bash
   conda env create -f environment.yml
   conda activate kdd
```

## Data Preparation
Our experiments use various datasets. Follow the instructions below to prepare each one.
### 1. CelebA Dataset
Download the CelebA using the script:

  ```
import os
import wget
import zipfile

data_root = "celeba/datasets"
base_url = "https://graal.ift.ulaval.ca/public/celeba/"
file_list = [
    "img_align_celeba.zip",
    "list_attr_celeba.txt",
    "identity_CelebA.txt",
    "list_bbox_celeba.txt",
    "list_landmarks_align_celeba.txt",
    "list_eval_partition.txt",
]

dataset_folder = f"{data_root}/celeba"
os.makedirs(dataset_folder, exist_ok=True)

for file in file_list:
    url = f"{base_url}/{file}"
    if not os.path.exists(f"{dataset_folder}/{file}"):
        wget.download(url, f"{dataset_folder}/{file}")

with zipfile.ZipFile(f"{dataset_folder}/img_align_celeba.zip", "r") as ziphandler:
    ziphandler.extractall(dataset_folder)

  ```
Place the CelebA dataset in the root directory.

### ISIC Dataset

2. Download the ISIC dataset using the script adapted from [this source](https://github.com/Wuyxin/DISC/blob/master/disc/download_datasets.py):

   ```python
   import os
   import gdown
   import zipfile

   data_root = '..'  # Set your ROOT directory
   os.makedirs(data_root, exist_ok=True)
   output = 'isic.zip'
   url = 'https://drive.google.com/uc?id=1Os34EapIAJM34DrwZMw2rRRJij3HAUDV'

   if not os.path.exists(os.path.join(data_root, 'isic')):
       gdown.download(url, os.path.join(data_root, output), quiet=False)
       with zipfile.ZipFile(os.path.join(data_root, output), 'r') as zip_ref:
           zip_ref.extractall(data_root)
   ```

### Other Datasets

- **Waterbirds**: Follow the instructions at [this link](https://github.com/kohpangwei/group_DRO) to download the Waterbirds dataset.
- **MultiNLI**: Download from [NYU's website](https://cims.nyu.edu/~sbowman/multinli/).
- **Adult**: This dataset is included in our repository.

## Reproducing Experiments

### CelebA Experiment

Navigate to the CelebA folder. If necessary, modify the dataset directory in lines 82 and 83 of the script. If your dataset is placed in the root directory, no modifications are needed. Run the experiment with the following command:

```bash
python CelebA_ours.py
```

### ISIC, Waterbirds, and Adults Datasets

For these datasets, we provide Jupyter notebook files. Open the respective notebook and ensure the dataset location is correctly set in the data directory variables.

Run the cells in the notebook to reproduce the experiments.

### MultiNLI Datasets
1 Modify the dataset location in lines 49 and 50 of the script as necessary.
2 Execute the experiment with the following command:

  ```bash
  python Ours_NLI.py
  ```


