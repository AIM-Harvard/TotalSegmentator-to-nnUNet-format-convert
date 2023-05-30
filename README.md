# Convert TotalSegmentator dataset into the nnUNet format
[TotalSegmentator](https://github.com/wasserth/TotalSegmentator) does not provide the scripts for preprocessing the dataset for nnUNet training (https://github.com/wasserth/TotalSegmentator/issues/124). This repo provides a way to do that. Supports nnUNet v2 only.

---

### Step 1 - Create the required environment
`conda create -n ts_to_nnunet pandas jupyter simpleitk=2.0.2 -c simpleitk`
SimpleITK v2.0.2 is required as explained [here](https://github.com/wasserth/TotalSegmentator/issues/32) due to an ITK [issue](https://github.com/InsightSoftwareConsortium/ITK/issues/3994).

### Step 2 - Download the TotalSegmentator dataset
Download the TotalSegmentator dataset from [Zenodo](https://doi.org/10.5281/zenodo.6802613).

### Step 3 - Convert the TotalSegmentator into the nnUNet format
Use the [`examples.ipynb`](examples.ipynb) either as a demo on how to convert the data or to actually convert your own. Make sure to use the environment created in the Step 1.

### Step 4 - Preprocess the dataset and specify the nnUNet's cross-validation folds
After preprocessing the dataset using `nnUNet_plan_and_preprocess`, copy the [`splits_final.json`](splits_final.json) into the dataset's preprocessed directory inside of your `nnUNet_preprocessed` dir. The `splits_final.json` follows TotalSegmentator's train/val split and specifies only fold 0.

### Step 5 - Train only on fold 0
Train your nnUNet on the fold 0 only, just like the TotalSegmentator team did. Other folds do not exist.
