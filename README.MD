#Sparse Local Patch Transformer

PyTorch training code for **SLPT** (Sparse Local Patch Transformer).

## Installation

Install system requirements:
```
sudo apt-get install python3-dev python3-pip python3-tk libglib2.0-0
```

Install python dependencies:
```
pip3 install -r requirements.txt
```

## Run training code on WFLW dataset
1. Download and process WFLW dataset
    * Download WFLW dataset and annotation from [Here](https://wywu.github.io/projects/LAB/WFLW.html).
    * Unzip WFLW dataset and annotations and move files into ```./Data``` directory. Your directory should look like this:
        ```
        SLPT
        └───Data
           │
           └───WFLW
              │
              └───WFLW_annotations
              │   └───list_98pt_rect_attr_train_test
              │   │
              │   └───list_98pt_test
              └───WFLW_images
                  └───0--Parade
                  │
                  └───...
        ```

2. Modify ```./Config/default.py```.
   ```
    _C.DATASET.DATASET = 'WFLW'.
    _C.TRAIN.LR_STEP = [120, 140]
    _C.TRAIN.NUM_EPOCH = 150
   ```
3. ```python ./train.py```.

## Run training code on 300W dataset
1. Download and process 300W dataset
    * Download 300W dataset and annotation from [Here](https://ibug.doc.ic.ac.uk/resources/300-W/).
    * Unzip 300W dataset and annotations and move files into ```./Data``` directory. Your directory should look like this:
        ```
        SLPT
        └───Data
           │
           └───300W
              │
              └───helen
              │   └───trainset
              │   │
              │   └───testset
              └───lfpw
              │   └───trainset
              │   │
              │   └───testset
              └───afw
              │
              └───ibug      
        ```

2. Modify ```./Config/default.py```.
   ```
    _C.DATASET.DATASET = '300W'.
    _C.TRAIN.LR_STEP = [80, 100]
    _C.TRAIN.NUM_EPOCH = 120
   ```
3. ```python ./train.py```.
   


##Citation
If you find this work or code is helpful in your research, please cite:
```
@inproceedings{SLPT,
  title={Sparse Local Patch Transformer for Robust Face Alignment and Landmarks},
  author={Jiahao Xia and Weiwei Qu and Jianguo Zhang and Xi Wang and Min Xu},
  booktitle={CVPR},
  year={2022}
}
```

# License
SLPT is released under the GPL-2.0 license. Please see the [LICENSE](License) file for more information.

# Acknowledgments
   * This repository borrows or partially modifies the models from [HRNet](https://github.com/HRNet/HRNet-Facial-Landmark-Detection)
and [DETR](https://github.com/facebookresearch/detr)
