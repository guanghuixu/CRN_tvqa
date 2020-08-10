

# Cascade Reasoning Network for Text-based Visual Question Answering

Pytorch implementation for the ACM MM 2020 paper: [Cascade Reasoning Network for Text-based Visual Question Answering](https://github.com/guanghuixu/CRN_tvqa)

![](https://github.com/guanghuixu/CRN_tvqa/blob/master/docs/source/models/overview.png)

## Install

Clone this repository, and build it with the following command.
```
# activate your own conda environment
# [Alternative]
# conda env create -f CRN.yaml
# conda activate CRN

git clone https://github.com/guanghuixu/CRN_tvqa.git
cd CRN_tvqa
python setup.py build develop
```
## Data

| Datasets      | Object Features | OCR Features |
|--------------|-------------------------------|:------------------------------|
| TextVQA      | [Open Images](https://dl.fbaipublicfiles.com/pythia/features/open_images.tar.gz) | [TextVQA Rosetta-en OCRs](https://dl.fbaipublicfiles.com/pythia/m4c/data/m4c_textvqa_ocr_en_frcn_features.tar.gz) |
| ST-VQA      | [ST-VQA Objects](https://dl.fbaipublicfiles.com/pythia/m4c/data/m4c_stvqa_obj_frcn_features.tar.gz) | [ST-VQA Rosetta-en OCRs](https://dl.fbaipublicfiles.com/pythia/m4c/data/m4c_stvqa_ocr_en_frcn_features.tar.gz) |
| OCR-VQA      | [OCR-VQA Objects](https://dl.fbaipublicfiles.com/pythia/m4c/data/m4c_ocrvqa_obj_frcn_features.tar.gz) | [OCR-VQA Rosetta-en OCRs](https://dl.fbaipublicfiles.com/pythia/m4c/data/m4c_ocrvqa_ocr_en_frcn_features.tar.gz) |

```
cd ~/CRN_tvqa

# Download dataset annotations
wget [data.tar.xz]  
tar xf data.tar.gz

cd data

# Download detectron weights
wget http://dl.fbaipublicfiles.com/pythia/data/detectron_weights.tar.gz
tar xf detectron_weights.tar.gz

# Now download the features required, feature link is taken from the table below [Provided by M4C]

cd crn_textvqa

wget https://dl.fbaipublicfiles.com/pythia/features/open_images.tar.gz
tar xf open_images.tar.gz

wget https://dl.fbaipublicfiles.com/pythia/m4c/data/m4c_textvqa_ocr_en_frcn_features.tar.gz
tar xf m4c_textvqa_ocr_en_frcn_features.tar.gz

cd ../..

# calculate the edge features for [train, val, test] split
bash scripts/process_dataset.sh crn_textvqa data/crn_textvqa/imdb/imdb_train_ocr_en.npy
```

## Training and Evaluation

The training and evaluation commands can be found in the `./scripts`. The config files can be found in the `./configs`

1) to train the model on the TextVQA training set:

```
# bash scripts/<train.sh> <GPU_ids> <save_dir>
bash scripts/train_textvqa.sh 0,1 textvqa_debug
```

(Note: replace `textvqa` with other datasets and other config files to train with other datasets and configurations.)

2) to evaluate the pretrained model on the TextVQA validation/test set:

```
# bash scripts/<val.sh> <GPU_ids> <save_dir> <checkpoint> <run_type>

bash scripts/val_textvqa.sh 0,1 textvqa_debug save/textvqa_debug/crn_textvqa_crn/best.ckpt val

bash scripts/val_textvqa.sh 0,1 textvqa_debug save/textvqa_debug/crn_textvqa_crn/best.ckpt inference
```
(Note: `--<run_type>` use `inference` instead of `val` to generate the EvalAI prediction files for the TextVQA test set )

## Citation

If you use our code in your research, please cite our paper:

```
@inproceedings{liu2020crn, 
title={Cascade Reasoning Network for Text-based Visual Question Answering},
author={Fen Liu, Guanghui Xu, Qi Wu, Qing Du, Wei Jia and Mingkui Tan}, 
booktitle={Proceedings of the 28th ACM International Conference on Multimedia},  
year={2020}
}
```

## Acknowledgment

The code is greatly inspired by the [MMF](https://mmf.readthedocs.io/en/latest/) and [M4C](https://github.com/ronghanghu/pythia).