# Examining a Consistency of Visual Commonsense Reasoning based on Person Grounding
Dataset and PyTorch implementation of "Examining a Consistency of Visual Commonsense Reasoning based on Person Grounding
<p align="center">
<img width="624" alt="image" src="https://github.com/Haena0320/consistency-pg/assets/68367329/3d6deaea-0148-4f7f-88d7-17b1f3ecdaf6">
</p>

## Introduction
Visual Commonsense Reasoning-Contrast Sets (VCR-CS) is a dataset that explores the consistent commonsense reasoning ability of models based on person grounding.

## Dataset Access
Our dataset is located at vcr_cs_val folder

## Dataset Format
Here is a introduction to the data format.
```
{'movie': '3036_IN_TIME',
 'objects': ['person','person','person','person', ...],
 'interesting_scores': [0],
 'answer_likelihood': 'possible',
 'img_fn': 'lsmdc_3036_IN_TIME/3036_IN_TIME_00.37.33.594-00.37.54.485@0.jpg',
 'metadata_fn': 'lsmdc_3036_IN_TIME/3036_IN_TIME_00.37.33.594-00.37.54.485@0.json',
 'original_question': ['What', 'is', [0], 'thinking', '?'],
 'counter_question': ['What', 'is', [1], 'thinking', '?'],
 'answer_match_iter': [1, 2, 3, 0],
 'answer_sources': [16688, 6052, 19179, 35],
 'answer_choices': [[[1],'is','thinking','she','would','rather',...]...],
 'counter_answer_label': 0,
 'original_answer_label': 3,
 'img_id': 'val-12',
 'question_number': 0,
 'annot_id': 'val-35',
 'match_fold': 'val-0',
 'match_index': 35}
```

## Framework Code
Here is an quick start for our framework, PINT.
Code described here includes (1) Data processing, (2) training PINT on VCR and (3) validation on VCR-CS.
We used single V100 GPU both to train and valid the PINT. 
 
### 0. Create a new python environment
```
 python -m venv pint
 source pint/bin/activate
 pip install -r requirements.txt
 
 ```
 
### 1. Data Processing
First, obtain the preprocessed image and text file from VILLA. 
Then, place them in .../vcr_data for further preprocessing step. 
Next, run the preprocessing scripts in the environment described below. First generate VCR training data for PINT
 ```
 python prepro_vcr_pg.py --annotation ../../train.jsonl --split train --output vcr_train_pint.db
 ```

Next, process VCR-CS validation data for PINT
 ```
 python prepro_vcr_pg.py --annotation ../../val_vcr_cs_original.jsonl --split val --output val_vcr_cs_original.db
  python prepro_vcr_pg.py --annotation ../../val_vcr_cs_counter.jsonl --split val --output val_vcr_cs_counter.db
 ```
 
### 2. Training PINT on VCR
 First, run launch_container.sh following the scripts with the paths for each argument.
  ```
 source launch_container.sh ../../txt_db ../../img_db ../../finetune ../../pretrained
  ```

Inside the created container above, run
  ```
 CUDA_VISIBLE_DEVICES=3 horovodrun -np 1 python train_multi.py --config config/pretrain-vcr-multi-base.json --output_dir {path_for_pint_checkpoint} --train_person_mask --train_person_corrupt
  ```
 
### 3. Validation on VCR-CS
Inside the created container above, run
 ```
CUDA_VISIBLE_DEVICES=3 horovodrun -np 1 python inf_vcr_ce.py --txt_db "{path_for_vcr_cs_original.db;path_for_vcr_cs_counter.db}" --img_db "./../img_db/vcr_gt_val;./../img_db/vcr_val" --split val --output_dir {path_for_pint_checkpoint} --checkpoint 7000 --pin_mem --fp16
 ``` 

## Baseline Models
We describe baseline models here. 
They are selected from six Transformer based multimodal pretrained models which have great performance on vision-language understanding tasks including VCR dataset.
1. [ViLBERT](https://github.com/jiasenlu/vilbert_beta/tree/master)
2. [UNITER](https://github.com/ChenRocks/UNITER)
3. [VILLA](https://github.com/zhegan27/VILLA)
4. [VL-Bart](https://github.com/j-min/VL-T5/tree/main)
5. [VL-T5](https://github.com/j-min/VL-T5/tree/main)
6. [MERLOT-Reserve](https://github.com/rowanz/merlot_reserve/blob/main/README.md)

## Acknowledgement
+ Our dataset source is [VCR](https://github.com/rowanz/r2c/)
+ Our framework is based on the code of [VILLA](https://github.com/zhegan27/VILLA)
