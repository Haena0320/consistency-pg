# Examining a Consistency of Visual Commonsense Reasoning based on Person Grounding
Dataset and PyTorch implementation of "Examining a Consistency of Visual Commonsense Reasoning based on Person Grounding" 

<p align="center"><img width="693"  src='https://github.com/Haena0320/vcr_pg_3/assets/68367329/d7ca78e5-b780-4368-855a-7cf7d04d4394'>

## Introduction
Visual Commonsense Reasoning-Contrast Sets (VCR-CS) is a dataset that explores the consistent commonsense reasoning ability of models based on person grounding.

## Dataset Access
Dataset can be downloaded at [Google Drive](https://drive.google.com/drive/folders/1BuVRy1XDNqIKMtdY1f79gK5lBdlK3knm?usp=sharing).

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

 ### (1) Data Processing
 Run the preprocessing scripts in the environment described below. First generate VCR training data for PINT:
 ```
 python prepro_vcr_pg.py --annotation ../../train.jsonl --split train --output vcr_train_pint.db
 ```

 Next, process VCR-CS validation data for PINT
 ```
 python prepro_vcr_pg.py --annotation ../../val_vcr_cs.jsonl --split val --output vcr_vcr_cs.db
 ```
 
 ### (2) Training PINT on VCR
 First, run launch_container.sh following the scripts with the paths for each argument.
  ```
 source launch_container.sh ../../txt_db ../../img_db ../../finetune ../../pretrained
  ```

Inside the created container above, run
  ```
 CUDA_VISIBLE_DEVICES=3 horovodrun -np 1 python train_multi.py --config config/pretrain-vcr-multi-base.json --output_dir {path_for_pint_checkpoint} --train_person_mask --train_person_corrupt
  ```
 
 ### (3) Validation on VCR-CS
 Inside the created container above, run
 ```
CUDA_VISIBLE_DEVICES=3 horovodrun -np 1 python inf_vcr_ce.py --txt_db "{path_for_vcr_vcr_cs.db}" --img_db "./../img_db/vcr_gt_val;./../img_db/vcr_val" --split val --output_dir {path_for_pint_checkpoint} --checkpoint 8000 --pin_mem --fp16
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
