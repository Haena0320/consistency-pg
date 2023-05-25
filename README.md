# Examining a Consistency of Visual Commonsense Reasoning based on Person Grounding
Dataset and PyTorch implementation of "Examining a Consistency of Visual Commonsense Reasoning based on Person Grounding" 

<img src="https://github.com/Haena0320/vcr_pg_3/files/11563322/figure1.1.pdf"  width="700" height="370">


## Introduction
Visual Commonsense Reasoning-Contrast Sets (VCR-CS) is a dataset that explores the consistent commonsense reasoning ability of models based on person grounding.

Through manually annotation from the experts, we create VCR-CS dataset with 159 pairs. Here are the stats for VCR-CS.

## Dataset Access
Dataset can be downloaded at [Google Drive](https://drive.google.com/drive/folders/1BuVRy1XDNqIKMtdY1f79gK5lBdlK3knm?usp=sharing).

## Dataset Format
Here is a introduction to the data format.

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
