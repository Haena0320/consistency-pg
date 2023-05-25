# Examining a Consistency of Visual Commonsense Reasoning based on Person Grounding
Dataset and PyTorch implementation of "Examining a Consistency of Visual Commonsense Reasoning based on Person Grounding" 

<img src="https://github.com/Haena0320/vcr_pg_3/files/11563322/figure1.1.pdf"  width="700" height="370">


## Introduction
Visual Commonsense Reasoning-Contrast Sets (VCR-CS) is a dataset that explores the consistent commonsense reasoning ability of models based on person grounding.
Through manually annotation from the experts, we create VCR-CS dataset with 159 pairs. Here are the stats for VCR-CS.

| |VCR-CS|
|------|---|
|Number of question|159|
|Number of images|143|
|Number of movie covered|16|
|Average quesiton length|6.38|
|Average answer length|7.08|
|Average # of object mentioned|1.88|
|Question/Image ratio|1.11|

## Dataset Access
Dataset can be downloaded at [Google Drive](https://drive.google.com/drive/folders/1BuVRy1XDNqIKMtdY1f79gK5lBdlK3knm?usp=sharing).

## Dataset Format
Here is a introduction to the data format.
'''
{'counter_question': ['What', 'is', [1], 'thinking', '?'],
 'counter_answer_label': 0,
 'movie': '3036_IN_TIME',
 'objects': ['person',
  'person',
  'person',
  'person',
  'person',
  'person',
  'person',
  'person',
  'tie',
  'chair'],
 'interesting_scores': [0],
 'answer_likelihood': 'possible',
 'img_fn': 'lsmdc_3036_IN_TIME/3036_IN_TIME_00.37.33.594-00.37.54.485@0.jpg',
 'metadata_fn': 'lsmdc_3036_IN_TIME/3036_IN_TIME_00.37.33.594-00.37.54.485@0.json',
 'original_question': ['What', 'is', [0], 'thinking', '?'],
 'answer_match_iter': [1, 2, 3, 0],
 'answer_sources': [16688, 6052, 19179, 35],
 'answer_choices': [[[1],
   'is',
   'thinking',
   'she',
   'would',
   'rather',
   'have',
   'gone',
   'to',
   'the',
   'science',
   'museum',
   'or',
   'the',
   'beach',
   '.'],
  ['She',
   'wants',
   [7],
   'to',
   'sit',
   'back',
   'in',
   [9],
   'and',
   'let',
   'her',
   'take',
   'care',
   'of',
   'the',
   'cleanup',
   '.'],
  ['She', 'can', "'", 't', 'believe', 'what', 'she', 'is', 'seeing', '.'],
  [[0],
   'is',
   'wondering',
   'if',
   [1],
   'is',
   'going',
   'to',
   'kiss',
   'her',
   '.']],
 'original_answer_label': 3,
 'img_id': 'val-12',
 'question_number': 0,
 'annot_id': 'val-35',
 'match_fold': 'val-0',
 'match_index': 35}
'''

## Framework Code
Here is an quick start for our framework, PINT.
Code described here includes (1) training on VCR and (2) validation on VCR-CS.

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
