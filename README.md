# MLLM-enriched Multimodal Teacher Action Quality Assessment (3MTAQA) Dataset 

## About the dataset：
![image](https://github.com/Anonymous-AQA/3MTAQADataset/blob/main/MUSDL%2BLVFL/fig/Dataset.jpg)
Figure 1.Overview of the 3MTAQA Dataset.This dataset consists of four types of actions, each labeled with key information such as action type, score, and text description. 

  The MLLM-enriched Multimodal Teacher Action Quality Assessment (3MTAQA) dataset is a multimodal dataset that includes both visual and text. The visual modality data of this dataset comes from the original records of the 6th National Teacher Skills Competition, and the video characteristics of each category are shown in Table 1. Text modality data is a high-quality text description generated by understanding the actions in the video through the multimodal large language model (MLLM). 

  			
Table 1. The detailed information of 3MTAQA dataset.
| Action_type | #Samples | Avg.Seq.Len | Min.Seq.Len| Max.Seq.Len |
| :---: | :---: | :---: | :---: | :---: | 
|Inviting students to answer questions|	596|	132	|112	|151|
|Pointing to teaching devices|	530|	464|402|	509|
|Walking around classroom|	585|	596|	550|	675|
|Writing on the blackboard|543	|592|	550|648|

1.**Action_type** represents the action category.

2.**#Samples** represents the number of samples in each category. 

3.**Avg.Seq.Len** represents the average number of video frames for each category. 

4.**Min.Seq.Len** represents the minimum number of video frames for each category.

5.**Max.Seq.Len** represents the maximum number of video frames for each category. 

The detailed partition of training set and test set is given in our paper.

## About the LVFL model：

### Requirement
Python >= 3.6

Pytorch >=1.8.0

### Dataset Preparation
**1.3MTAQA dataset**

If the article is accepted for publication, you can download our prepared 3MTAQA dataset demo from ["Google Drive"](https://drive.google.com/file/d/16KydZ6cJCjpulp5NRAzmCdCcWb0fb80-/view?usp=sharing) . Then, please move the uncompressed data folder to `./data/frames`. We used the I3D backbone pretrained on Kinetics([Google Drive](https://drive.google.com/file/d/1M_4hN-beZpa-eiYCvIE7hsORjF18LEYU/)).

**2.MTL-AQA dataset**(["Google Drive"](https://drive.google.com/file/d/1T7bVrqdElRLoR3l6TxddFQNPAUIgAJL7/))

### Training & Evaluation

LVFL, as a plug and play module, has good versatility and can be easily integrated into other models. In this study, we first select classic action quality assessment models R(2+1)D-34-WD, USDL, MUSDL, and DAE as baseline models, and then integrate LVFL into the above models, namely R(2+1)D-34-WD+LVFL,USDL+LVFL, MUSDL+LVFL and DAE+LVFL, respectively, to evaluate the action quality.Take **MUSDL+LVFL** as an example,To train and evaluate on 3MTAQA:

` python -u main.py  --lr 1e-4 --weight_decay 1e-5 --gpu 0 `

If you use the 3MTAQA dataset, please cite this paper: A Language-Guided Visual Feature Learning Strategy for Multimodal Teacher Action Quality Assessment.
