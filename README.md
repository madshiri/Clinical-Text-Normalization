# Clinical-Text-Norm-BERT
This repository holds code to implement personalized text normalization of clinical notes

## Evnvironment Setup
```python
run pip install requirements.txt
```
## Data Download
1. n2c2 NLP research data - 2018(Track2): Adverse Drug Events and Medication Extraction downloaded from [Harvard Medical School](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/). Need to sign agreement forms and request access for the data.
2. MMIC-III Clinical Database downloaded from [PhysioNet](https://physionet.org/content/mimiciii/1.4/). Needs to go through CITI Data or Specimens Only Research Training. Access will be granted after submitting the training certificate. 

## Data Preprocess
To preprocess the data, run ./data_preprocess/preprocess_mimic_notes.ipynb. Change data paths to your data directory.



## Model
[ClinicalBERT](https://aclanthology.org/W19-1909.pdf) which is finetuned on clinical notes is to be adopted for clinical notes text normalization. You can download the pretrained ClinicalBERT model from hugginface (https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT).


## Train Clinical-Text-Norm-BERT Encoder-Decoder
Run finetune_bert2bert.py (give appropriate data folder paths)

