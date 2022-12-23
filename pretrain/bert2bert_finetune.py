#!/usr/bin/env python
# coding:utf-8
"""
Name: bert2bert_finetune.py
Author: Sanjeeva Reddy Dodlapati
Time: 12/12/22 10:55 PM
Desc:

"""

import numpy as np
import pandas as pd
import random

import datasets
from datasets import ClassLabel, load_dataset
from IPython.display import display, HTML

from transformers import EncoderDecoderModel
from transformers import BertTokenizerFast
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import time

train_data = load_dataset('text', data_files = '/home/sdodl001/Downloads/Repos_0/NLP/data/MIMIC/clinical_sentences_pretrain.txt', split='train')
val_data = load_dataset('text', data_files = '/home/sdodl001/Downloads/Repos_0/NLP/data/MIMIC/test.txt', split='train')


tokenizer = BertTokenizerFast.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

encoder_max_length=50
decoder_max_length=50

def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=encoder_max_length)
    # print(f'inputs: {inputs}')
    outputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=decoder_max_length)
    # print(f'outputs: {outputs}')

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

    return batch


batch_size=16

train_data = train_data.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    # remove_columns=["article", "highlights", "id"]
    remove_columns=["text"])

train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],)


val_data = val_data.select(random.sample(range(1, val_data.num_rows), 160000))

val_data = val_data.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    # remove_columns=["article", "highlights", "id"]
    remove_columns=["text"])


bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("emilyalsentzer/Bio_ClinicalBERT", "emilyalsentzer/Bio_ClinicalBERT")

bert2bert.config.decoder_start_token_id = tokenizer.cls_token_id
bert2bert.config.eos_token_id = tokenizer.sep_token_id
bert2bert.config.pad_token_id = tokenizer.pad_token_id
bert2bert.config.vocab_size = bert2bert.config.encoder.vocab_size

bert2bert.config.max_length = 40
bert2bert.config.min_length = 5
bert2bert.config.no_repeat_ngram_size = 2
bert2bert.config.early_stopping = True
bert2bert.config.length_penalty = 1.5
bert2bert.config.num_beams = 4

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    output_dir="./",
    logging_steps=10000,
    save_steps=50000,
    eval_steps=50000,
    num_train_epochs=10)



rouge = datasets.load_metric("rouge")
def compute_metrics(pred):
    # print(f'Predictions: {pred}')
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    # print(f'Predictions...............................: {pred_str}')
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    # print(f'Labels~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~: {label_str}')

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge1"])["rouge1"].mid
    print(f'rouge1_precision : {round(rouge_output.precision, 4)}, rouge1_recall : {round(rouge_output.recall, 4)}, rouge1_fmeasure": round(rouge_output.fmeasure, 4) ')

    return {
        "rouge1_precision": round(rouge_output.precision, 4),
        "rouge1_recall": round(rouge_output.recall, 4),
        "rouge1_fmeasure": round(rouge_output.fmeasure, 4),
    }
# instantiate trainer
trainer = Seq2SeqTrainer(
    model=bert2bert,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
)
trainer.train()


test_data = val_data.select(random.sample(range(1, val_data.num_rows), 160000))


def generate_summary(batch):
    # cut off at BERT max length 512
    inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=50, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = bert2bert.generate(input_ids, attention_mask=attention_mask)

    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # print(f'Inputs ...............................: {batch["text"]}')
    # print(f'Predictions~~~~~~~~~~~~~~~~~~~~~~~~~~~: {output_str}')

    batch["outputs"] = output_str

    return batch


results = test_data.map(generate_summary, batched=True, batch_size=batch_size)
print(rouge.compute(predictions=results["outputs"], references=results["text"], rouge_types=["rouge1"])["rouge1"].mid)





