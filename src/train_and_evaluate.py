import os
import warnings
import sys
import pandas as pd
from get_data import read_params
import argparse
import joblib
import json
import re
import tqdm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import numpy as np
import datasets
from tabulate import tabulate
import nltk
from datetime import datetime
from datasets import Dataset, DatasetDict, load_metric
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer

os.environ["USE_DEEPSPEED"] = "1"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

os.environ["WANDB_DISABLED"] = "true"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] training using {}".format(torch.cuda.get_device_name(0)))

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
  
    
    return {k: round(v, 4) for k, v in result.items()}

def preprocess_function(examples):
    inputs = ['[{}] '.format(len(ttl.split()))+doc for doc,ttl in zip(examples["text"],examples["title"])]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["title"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def train_and_evaluate(config):
#     config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]
    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    dataset = DatasetDict({'train':Dataset.from_dict(train),'valid':Dataset.from_dict(test)})
    tokenized_train = dataset['train'].map(preprocess_function, batched=True)
    tokenized_valid = dataset['valid'].map(preprocess_function, batched=True)
    
    args = Seq2SeqTrainingArguments(
        output_dir ='saved_models',
        overwrite_output_dir = True,
        logging_dir = 'saved_models/logs',
        evaluation_strategy = "epoch",
        learning_rate = config["model_param"]["learning_rate"],
        per_device_train_batch_size = config["model_param"]["batch_size"],
        per_device_eval_batch_size = config["model_param"]["batch_size"],
        weight_decay = config["model_param"]["weight_decay"],
        num_train_epochs = config["model_param"]["num_train_epochs"],
        lr_scheduler_type = config["model_param"]["lr_scheduler_type"],
        do_train = True,
        do_eval = False,
        predict_with_generate=True,
        report_to = None,
        save_total_limit = config["model_param"]["save_total_limit"],
        warmup_steps = config["model_param"]["warmup_steps"],
    )


    trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
    
    train_result = trainer.train()
    trainer.save_model(os.path.join(model_dir,config["model_param"]["model_name"]))
    
    
    valuation =  trainer.evaluate()
    scores_file = config["reports"]["metrics"]
    with open(scores_file, "w") as fd:
        json.dump(valuation, fd)

        
    

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    
    config = read_params(parsed_args.config)

    model_checkpoint = config["model_param"]["model_checkpoint"]
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    
    max_input_length = config["model_param"]["max_input_length"]
    max_target_length = config["model_param"]["max_target_length"]
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    pad_on_right = tokenizer.padding_side == "right"
    
    nltk.download("punkt", quiet=True)
    metric = datasets.load_metric("rouge")
    
    train_and_evaluate(config)