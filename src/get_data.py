# -*- coding: utf-8 -*-
import os
import yaml
import pandas as pd
import argparse
import re

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    config = read_params(config_path)
    data_path_tech = config["load_data"]["raw_dataset_tech"]
    data_path_business = config["load_data"]["raw_dataset_business"]
    
    data_tech = pd.read_csv(data_path_tech)
    data_business = pd.read_csv(data_path_business)
    
    data = pd.concat([data_tech,data_business])
    
    data = data.drop_duplicates()
    data = data[~data.title.apply(lambda x: len(x.split())<4)]
    data = data[data.text.str.len()<7000]
    data['title'] = data.title.str.replace('\[ВИДЕО\]','')
    data['title'] = data.title.str.replace('\[ФОТО\]','')
    data['title'] = data.title.str.replace('&quot;','')

    data['title'] = data.title.apply(lambda x: x.strip())
    data = data[~data.title.str.contains('Обзор')]
    data = data[~data.title.str.contains('S.T.A.L.K.E.R')]
    data = data[~data.title.str.startswith('Как ')]
    data = data[~data.text.str.contains('Окпо')]
    
    data['text'] = data['text'].str.replace('\n','')
    data['text'] = data['text'].str.replace('¦','')
    data['text'] = data['text'].str.strip()
    data['title'] = data['title'].str.replace('&amp;','&')
    data['title'] = data['title'].str.replace('Nature: ','')
    data['title'] = data['title'].str.replace('Видео: ','')
    data['title'] = data['title'].str.replace('Фото: ','')
    data['title'] = data['title'].str.replace('ОБЗОР: ','')
    data['title'] = data['title'].str.replace('«|»','')
    
    sur_regex = re.compile('^[«»\w]+: ')
    def sur_chng(x):
        if sur_regex.search(x):
            return sur_regex.sub('',x).capitalize()
        else:
            return x
    data.title = data.title.apply(sur_chng)
    data = data[~data.title.str.contains('^realme')]
    data =  data[~data.title.str.contains('^Resident')]
    data = data[~data.title.str.isupper()]
    data = data.reset_index(drop=True)
  
    data = data[data.title.apply(lambda x: len(x.split())<=20)]
    data = data[data.title.apply(lambda x: len(x.split())>=9)]
    
    data = data[data.title.str.len()*8<data.text.str.len()]
    data = data[~data.title.str.contains('это работает')]
    
    
    data.to_csv(config["load_data"]["processed_data"] ,index=False)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = get_data(config_path = parsed_args.config)