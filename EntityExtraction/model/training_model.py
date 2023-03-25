import argparse
import math
import os
from typing import Type
import os,sys
sys.path.append('./model/')
import torch
from torch.nn import DataParallel
from torch.optim import Optimizer
import transformers
from torch.utils.data import DataLoader
from transformers import AdamW, BertConfig, XLNetConfig, AlbertConfig
from transformers import BertTokenizer,XLNetTokenizer, AlbertTokenizer
# -*- coding:utf-8 -*-
from model import models, prediction
from model import sampling
from model import sampling
from model import util
from model.entities import Dataset
from model.evaluator import Evaluator
from model.input_reader import  BaseInputReader, JsonPredictionInputReader, JsonPredictionInputReaderen
from model.loss import SpERTLoss, Loss
from model.trainer import BaseTrainer
from tqdm import tqdm

import re

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class modelTrainer(BaseTrainer):
    """ Joint entity and relation extraction training and evaluation """

    def __init__(self):
        
        
        self._tokenizer = BertTokenizer.from_pretrained("./model/scripts/data/models/tc2000model",
                                                        do_lower_case=False,
                                                        cache_dir=None)
        self._tokenizer_en = BertTokenizer.from_pretrained("./model/scripts/data/models/enmodel",
                                                        do_lower_case=False,
                                                        cache_dir=None)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prediction = []
        self.types_path = "./model/scripts/data/datasets/type/testcaseplusre_type.json"
        self.max_span_size = 15
        self.spacy_model = "en_core_web_sm"
        self.input_reader = JsonPredictionInputReader
        self.input_reader = self.input_reader(self.types_path, self._tokenizer,
                                        max_span_size=self.max_span_size,
                                        spacy_model=self.spacy_model)
        
        
        self.model = self._load_model(self.input_reader)
        self.model.to(self._device)

    def split2_short_sent(self, content):
        pattern = re.compile('[，。；、]')
        content = content.replace('\r\n', '')
        result = re.split(pattern, content)
        return result

    def predict(self, text: list):
        dataset = self.input_reader.read(text, 'dataset')

        return self._predict(self.model,text[0], dataset, self.input_reader)
    def predict_en(self, text: list):
        
        dataset = self.input_reader_en.read(text, 'dataset')

        return self._predict(self.model_en, dataset, self.input_reader_en)
    

    def _load_model(self, input_reader):
        model_class = models.get_model("model")
        model_path = "./model/scripts/data/models/tc2000model"
        max_pairs = 1000
        prop_drop = 0.1
        size_embedding = 25
        freeze_transformer = False
        cache_dir = None

        config = BertConfig.from_pretrained(model_path)
        util.check_version(config, model_class, model_path)

        config.model_version = model_class.VERSION
        model = model_class.from_pretrained(model_path,
                                            config=config,
                                            # model model parameters
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            relation_types=input_reader.relation_type_count - 1,
                                            entity_types=input_reader.entity_type_count,
                                            max_pairs=max_pairs,
                                            prop_drop=prop_drop,
                                            size_embedding=size_embedding,
                                            freeze_transformer=freeze_transformer,
                                            cache_dir=cache_dir)

        return model

    def _predict(self, model: torch.nn.Module,sentence:str, dataset: Dataset, input_reader: BaseInputReader):
        dataset.switch_mode(Dataset.EVAL_MODE)
        eval_batch_size = 1
        sampling_processes = 4
        rel_filter_threshold = 0.4
        predictions_path = ""
        data_loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False, drop_last=False,
                                 num_workers=sampling_processes, collate_fn=sampling.collate_fn_padding, pin_memory=True)

        pred_entities = []
        pred_relations = []
        entity_veclist = []

        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.document_count / eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc='Predict'):
                # move batch to selected device
                batch = util.to_device(batch, self._device)

                # run model (forward pass)
                result = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                               entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                               entity_spans=batch['entity_spans'], entity_sample_masks=batch['entity_sample_masks'],
                               inference=True)
                entity_clf, rel_clf, rels = result

                # convert predictions
                predictions = prediction.convert_predictions(entity_clf, rel_clf, rels,
                                                             batch, rel_filter_threshold,
                                                             input_reader)

                batch_pred_entities, batch_pred_relations = predictions
                pred_entities.extend(batch_pred_entities)
                pred_relations.extend(batch_pred_relations)
                # entity_veclist.extend(entity_vec)
        self.prediction = prediction.store_predictions(dataset.documents, sentence, pred_entities, pred_relations, predictions_path)
        return self.prediction
    
   
    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self._args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        return optimizer_params




