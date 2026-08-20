import logging
import re

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, \
    accuracy_score
from tqdm import tqdm
from src.evaluation.metrics import Metric
from src.utils.utils import tensor_to_device


class Estimator:
    def __init__(self, config):
        self.config = config
        self.metrics = config.metric
        self.k_list = config.k
        self.dev = config.device
        self.metric_res_dict = {}
        self.eval_loss = 0.
        self.max_k = max(self.k_list)
        self.split_type = config.split_type
        self.eval_mode = config.eval_mode
        self.mask_history = config.mask_history
        self.test_device = config.test_device
        self.neg_size = 0
        if self.eval_mode != 'full':
            self.neg_size = int(re.findall(r'\d+', self.eval_mode)[0])
            self.eval_mode = self.eval_mode[:3]
        self._reset_metrics()

    def _reset_metrics(self):
        for metric in self.metrics:
            self.metric_res_dict[metric] = 0.
        self.eval_loss = 0.

    @torch.no_grad()
    def evaluate(self, eval_loader, model):
        if self.test_device == 'cpu':
            return self.evaluate_on_cpu(eval_loader, model)
        return self.evaluate_on_gpu(eval_loader, model)

    @torch.no_grad()
    def test(self, eval_loader, model):
        if self.test_device == 'cpu':
            return self.test_on_cpu(eval_loader, model, save_emb=False)
        return self.test_on_gpu(eval_loader, model)

    @torch.no_grad()
    def evaluate_on_gpu(self, eval_loader, model):
        model.eval()
        self._reset_metrics()

        eval_sample_size = len(eval_loader.dataset)
        eval_iter = tqdm(enumerate(eval_loader), total=len(eval_loader))
        eval_iter.set_description(f'do evaluation...')
        for _, batch_dict in eval_iter:
            tensor_to_device(batch_dict, self.dev)
            logits = model(batch_dict)
            model_loss = model.calc_loss(batch_dict)
            logits = self.neg_sample_select(batch_dict, logits)
            self.calc_metrics(logits, batch_dict['label'])
            self.eval_loss += model_loss.item()

        for metric in self.metrics:
            for k in self.k_list:
                self.metric_res_dict[f'{metric}@{k}'] /= float(eval_sample_size)

        eval_loss = self.eval_loss / float(len(eval_loader))

        return self.metric_res_dict, eval_loss

    @torch.no_grad()
    def evaluate_on_cpu(self, eval_loader, model):
        return self.test_on_cpu(eval_loader, model, mode='eval')

    @torch.no_grad()
    def test_on_gpu(self, test_loader, model, save_emb=False):
        model.eval()
        self._reset_metrics()

        test_sample_size = len(test_loader.dataset)
        test_iter = tqdm(enumerate(test_loader), total=len(test_loader))
        test_iter.set_description(f'do test...')
        for _, batch_dict in test_iter:
            tensor_to_device(batch_dict, self.dev)
            logits = model(batch_dict)
            logits = self.neg_sample_select(batch_dict, logits)
            self.calc_metrics(logits, batch_dict['label'])

        for metric in self.metrics:
            for k in self.k_list:
                self.metric_res_dict[f'{metric}@{k}'] /= float(test_sample_size)
        return self.metric_res_dict

    @torch.no_grad()
    def test_on_cpu(self, test_loader, model, mode='test', save_emb=False):
        model.eval()

        if save_emb:
            all_labels = []
            all_news_emb = []
            test_iter = tqdm(enumerate(test_loader), total=len(test_loader), desc='save news embeddings')
            for _, batch_dict in test_iter:
                tensor_to_device(batch_dict, self.dev)
                batch_news_emb = model.get_news_emb(batch_dict)
                all_news_emb.append(batch_news_emb)
                all_labels.append(batch_dict['label'])

            all_labels = torch.cat(all_labels, dim=0).detach().cpu().numpy()
            all_news_emb = torch.cat(all_news_emb, dim=0).detach().cpu().numpy()

            np.save(f'vis/{self.config.dataset}-{self.config.model}-emb', all_news_emb)
            np.save(f'vis/{self.config.dataset}-{self.config.model}-label', all_labels)

            logging.info('Saving news embeddings... Done.')

        self._reset_metrics()

        eval_iter = tqdm(enumerate(test_loader), total=len(test_loader))
        eval_iter.set_description(f'do evaluation...')

        logits_list = []
        answer_list = []
        total_loss = 0.

        for i, batch_dict in eval_iter:
            tensor_to_device(batch_dict, self.dev)
            logits = model(batch_dict).cpu().numpy()
            loss = model.calc_loss(batch_dict)
            ans = batch_dict['label'].cpu().numpy()
            logits_list.append(logits)
            answer_list.append(ans)
            total_loss += loss.item()

        test_loss = total_loss / float(len(test_loader))
        y_logits = np.concatenate(logits_list, axis=0)
        y_true = np.concatenate(answer_list, axis=0)

        final_preds = (y_logits >= 0.5).astype(int)

        report = classification_report(y_true, final_preds, digits=4)
        logging.info(report)

        test_acc = accuracy_score(y_true, final_preds)
        precision = precision_score(y_true, final_preds, average='macro', zero_division=0)
        recall = recall_score(y_true, final_preds, average='macro', zero_division=0)
        f1 = f1_score(y_true, final_preds, average='macro', zero_division=0)

        self.metric_res_dict['acc'] = test_acc
        self.metric_res_dict['precision'] = precision
        self.metric_res_dict['recall'] = recall
        self.metric_res_dict['f1'] = f1
        if mode == 'test':
            self.metric_res_dict['classification_report'] = report

        return self.metric_res_dict, test_loss

    def calc_metrics(self, prediction, target):
        _, topk_index = torch.topk(prediction, self.max_k, -1)
        topk_socre = torch.gather(prediction, index=topk_index, dim=-1)
        idx_sorted = torch.argsort(topk_socre, dim=-1, descending=True)
        top_k_item_sorted = torch.gather(topk_index, index=idx_sorted, dim=-1)

        for metric in self.metrics:
            for k in self.k_list:
                score = getattr(Metric, f'{metric.upper()}')(top_k_item_sorted, target, k)
                self.metric_res_dict[f'{metric}@{k}'] += score

    def calc_metrics_(self, prediction, target):
        _, topk_index = torch.topk(prediction, self.max_k, -1)
        topk_socre = torch.gather(prediction, index=topk_index, dim=-1)
        idx_sorted = torch.argsort(topk_socre, dim=-1, descending=True)
        max_k_item_sorted = torch.gather(topk_index, index=idx_sorted, dim=-1)

        metric_res_dict = {}
        for metric in self.metrics:
            for k in self.k_list:
                score = getattr(Metric, f'{metric.upper()}')(max_k_item_sorted, target, k)
                metric_res_dict[f'{metric}@{k}'] += score

        return metric_res_dict

    def neg_sample_select(self, data_dict, prediction):
        if self.mask_history:
            item_seq = data_dict['item_seq']
            prediction = torch.scatter(prediction, -1, item_seq, -1e7)

        if self.eval_mode == 'full':
            return prediction

        item_seq, target = data_dict['item_seq'], data_dict['label']
        target = target.unsqueeze(-1)
        mask_item = torch.cat([item_seq, target], dim=-1)

        if self.eval_mode == 'uni':
            sample_prob = torch.ones_like(prediction, device=self.dev) / prediction.size(-1)
        elif self.eval_mode == 'pop':
            if self.popularity.size(0) != prediction.size(-1):
                self.popularity = torch.cat([self.popularity, torch.zeros((1,)).to(self.dev)], -1)
            sample_prob = self.popularity.unsqueeze(0).repeat(prediction.size(0), 1)
        else:
            raise KeyError(f'Invalid eval_model: {self.eval_mode}! Choose from [full, popxxx, unixxx].')
        sample_prob = sample_prob.scatter(dim=-1, index=mask_item, value=0.)
        neg_item = torch.multinomial(sample_prob, self.neg_size)

        rank_item = torch.cat([neg_item, target], dim=-1)
        mask = torch.ones_like(prediction, device=self.dev).bool()
        mask = mask.scatter(dim=-1, index=rank_item, value=False)
        masked_pred = torch.masked_fill(prediction, mask, 0.)

        return masked_pred


if __name__ == '__main__':
    a = torch.arange(0, 9).view(3, -1)
    mask = torch.ones_like(a).bool()
    mask[-1][-1] = False
    print(~mask)
    res = torch.masked_fill(a, mask, False)
    print(a)
    print(res)
