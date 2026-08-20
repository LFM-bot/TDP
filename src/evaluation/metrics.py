import math

import numpy as np
import torch


class Metric:
    @staticmethod
    def HIT(prediction, target, k):
        prediction, target = Metric.process(prediction, target, k)
        hit = ((prediction - target) == 0).sum(dim=-1).double()
        hit = hit.sum().item()
        return hit

    @staticmethod
    def NDCG(prediction, target, k):
        prediction, target = Metric.process(prediction, target, k)
        hit = ((prediction - target) == 0).sum(dim=-1).double()
        row, col = ((prediction - target) == 0.).nonzero(as_tuple=True)
        ndcg = hit.scatter(index=row, src=1. / torch.log2(col + 2).double(), dim=-1)
        ndcg = ndcg.sum().item()
        return ndcg

    @staticmethod
    def MRR(prediction, target, k):
        prediction, target = Metric.process(prediction, target, k)
        hit = ((prediction - target) == 0).sum(dim=-1).double()
        row, col = ((prediction - target) == 0.).nonzero(as_tuple=True)
        mrr = hit.scatter(index=row, src=1. / (col + 1).double(), dim=-1)
        mrr = mrr.sum().item()
        return mrr

    @staticmethod
    def RECALL(prediction, target, k):
        return Metric.HIT(prediction, target, k)

    @staticmethod
    def process(prediction, target, k):
        if k < prediction.size(-1):
            prediction = prediction[:, :k]
        target = target.unsqueeze(-1)
        return prediction, target


def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT / len(pred_list), NDCG / len(pred_list), MRR / len(pred_list)


def get_full_sort_score(answers, pred_list):

    recall, ndcg = [], []
    for k in [5, 10, 15, 20, 50]:
        recall.append(recall_at_k(answers, pred_list, k))
        ndcg.append(ndcg_k(answers, pred_list, k))

    metric_results = {
        "hit@5": recall[0],
        "ndcg@5": ndcg[0],
        "hit@10": recall[1],
        "ndcg@10": ndcg[1],
        "hit@20": recall[3],
        "ndcg@20": ndcg[3],
        "hit@50": recall[4],
        "ndcg@50": ndcg[4],
    }

    return metric_results


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set([actual[i]])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        cur_actual = [actual[user_id]]
        k = min(topk, len(cur_actual))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in set(cur_actual)) / math.log(j + 2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))


def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


if __name__ == '__main__':
    a = torch.arange(12).view(3, -1)
    a[1, -1] = 0
    print(a)
    hit = (a == 0).sum(dim=-1).float()
    hit_index, rank = (a == 0).nonzero(as_tuple=True)
    print(hit_index, rank)
    score = torch.scatter(hit, index=hit_index, src=1. / torch.log2(rank + 2), dim=-1)
    print(score)
    score = score.mean().cpu().numpy()
    print(score)
