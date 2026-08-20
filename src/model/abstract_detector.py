import torch
import torch.nn as nn
import torch.nn.functional as F


class AbstractDetector(nn.Module):
    def __init__(self, config):
        super(AbstractDetector, self).__init__()
        self.dev = config.device
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, data_dict: dict):
        pass

    def calc_loss(self, data_dict: dict):
        logits = self.forward(data_dict)
        return self.get_loss(data_dict, logits)

    def load_basic_SR_data(self, data_dict):
        return data_dict['item_seq'], data_dict['seq_len'], data_dict['target']

    def get_loss(self, data_dict, logits, item_seq=None, target=None):
        if item_seq is None:
            item_seq = data_dict['item_seq']
        if target is None:
            target = data_dict['target']

        if self.loss_type.upper() == 'BCE':

            neg_item = self.get_negative_items(item_seq, target, num_samples=1)
            pos_score = torch.gather(logits, -1, target.unsqueeze(-1))
            neg_score = torch.gather(logits, -1, neg_item)
            loss = -torch.mean(
                F.logsigmoid(pos_score) + torch.log(1 - torch.sigmoid(neg_score) + 1e-7))
        elif self.loss_type.upper() == 'BPR':
            neg_item = self.get_negative_items(item_seq, target, num_samples=1)
            pos_score = torch.gather(logits, -1, target.unsqueeze(-1))
            neg_score = torch.gather(logits, -1, neg_item)
            loss = -torch.mean(F.logsigmoid(pos_score - neg_score))
        elif self.loss_type.upper() == 'CE':
            loss = self.cross_entropy(logits, target)
        else:
            loss = torch.zeros((1,)).to(self.dev)
        return loss

    def gather_index(self, output, index):
        gather_index = index.view(-1, 1, 1).repeat(1, 1, output.size(-1))
        gather_output = output.gather(dim=1, index=gather_index)
        return gather_output.squeeze()

    def get_target_and_length(self, target_info):
        target = target_info['target']
        try:
            tar_len = target_info['target_len']
        except:
            raise Exception(f"{self.__class__.__name__} requires target sequences, set use_tar_seq to true in "
                            f"experimental settings")
        return target, tar_len

    def get_negative_items(self, input_item, target, num_samples=1):
        sample_prob = torch.ones(input_item.size(0), self.num_items, device=target.device)
        sample_prob.scatter_(-1, input_item, 0.)
        sample_prob.scatter_(-1, target.unsqueeze(-1), 0.)
        neg_items = torch.multinomial(sample_prob, num_samples)

        return neg_items

    def pack_to_batch(self, prediction):
        if prediction.dim() < 2:
            prediction = prediction.unsqueeze(0)
        return prediction

    def calc_total_params(self):
        return sum([p.nelement() for p in self.parameters()])

    def load_pretrain_model(self, pretrain_model):
        self.load_state_dict(pretrain_model.state_dict())
        del pretrain_model

    def MISP_pretrain_forward(self, data_dict: dict):
        pass

    def MIM_pretrain_forward(self, data_dict: dict):
        pass

    def PID_pretrain_forward(self, data_dict: dict):
        pass
