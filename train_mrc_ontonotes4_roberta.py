import os
from bert_mrc_pytorch.bert_mrc_global_pointer import BertGlobalPointer,\
    BertGlobalPointerWithLabel,\
    BertEfficientGlobalPointerWithLabel,\
    BertEfficientGlobalPointer
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, AdamW, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
import json
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from einops import rearrange
from config import ontonotes4_config as config
import argparse
import re
import matplotlib.pyplot as plt
from pathlib import Path


torch.manual_seed(3407)  # pytorch random seed
torch.cuda.manual_seed(3407)
torch.cuda.manual_seed_all(3407)
np.random.seed(3407)
# torch.backends.cudnn.deterministic = True
tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_model_path)
maxlen = config.max_seq_len
categories = set()
if not os.path.exists(config.model_save_dir):
    os.makedirs(Path(config.model_save_dir).parent, exist_ok=True)


def load_msra_data(filename, mode='train'):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    sample_lens = []
    with open(filename, encoding='utf-8') as f:
        l = json.load(f)
    for line in l:
        sample_lens.append(len(line['context'].split()))
        if mode != 'test':
            if line['start_position']:
                D.append(line)
                categories.add(line['entity_label'])
        else:
            D.append(line)
            if line['start_position']:
                categories.add(line['entity_label'])
    print(f"{filename}: maxlen {max(sample_lens)}, avglen {np.mean(sample_lens)}, minlen {min(sample_lens)}")
    # plt.hist(sample_lens, 50, density=True, alpha=0.75)
    # plt.grid(True)
    # plt.show()
    return D


train_data = load_msra_data(config.train_data_path, mode='test')
valid_data = load_msra_data(config.dev_data_path, mode='test')
test_data = load_msra_data(config.test_data_path, mode='test')

categories = list(sorted(categories))
if not config.random_label_embedding:
    word_embedding = {}
    with open(config.label_embedding_path, encoding='utf8') as f:
        f.readline()
        for line in f:
            line = line.split()
            word = line[0]
            embedding = list(map(float, line[1:]))
            word_embedding[word] = embedding
    label_embedding = []
    for cat in categories:
        label_embedding.append(word_embedding['['+cat+']'])

    label_embedding = torch.from_numpy(np.array(label_embedding, dtype=np.float32))
    print(f'label embedding generated from {config.label_embedding_path}')
else:
    label_embedding = None
if config.single_anwser:
    num_heads = 1
else:
    num_heads = len(categories)
if config.use_efficient_global_pointer:
    if config.add_label_embedding:
        model = BertEfficientGlobalPointerWithLabel.from_pretrained(config.pretrained_model_path, head=num_heads,
                                                             head_size=64, label_embedding=label_embedding, dropout=config.dropout_rate)
    else:
        model = BertEfficientGlobalPointer.from_pretrained(config.pretrained_model_path, head=num_heads,
                                                    head_size=64, dropout=config.dropout_rate)
else:
    if config.add_label_embedding:
        model = BertGlobalPointerWithLabel.from_pretrained(config.pretrained_model_path, head=num_heads,
                                                    head_size=64, label_embedding=label_embedding, dropout=config.dropout_rate)
    else:
        model = BertGlobalPointer.from_pretrained(config.pretrained_model_path, head=num_heads,
                                                    head_size=64, dropout=config.dropout_rate)


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)

    return np.array(outputs)


def get_map(t, offsets, token_type_ids):
    """map的主要目的是将一个片段的token的开始和结尾映射到tokenize后的位置"""
    start_map, end_map = {}, {}
    for i, (token, type) in enumerate(zip(t, token_type_ids)):
        if token in tokenizer.all_special_tokens:
            continue
        if type == 0:
            continue
        start, end = offsets[i]
        start_map[start] = i
        end_map[end-1] = i
    return start_map, end_map


def reloc(start_position, end_position, text):
    """按照字母级别进行重定位英文文本"""
    # 需要保留空格，空格的位置也在offset中体现了，第一个不用加1，其他要加上空格的位置
    words = text.split()
    lens = [-1] + np.cumsum([len(x) if i == 0 else len(x)+1 for i, x in enumerate(words)]).tolist()
    new_start, new_end = [], []
    for start, end in zip(start_position, end_position):
        new_start.append(lens[start]+1)
        new_end.append(lens[end+1]-1)
    return new_start, new_end


class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len, mode='train'):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.mode = mode

    def __getitem__(self, index):
        d = self.data[index]
        d['query'] = ''.join(d['query'].split())
        d['context'] = ''.join(d['context'].split())
        outputs = tokenizer(d['query'], d['context'], return_offsets_mapping=True, truncation=True, max_length=self.seq_len)
        tokens = tokenizer.convert_ids_to_tokens(outputs['input_ids'])
        start_map, end_map = get_map(tokens, outputs['offset_mapping'], outputs['token_type_ids'])
        token_ids = outputs['input_ids']
        segment_ids = outputs['token_type_ids']
        mask = outputs['attention_mask']
        if config.single_anwser:
            labels = np.zeros((1, self.seq_len, self.seq_len))
        else:
            labels = np.zeros((len(categories), self.seq_len, self.seq_len))

        for start, end in zip(d['start_position'], d['end_position']):
            label = d['entity_label']
            if start in start_map and end in end_map:
                start = start_map[start]
                end = end_map[end]
                label = categories.index(label)
                if config.single_anwser:
                    labels[0, start, end] = 1
                else:
                    labels[label, start, end] = 1

        return d, token_ids, mask, segment_ids, labels[:, :len(token_ids), :len(token_ids)]

    def pad(self, lst, value=0, max_length=None):
        max_length = max_length or self.seq_len
        if len(lst) > max_length:
            truncate_len = len(lst) - max_length
            lst = lst[:-truncate_len-1] + lst[-1:]
        else:
            add_len = max_length - len(lst)
            lst = lst + [value] * add_len
        return lst

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    d, token_ids, mask, segment_ids, labels = list(zip(*batch))

    token_ids = torch.LongTensor(sequence_padding(token_ids))
    segment_ids = torch.LongTensor(sequence_padding(segment_ids))
    labels = torch.LongTensor(sequence_padding(labels, seq_dims=3))
    mask = torch.LongTensor(sequence_padding(mask))

    return d, token_ids, mask, segment_ids, labels

train_dataset = TextSamplerDataset(train_data, maxlen)
valid_dataset = TextSamplerDataset(valid_data, maxlen)
test_dataset = TextSamplerDataset(test_data, maxlen)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=8)
valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=8)

def build_optimizer_and_scheduler(model, config, total_steps):

    module = (model.module if hasattr(model, "module") else model)
    model_param = list(module.named_parameters())

    bert_param = []
    other_param = []
    for name, param in model_param:
        if name.split('.')[0] == 'bert':
            bert_param.append((name, param))
        else:
            other_param.append((name, param))

    no_decay = ["bias", "LayerNorm.weight", 'layer_norm']
    optimizer_grouped_parameters = [
        # bert module
        {'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay, 'lr': config.lr},
        {'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': config.lr},

        # other module
        {'params': [p for n, p in other_param if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay, 'lr': config.other_lr},
        {'params': [p for n, p in other_param if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': config.other_lr}
    ]

    warmup_steps = int(config.warmup_proportion * total_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr, eps=config.adam_epsilon)
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps,
                                                          num_training_steps=total_steps, lr_end=config.lr_end)
    return optimizer, scheduler


total_steps = len(train_loader) * config.max_epoches // config.gradient_accumulation_steps
optimizer, scheduler = build_optimizer_and_scheduler(model, config, total_steps)
loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')

def multi_label_crossentropy(y_true, y_pred):
    y_pred = (1 - 2 * y_true) * y_pred
    y_neg = y_pred - y_true * 1e12
    y_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_neg = torch.concat([y_neg, zeros], dim=-1)
    y_pos = torch.concat([y_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pos, dim=-1)
    return neg_loss + pos_loss

def calculate_loss(y_true, logits, mask):
    y_true = rearrange(y_true, 'b h m n -> (b h) (m n)')
    logits = rearrange(logits, 'b h m n -> (b h) (m n)')
    loss = multi_label_crossentropy(y_true, logits)
    return torch.mean(loss)

def global_pointer_f1_score_v1(y_true, y_pred):
    """给GlobalPointer设计的F1
    """
    y_pred = torch.greater(y_pred, 0).float()
    return torch.sum(y_true * y_pred), torch.sum(y_true),  torch.sum(y_pred)

def global_pointer_f1_score_v2(y_true, y_pred):
    """给GlobalPointer设计的F1
    """
    y_pred = torch.greater(y_pred, 0).float()
    tp = torch.sum(y_true * y_pred)
    fp = torch.sum((1-y_true) * y_pred)
    fn = torch.sum(y_true * (1-y_pred))
    tn = torch.sum((1-y_true) * (1-y_pred))
    return tp, fp, tn, fn

def global_pointer_f1_score(y_true, y_pred):
    """给GlobalPointer设计的F1
    """
    y_pred = torch.greater(y_pred, 0).float()
    return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred).clamp(min=1e-9)

def train():
    if config.continue_train:
        model.load_state_dict(torch.load(config.model_save_dir))
    best_f1 = 0.
    best_valid, best_test = (0, 0, 0), (0, 0, 0)
    for _ in range(config.max_epoches):
        model.cuda()
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        total_loss, total_f1 = 0, 0
        for batch_ind, batch in pbar:
            texts = batch[0]
            batch = [x.cuda() for x in batch[1:]]
            logits = model(*batch[:3])
            # logits = model(batch[0], batch[2], batch[1])
            loss = calculate_loss(batch[3], logits, batch[2])
            f1 = global_pointer_f1_score(batch[3], logits)

            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()
            if (batch_ind + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()  # update parameters of net
                scheduler.step()  # update learning rate schedule
                optimizer.zero_grad()  # reset gradients

            total_f1 += f1.item()
            total_loss += loss.item()
            avg_loss = total_loss / (batch_ind + 1)
            avg_f1 = total_f1 / (batch_ind + 1)
            pbar.set_description(f"{_ + 1}/{config.max_epoches} Epochs")
            pbar.set_postfix(loss=avg_loss, f1=avg_f1, lr=optimizer.param_groups[0]['lr'])

            if _ >= config.valid_start_epoch:
                if (batch_ind % np.ceil(len(train_loader) * config.valid_portion) == 0 and batch_ind != 0) or batch_ind == len(train_loader)-1:
                    model.eval()
                    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
                    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
                    with torch.no_grad():
                        for batch_ind, batch in pbar:
                            texts = batch[0]
                            batch = [x.cuda() for x in batch[1:]]
                            logits = model(*batch[:3])
                            tp, fp, tn, fn = global_pointer_f1_score_v2(batch[3], logits)
                            total_tp += tp
                            total_fp += fp
                            total_tn += tn
                            total_fn += fn
                            precision = total_tp / (total_tp + total_fp + 1e-10)
                            recall = total_tp / (total_tp + total_fn + 1e-10)
                            f1 = 2 * precision * recall / (precision + recall + 1e-10)
                            avg_f1 = f1.item()
                            pbar.set_postfix(p=precision.item(), r=recall.item(), f1=avg_f1)

                    if avg_f1 > best_f1:
                        if config.save_model:
                            torch.save(model.state_dict(), config.model_save_dir)
                        precision, recall, avg_f1 = np.round(precision.item(), 4), np.round(recall.item(), 4), np.round(avg_f1, 4)
                        print(f'best model saved with p & r & f1 {precision} & {recall} & {avg_f1}')
                        best_f1 = avg_f1

                    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
                    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
                    with torch.no_grad():
                        for batch_ind, batch in pbar:
                            texts = batch[0]
                            batch = [x.cuda() for x in batch[1:]]
                            logits = model(*batch[:3])
                            tp, fp, tn, fn = global_pointer_f1_score_v2(batch[3], logits)
                            total_tp += tp
                            total_fp += fp
                            total_tn += tn
                            total_fn += fn
                            precision = total_tp / (total_tp + total_fp + 1e-10)
                            recall = total_tp / (total_tp + total_fn + 1e-10)
                            f1 = 2 * precision * recall / (precision + recall + 1e-10)
                            avg_f1 = f1.item()
                            pbar.set_postfix(p=precision.item(), r=recall.item(), f1=avg_f1)

                    if avg_f1 > best_test[2]:
                        precision, recall, avg_f1 = np.round(precision.item(), 4), np.round(recall.item(), 4), np.round(
                            avg_f1, 4)
                        best_test = (precision, recall, avg_f1)
                    print(f'current test p & r & f1 {precision} & {recall} & {avg_f1} {str(best_test)}')
                    model.train()


class NamedEntityRecognizer(object):
    """命名实体识别器
    """
    def recognize(self, d, model, threshold=0):
        d['query'] = ''.join(d['query'].split())
        d['context'] = ''.join(d['context'].split())
        outputs = tokenizer(d['query'], d['context'], return_offsets_mapping=True, truncation=True,
                            max_length=maxlen)
        tokens = tokenizer.convert_ids_to_tokens(outputs['input_ids'])
        start_map, end_map = get_map(tokens, outputs['offset_mapping'], outputs['token_type_ids'])
        start_map_inv, end_map_inv = {y:x for x,y in start_map.items()}, {y:x for x,y in end_map.items()}
        token_ids = outputs['input_ids']
        segment_ids = outputs['token_type_ids']
        mask = outputs['attention_mask']
        token_ids = torch.LongTensor(token_ids).cuda()[None, ...]
        segment_ids = torch.LongTensor(segment_ids).cuda()[None, ...]
        mask = torch.LongTensor(mask).cuda()[None, ...]
        scores = model(token_ids, mask=mask, token_type_ids=segment_ids)[0]
        scores[:, [0, -1]] -= np.inf
        scores[:, :, [0, -1]] -= np.inf
        entities = []
        for l, start, end in zip(*torch.where(scores > threshold)):
            entities.append(
                (start_map_inv[start.item()], end_map_inv[end.item()], categories[l.item()])
            )
        return entities


NER = NamedEntityRecognizer()

def evaluate(*args):
    model.load_state_dict(torch.load(config.model_save_dir))
    model.cuda()
    model.eval()
    pbar = tqdm(enumerate(test_loader))
    total_loss, total_correct, total_true, total_pred = 0, 0, 0, 0
    total_f1 = 0
    with torch.no_grad():
        for batch_ind, batch in pbar:
            texts = batch[0]
            batch = [x.cuda() for x in batch[1:]]
            logits = model(*batch[:3])
            f1 = global_pointer_f1_score(batch[3], logits)
            total_f1 += f1
            avg_f1 = total_f1 / (batch_ind + 1)
            pbar.set_postfix(f1=avg_f1)


def evaluate_v1(*args):
    model.load_state_dict(torch.load(config.model_save_dir))
    model.cuda()
    model.eval()
    pbar = tqdm(enumerate(test_loader))
    total_loss, total_correct, total_true, total_pred = 0, 0, 0, 0
    total_f1 = 0
    with torch.no_grad():
        for batch_ind, batch in pbar:
            texts = batch[0]
            batch = [x.cuda() for x in batch[1:]]
            logits = model(*batch[:3])
            correct, true, pred = global_pointer_f1_score_v1(batch[3], logits)
            total_correct += correct.item()
            total_true += true.item()
            total_pred += pred.item()
            avg_f1 = 2*total_correct/(total_true+total_pred)
            pbar.set_postfix(f1=avg_f1)
    print(f'test f1: {avg_f1}')


def evaluate_v2(*args):
    model.load_state_dict(torch.load(config.model_save_dir))
    model.cuda()
    model.eval()
    pbar = tqdm(enumerate(test_loader))
    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
    with torch.no_grad():
        for batch_ind, batch in pbar:
            texts = batch[0]
            batch = [x.cuda() for x in batch[1:]]
            logits = model(*batch[:3])
            tp, fp, tn, fn = global_pointer_f1_score_v2(batch[3], logits)
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn
            precision = total_tp / (total_tp + total_fp + 1e-10)
            recall = total_tp / (total_tp + total_fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            f1 = f1.item()
            pbar.set_postfix(f1=f1)
    print(f'test f1: {f1}')


def evaluate_entity(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    model.load_state_dict(torch.load('best_model.pt'))
    model.cuda()
    model.eval()
    pbar = tqdm(enumerate(data), ncols=100)
    for i, d in pbar:
        R = set(NER.recognize(d, model))
        true_labels = []
        for start, end in zip(d['start_position'], d['end_position']):
            true_labels.append([start, end, d['entity_label']])
        T = set([tuple(i) for i in true_labels])
        if i == 664:
            print(i)
        X += len(R & T)
        Y += len(R)
        Z += len(T)

        pbar.set_description('f1 %f' % (2 * X / (Y + Z)))
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gau')
    parser.add_argument('-m',
                        type=str,
                        default='train',
                        help='train or eval')
    parser.add_argument('-r',
                        type=str,
                        default='all',
                        help='part or all')
    parser.add_argument('--eval_num',
                        type=int,
                        default=10,
                        help='eval number')
    parser.add_argument('--device_id',
                        type=int,
                        default=3,
                        help='device id')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
    print(config.__dict__)
    if args.m == 'train':
        train()
    else:
        print(evaluate_v2(test_data))