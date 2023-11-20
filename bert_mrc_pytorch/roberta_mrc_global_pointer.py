from transformers import RobertaPreTrainedModel, RobertaModel
from torch import nn
from einops import einsum, rearrange, reduce
from rotary_embedding import RotaryEmbedding
import torch


def exists(val):
    return val is not None

class RobertGlobalPointer(RobertaPreTrainedModel):
    def __init__(self, config, head, head_size, **kwargs):
        super(RobertGlobalPointer, self).__init__(config)
        self.roberta = RobertaModel(config)

        config.head_size = head_size
        config.head = head
        self.config = config
        self.head_size = config.head_size
        self.rotary_emb = RotaryEmbedding(config.head_size)
        self.pos = nn.Linear(config.hidden_size, config.head * config.head_size * 2)
        self.start_pos = nn.Linear(config.hidden_size, config.head * config.head_size)
        self.end_pos = nn.Linear(config.hidden_size, config.head * config.head_size)

    def forward(self, x, mask=None, token_type_ids=None):
        seqlen, device = x.shape[-1], x.device
        # batch, seq, dim
        x = self.roberta(x, attention_mask=mask).last_hidden_state

        # start, end = torch.split(self.pos(x), self.config.head*self.config.head_size, dim=-1)
        start = rearrange(self.start_pos(x), 'b m (c h) -> b m c h', h=self.head_size)
        end = rearrange(self.end_pos(x), 'b m (c h) -> b m c h', h=self.head_size)
        start, end = map(self.rotary_emb, (start, end))

        # batch, seq, seq
        x = einsum(start, end, 'b m h d, b n h d -> b h m n')

        # pad sequence
        mask = rearrange(mask, 'b s -> b 1 1 s').bool()
        x = x.masked_fill(~mask, -1e12)
        mask = rearrange(mask, 'b 1 1 s -> b 1 s 1').bool()
        x = x.masked_fill(~mask, -1e12)

        # pad lower tril
        tril_mask = torch.ones((seqlen, seqlen), device=device).triu().bool()
        x = x.masked_fill(~tril_mask, -1e12)

        return x / self.head_size**0.5


class RobertEfficientGlobalPointer(RobertaPreTrainedModel):
    def __init__(self, config, head, head_size, **kwargs):
        super(RobertEfficientGlobalPointer, self).__init__(config)
        self.roberta = RobertaModel(config)

        config.head_size = head_size
        config.head = head
        self.config = config
        self.head_size = config.head_size
        self.rotary_emb = RotaryEmbedding(config.head_size)
        self.p_dense = nn.Linear(config.hidden_size, config.head_size * 2)
        self.q_dense = nn.Linear(config.hidden_size, config.head * 2)

    def forward(self, x, mask=None, token_type_ids=None):
        seqlen, device = x.shape[-1], x.device
        # batch, seq, dim
        inputs = self.roberta(x, attention_mask=mask).last_hidden_state

        x = self.p_dense(inputs)
        qw, kw = x[..., ::2], x[..., 1::2]
        qw, kw = map(self.rotary_emb, (qw, kw))

        # batch, seq, seq
        x = einsum(qw, kw, 'b m d, b n d -> b m n') / self.head_size**0.5
        bias = rearrange(self.q_dense(inputs), 'b n h->b h n') / 2
        x = x[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]

        # pad sequence
        mask = rearrange(mask, 'b s -> b 1 1 s').bool()
        x = x.masked_fill(~mask, -1e12)
        mask = rearrange(mask, 'b 1 1 s -> b 1 s 1').bool()
        x = x.masked_fill(~mask, -1e12)

        # pad lower tril
        tril_mask = torch.ones((seqlen, seqlen), device=device).triu().bool()
        x = x.masked_fill(~tril_mask, -1e12)

        return x


class RobertGlobalPointerWithLabel(RobertaPreTrainedModel):
    def __init__(self, config, head, head_size, label_embedding, **kwargs):
        super(RobertGlobalPointerWithLabel, self).__init__(config)
        self.roberta = RobertaModel(config)

        drop_rate = kwargs.get('dropout', 0.1)
        config.head_size = head_size
        config.head = head
        self.config = config
        self.head_size = config.head_size
        self.rotary_emb = RotaryEmbedding(config.head_size)
        # self.pos = nn.Linear(config.hidden_size, config.head * config.head_size * 2)
        self.start_pos = nn.Linear(config.hidden_size*2, config.head * config.head_size)
        self.end_pos = nn.Linear(config.hidden_size*2, config.head * config.head_size)

        self.label_embedding = nn.Parameter(torch.randn(head, 300))
        if exists(label_embedding):
            self.label_embedding.data = label_embedding
        self.label_embedding.requires_grad = True
        self.label_norm = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.LayerNorm(300)
        )

        self.label_linear = nn.Sequential(
            nn.Linear(300, config.hidden_size),
            nn.GELU()
        )
        self.drop_norm = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.LayerNorm(config.hidden_size)
        )

    def forward(self, x, mask=None, token_type_ids=None):
        batch, seqlen, device = x.shape[0], x.shape[-1], x.device
        # batch, seq, dim
        x = self.roberta(x, attention_mask=mask).last_hidden_state

        # 引入标签信息
        # label_embedding = self.label_embedding(torch.arange(0, self.config.head, dtype=torch.int64, device=device)).unsqueeze(0)
        label_embedding = self.label_norm(self.label_embedding.unsqueeze(0))
        label_embedding = self.label_linear(label_embedding)
        qk = einsum(x, label_embedding, 'b m n, b q n -> b m q') / label_embedding.shape[-1] ** 0.5
        qkv = einsum(qk, label_embedding, 'b m n, b n d -> b m d')
        qkv = self.drop_norm(qkv)
        x = torch.concat([x, qkv], dim=-1)

        start = self.start_pos(x)
        end = self.end_pos(x)

        start = rearrange(start, 'b m (c h) -> b m c h', h=self.head_size)
        end = rearrange(end, 'b m (c h) -> b m c h', h=self.head_size)
        start, end = map(self.rotary_emb, (start, end))

        # batch, seq, seq
        x = einsum(start, end, 'b m h d, b n h d -> b h m n')

        # pad sequence
        mask = rearrange(mask, 'b s -> b 1 1 s').bool()
        x = x.masked_fill(~mask, -1e12)
        mask = rearrange(mask, 'b 1 1 s -> b 1 s 1').bool()
        x = x.masked_fill(~mask, -1e12)

        # pad lower tril
        tril_mask = torch.ones((seqlen, seqlen), device=device).triu().bool()
        x = x.masked_fill(~tril_mask, -1e12)

        return x / self.head_size**0.5


class RobertEfficientGlobalPointerWithLabel(RobertaPreTrainedModel):
    def __init__(self, config, head, head_size, label_embedding, **kwargs):
        super(RobertEfficientGlobalPointerWithLabel, self).__init__(config)
        self.roberta = RobertaModel(config)

        drop_rate = kwargs.get('dropout', 0.1)
        config.head_size = head_size
        config.head = head
        self.config = config
        self.head_size = config.head_size
        self.rotary_emb = RotaryEmbedding(config.head_size)
        self.p_dense = nn.Linear(config.hidden_size*2, config.head_size * 2)
        self.q_dense = nn.Linear(config.hidden_size*2, config.head * 2)

        self.label_embedding = nn.Parameter(torch.randn(head, 300))
        if exists(label_embedding):
            self.label_embedding.data = label_embedding
        self.label_embedding.requires_grad = True
        self.label_norm = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.LayerNorm(300)
        )

        self.label_linear = nn.Sequential(
            nn.Linear(300, config.hidden_size),
            nn.GELU()
        )
        self.drop_norm = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.LayerNorm(config.hidden_size)
        )

    def forward(self, x, mask=None, token_type_ids=None):
        seqlen, device = x.shape[-1], x.device
        # batch, seq, dim
        inputs = self.roberta(x, attention_mask=mask).last_hidden_state

        # 引入标签信息
        # label_embedding = self.label_embedding(torch.arange(0, self.config.head, dtype=torch.int64, device=device)).unsqueeze(0)
        label_embedding = self.label_norm(self.label_embedding.unsqueeze(0))
        label_embedding = self.label_linear(label_embedding)
        qk = einsum(inputs, label_embedding, 'b m n, b q n -> b m q') / label_embedding.shape[-1] ** 0.5
        qkv = einsum(qk, label_embedding, 'b m n, b n d -> b m d')
        qkv = self.drop_norm(qkv)
        inputs = torch.concat([inputs, qkv], dim=-1)

        x = self.p_dense(inputs)
        qw, kw = x[..., ::2], x[..., 1::2]
        qw, kw = map(self.rotary_emb, (qw, kw))

        # batch, seq, seq
        x = einsum(qw, kw, 'b m d, b n d -> b m n') / self.head_size**0.5
        bias = rearrange(self.q_dense(inputs), 'b n h->b h n') / 2
        x = x[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]

        # pad sequence
        mask = rearrange(mask, 'b s -> b 1 1 s').bool()
        x = x.masked_fill(~mask, -1e12)
        mask = rearrange(mask, 'b 1 1 s -> b 1 s 1').bool()
        x = x.masked_fill(~mask, -1e12)

        # pad lower tril
        tril_mask = torch.ones((seqlen, seqlen), device=device).triu().bool()
        x = x.masked_fill(~tril_mask, -1e12)

        return x


class EmbeddingLayer(torch.nn.Module):
    """Create embedding from glove.6b for LSTM module"""
    def __init__(self, config):
        super().__init__()
        self.embedding = torch.nn.Embedding(21128, 300, padding_idx=1)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(300, config.hidden_size // 2, bidirectional=True, batch_first=True, dropout=0.5)

    def __call__(self, vocab_id_list):
        embedding = self.embedding(vocab_id_list)
        return self.lstm(self.dropout(embedding))


class LSTMGlobalPointer(RobertaPreTrainedModel):
    def __init__(self, config, head, head_size, **kwargs):
        super(LSTMGlobalPointer, self).__init__(config)
        self.lstm = EmbeddingLayer(config)

        config.head_size = head_size
        config.head = head
        self.config = config
        self.head_size = config.head_size
        self.rotary_emb = RotaryEmbedding(config.head_size)
        self.pos = nn.Linear(config.hidden_size, config.head * config.head_size * 2)
        self.start_pos = nn.Linear(config.hidden_size, config.head * config.head_size)
        self.end_pos = nn.Linear(config.hidden_size, config.head * config.head_size)

    def forward(self, x, mask=None, token_type_ids=None):
        seqlen, device = x.shape[-1], x.device
        # batch, seq, dim
        x, (_, _) = self.lstm(x)

        # start, end = torch.split(self.pos(x), self.config.head*self.config.head_size, dim=-1)
        start = rearrange(self.start_pos(x), 'b m (c h) -> b m c h', h=self.head_size)
        end = rearrange(self.end_pos(x), 'b m (c h) -> b m c h', h=self.head_size)
        start, end = map(self.rotary_emb, (start, end))

        # batch, seq, seq
        x = einsum(start, end, 'b m h d, b n h d -> b h m n')

        # pad sequence
        mask = rearrange(mask, 'b s -> b 1 1 s').bool()
        x = x.masked_fill(~mask, -1e12)
        mask = rearrange(mask, 'b 1 1 s -> b 1 s 1').bool()
        x = x.masked_fill(~mask, -1e12)

        # pad lower tril
        tril_mask = torch.ones((seqlen, seqlen), device=device).triu().bool()
        x = x.masked_fill(~tril_mask, -1e12)

        return x / self.head_size**0.5


class LSTMGlobalPointerWithLabel(RobertaPreTrainedModel):
    def __init__(self, config, head, head_size, label_embedding, **kwargs):
        super(LSTMGlobalPointerWithLabel, self).__init__(config)

        drop_rate = kwargs.get('dropout', 0.1)
        config.head_size = head_size
        config.head = head
        self.lstm = EmbeddingLayer(config)
        self.config = config
        self.head_size = config.head_size
        self.rotary_emb = RotaryEmbedding(config.head_size)
        # self.pos = nn.Linear(config.hidden_size, config.head * config.head_size * 2)
        self.start_pos = nn.Linear(config.hidden_size*2, config.head * config.head_size)
        self.end_pos = nn.Linear(config.hidden_size*2, config.head * config.head_size)

        self.label_embedding = nn.Parameter(torch.randn(head, 300))
        if exists(label_embedding):
            self.label_embedding.data = label_embedding
        self.label_embedding.requires_grad = True
        self.label_norm = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.LayerNorm(300)
        )

        self.label_linear = nn.Sequential(
            nn.Linear(300, config.hidden_size),
            nn.GELU()
        )
        self.drop_norm = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.LayerNorm(config.hidden_size)
        )

    def forward(self, x, mask=None, token_type_ids=None):
        batch, seqlen, device = x.shape[0], x.shape[-1], x.device
        # batch, seq, dim
        x, (_, _) = self.lstm(x)

        # 引入标签信息
        # label_embedding = self.label_embedding(torch.arange(0, self.config.head, dtype=torch.int64, device=device)).unsqueeze(0)
        label_embedding = self.label_norm(self.label_embedding.unsqueeze(0))
        label_embedding = self.label_linear(label_embedding)
        qk = einsum(x, label_embedding, 'b m n, b q n -> b m q') / label_embedding.shape[-1] ** 0.5
        qkv = einsum(qk, label_embedding, 'b m n, b n d -> b m d')
        qkv = self.drop_norm(qkv)
        x = torch.concat([x, qkv], dim=-1)

        start = self.start_pos(x)
        end = self.end_pos(x)

        start = rearrange(start, 'b m (c h) -> b m c h', h=self.head_size)
        end = rearrange(end, 'b m (c h) -> b m c h', h=self.head_size)
        start, end = map(self.rotary_emb, (start, end))

        # batch, seq, seq
        x = einsum(start, end, 'b m h d, b n h d -> b h m n')

        # pad sequence
        mask = rearrange(mask, 'b s -> b 1 1 s').bool()
        x = x.masked_fill(~mask, -1e12)
        mask = rearrange(mask, 'b 1 1 s -> b 1 s 1').bool()
        x = x.masked_fill(~mask, -1e12)

        # pad lower tril
        tril_mask = torch.ones((seqlen, seqlen), device=device).triu().bool()
        x = x.masked_fill(~tril_mask, -1e12)

        return x / self.head_size**0.5