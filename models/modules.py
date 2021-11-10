import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttend(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super(SelfAttend, self).__init__()

        self.h1 = nn.Sequential(
            nn.Linear(embedding_size, 200),
            nn.Tanh()
        )
        
        self.gate_layer = nn.Linear(200, 1)

    def forward(self, seqs, seq_masks=None):
        """
        :param seqs: shape [batch_size, seq_length, embedding_size]
        :param seq_lens: shape [batch_size, seq_length]
        :return: shape [batch_size, seq_length, embedding_size]
        """
        gates = self.gate_layer(self.h1(seqs)).squeeze(-1)
        if seq_masks is not None:
            gates = gates.masked_fill(seq_masks == 0, -1e9)
        p_attn = F.softmax(gates, dim=-1)
        p_attn = p_attn.unsqueeze(-1)
        h = seqs * p_attn
        output = torch.sum(h, dim=-2)
        return output

class TitleEncoder(nn.Module):
    def __init__(self, cfg):
        super(TitleEncoder, self).__init__()
        self.cfg = cfg
        self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(cfg.word_emb), freeze=False)

        self.mh_self_attn = nn.MultiheadAttention(
            cfg.hidden_size, num_heads=cfg.head_num
        )
        self.word_self_attend = SelfAttend(cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)
        

    def _extract_hidden_rep(self, seqs):
        """
        Encoding
        :param seqs: [*, seq_length]
        :param seq_lens: [*]
        :return: Tuple, (1) [*, seq_length, hidden_size] (2) [*, seq_length];
        """
        embs = self.word_embedding(seqs)
        X = self.dropout(embs)

        X = X.permute(1, 0, 2)
        output, _ = self.mh_self_attn(X, X, X)
        output = output.permute(1, 0, 2)
        output = self.dropout(output)
        X = X.permute(1, 0, 2)
        # output = self.word_proj(output)

        return self.word_layer_norm(output + X)

    def encode_news(self, seqs):
        """

        Args:
            seqs: [*, max_news_len]
            seq_lens: [*]

        Returns:
            [*, hidden_size]
        """
        hiddens = self._extract_hidden_rep(seqs)

        # [*, hidden_size]
        self_attend = self.word_self_attend(hiddens)

        return self_attend


class EntityEncoder(nn.Module):
    def __init__(self, cfg):
        super(EntityEncoder, self).__init__()
        self.cfg = cfg
        self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(cfg.entity_emb), freeze=False)

        self.mh_self_attn = nn.MultiheadAttention(
            cfg.entity_dim, num_heads=cfg.head_num
        )
        self.word_self_attend = SelfAttend(cfg.entity_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def _extract_hidden_rep(self, seqs):
        """
        Encoding
        :param seqs: [*, seq_length]
        :param seq_lens: [*]
        :return: Tuple, (1) [*, seq_length, hidden_size] (2) [*, seq_length];
        """
        embs = self.word_embedding(seqs)
        X = self.dropout(embs)

        X = X.permute(1, 0, 2)
        output, _ = self.mh_self_attn(X, X, X)
        output = output.permute(1, 0, 2)
        output = self.dropout(output)
        X = X.permute(1, 0, 2)
        # output = self.word_proj(output)

        return self.word_layer_norm(output + X)

    def encode_news(self, seqs):
        """

        Args:
            seqs: [*, max_news_len]
            seq_lens: [*]

        Returns:
            [*, hidden_size]
        """
        hiddens = self._extract_hidden_rep(seqs)

        # [*, hidden_size]
        self_attend = self.word_self_attend(hiddens)

        return self_attend

