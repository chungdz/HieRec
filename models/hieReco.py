import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import EntityEncoder, TitleEncoder, SelfAttend


class HieRec(nn.Module):
    def __init__(self, cfg):
        super(HieRec, self).__init__()

        self.title_encoder = TitleEncoder(cfg)
        self.entity_encoder = EntityEncoder(cfg)
        self.cfg = cfg
        self.category_emb = nn.Embedding(cfg.cate_num, cfg.hidden_size)
        self.subcategory_emb = nn.Embedding(cfg.subcate_num, cfg.hidden_size)
        self.news_title_indexes = nn.Parameter(torch.LongTensor(self.cfg.news_title_emb), requires_grad=False)
        self.news_entity_indexes = nn.Parameter(torch.LongTensor(self.cfg.news_entity_emb), requires_grad=False)


    def forward(self, data, test_mode=False):
        news_num = self.cfg.neg_count + 1
        if test_mode:
            news_num = 1
        
        batch_size = data.size(0)
        start_point = torch.arange(batch_size)
        start_point = start_point.unsqueeze(-1)
        
        news_id = data[:, :news_num]
        news_category_index = data[:, news_num: news_num * 2]
        news_subcategory_index = data[:, news_num * 2: news_num * 3]
        news_coefficient_category = data[:, news_num * 3: news_num * 4]
        # add batch size offset to index to get final real index in 2-D matrix
        news_category_index = start_point * self.cfg.ucatgeory_number + news_category_index
        news_subcategory_index = start_point * self.cfg.ucatgeory_number * self.cfg.usubcate_number + news_subcategory_index
        
        user_emb = data[:, news_num * 5:]
        user_categories = user_emb[:, :self.cfg.ucatgeory_number]
        user_subcategories = user_emb[:, self.cfg.ucatgeory_number: self.cfg.ucatgeory_number + self.cfg.ucatgeory_number * self.cfg.usubcate_number].reshape(-1, self.cfg.ucatgeory_number, self.cfg.usubcate_number)
        user_news_id = user_emb[:, self.cfg.ucatgeory_number + self.cfg.ucatgeory_number * self.cfg.usubcate_number:].reshape(-1, self.cfg.ucatgeory_number, self.cfg.usubcate_number, self.cfg.usubcate_news)

        target_news_title = self.news_title_indexes[news_id]
        target_news_entity = self.news_entity_indexes[news_id]

        target_news_title = self.title_encoder(target_news_title.reshape(-1, self.cfg.max_title_len)).reshape(-1, news_num, self.cfg.word_dim)
        target_news_entity = self.entity_encoder(target_news_entity.reshape(-1, self.cfg.max_entity_len)).reshape(-1, news_num, self.cfg.entity_dim)
        print(target_news_title.size(), target_news_entity.size())

        user_news_title = self.news_title_indexes[user_news_id]
        user_news_entity = self.news_entity_indexes[user_news_id]

        user_news_title = self.title_encoder(user_news_title.reshape(-1, self.cfg.max_title_len)).reshape(-1, self.cfg.ucatgeory_number, self.cfg.usubcate_number, self.cfg.word_dim)
        user_news_entity = self.entity_encoder(user_news_entity.reshape(-1, self.cfg.max_entity_len)).reshape(-1, self.cfg.ucatgeory_number, self.cfg.usubcate_number, self.cfg.entity_dim)
        print(user_news_title.size(), user_news_entity.size())

        return torch.randn(batch_size, news_num)