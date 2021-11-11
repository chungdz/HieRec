import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import NewsEncoder, SelfAttend


class HieRec(nn.Module):
    def __init__(self, cfg):
        super(HieRec, self).__init__()

        self.news_encoder = NewsEncoder(cfg)
        self.cfg = cfg
        self.category_emb = nn.Embedding(cfg.cate_num, cfg.hidden_size)
        self.news_title_indexes = nn.Parameter(torch.LongTensor(self.cfg.news_title_emb), requires_grad=False)
        self.news_entity_indexes = nn.Parameter(torch.LongTensor(self.cfg.news_entity_emb), requires_grad=False)

        self.cate_attend = SelfAttend(cfg.hidden_size)
        self.user_attend = SelfAttend(cfg.hidden_size)

    def forward(self, data, start_point, test_mode=False):
        news_num = self.cfg.neg_count + 1
        if test_mode:
            news_num = 1
        
        news_id = data[:, :news_num]
        news_category_index = data[:, news_num: news_num * 2]
        news_coefficient_category = data[:, news_num * 2: news_num * 3] * self.cfg.lambda_t
        # add batch size offset to index to get final real index in 2-D matrix
        news_category_index = start_point * self.cfg.ucatgeory_number + news_category_index
        
        user_emb = data[:, news_num * 3:]
        user_categories = user_emb[:, :self.cfg.ucatgeory_number]
        user_categories = self.category_emb(user_categories)
        user_news_id = user_emb[:, self.cfg.ucatgeory_number:].reshape(-1, self.cfg.ucatgeory_number, self.cfg.ucate_news)

        target_news_title = self.news_title_indexes[news_id]
        target_news_entity = self.news_entity_indexes[news_id]
        target_news = self.news_encoder(target_news_title.reshape(-1, self.cfg.max_title_len), target_news_entity.reshape(-1, self.cfg.max_entity_len)).reshape(-1, news_num, self.cfg.hidden_size)

        user_news_title = self.news_title_indexes[user_news_id]
        user_news_entity = self.news_entity_indexes[user_news_id]
        user_news = self.news_encoder(user_news_title.reshape(-1, self.cfg.max_title_len), user_news_entity.reshape(-1, self.cfg.max_entity_len)).reshape(-1, self.cfg.ucatgeory_number, self.cfg.ucate_news, self.cfg.hidden_size)
        
        user_cate = self.cate_attend(user_news) + user_categories
        news_cate_target_repre = user_cate.reshape(-1, self.cfg.hidden_size)[news_category_index]

        user = self.user_attend(user_cate)
        user = user.repeat(1, news_num).view(-1, news_num, self.cfg.hidden_size)

        user_score = torch.sum(user * target_news, dim=-1) * (1 - news_coefficient_category)
        cate_score = torch.sum(news_cate_target_repre * target_news, dim=-1) * news_coefficient_category
        final_score = user_score + cate_score

        return final_score