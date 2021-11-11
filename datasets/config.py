import json
import pickle
import numpy as np
import os

class ModelConfig():
    def __init__(self, root):

        self.word_emb = np.load('{}/word_emb.npy'.format(root))
        self.word_num = len(self.word_emb)
        self.news_title_emb = np.load('{}/title_emb.npy'.format(root))
        self.news_entity_emb = np.load('{}/news_entity_emb.npy'.format(root))
        self.news_num = len(self.news_title_emb)

        self.category_dict = json.load(open(os.path.join(root, "category.json"), 'r', encoding='utf-8'))
        self.cate_num = len(self.category_dict)
        self.entity_dict = json.load(open(os.path.join(root, "entity.json"), 'r', encoding='utf-8'))
        self.entity_num = len(self.entity_dict)

        self.ucatgeory_number = 8
        self.ucate_news = 16
        self.max_title_len = 30
        self.max_entity_len = 5
        self.neg_count = 4
        self.word_dim = 300
        self.entity_dim = 100
        self.hidden_size = 400
        self.head_num = 5
        self.dropout = 0.2
        self.lambda_t = 0.15
        
        return None