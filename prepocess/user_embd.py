import json
import numpy as np
import os
from tqdm import tqdm

data_path = 'adressa'

cate_dict = json.load(open(os.path.join(data_path, "category.json"), 'r', encoding='utf-8'))
user_dict = json.load(open(os.path.join(data_path, "user.json"), 'r', encoding='utf-8'))
news_dict = json.load(open(os.path.join(data_path, "news.json"), 'r', encoding='utf-8'))

user_emb_path = os.path.join(data_path, "user_emb.npy")

catgeory_number = 8
cate_news = 16
user_embedding_length = catgeory_number + catgeory_number * cate_news

user_embed = np.zeros((len(user_dict), user_embedding_length))
for uid, uinfo in tqdm(user_dict.items(), total=len(user_dict), desc='parse user embd'):
    
    ucate_dict = uinfo['cate_dict']

    cur_line = []
    
    sorted_cate_len = [t[0] for t in sorted(ucate_dict, key=lambda x: len(ucate_dict[x]), reverse=True)]
    if len(sorted_cate_len) < catgeory_number:
        sorted_cate_len += [cate_dict['<pad>']] * (catgeory_number - len(sorted_cate_len))
    sorted_cate_len = sorted_cate_len[:catgeory_number]
    # insert category index 
    cur_line += sorted_cate_len
    # insert news index
    for cur_cate_index in range(catgeory_number):
        
        cur_cate = cur_line[cur_cate_index]
        if cur_cate == cate_dict['<pad>']:
            cur_line += [news_dict['<his>']['idx']] * cate_news
        else:
            news_list = ucate_dict[cur_cate]
            if len(news_list) < cate_news:
                news_list += [news_dict['<his>']['idx']] * (cate_news - len(news_list))
            cur_line += news_list[:cate_news]
    
    user_embed[uinfo['idx']] = np.array(cur_line)

print('user embedding shape', user_embed.shape)
np.save(user_emb_path, user_embed)