import json
import numpy as np
import os
from tqdm import tqdm

data_path = 'data'

cate_dict = json.load(open(os.path.join(data_path, "category.json"), 'r', encoding='utf-8'))
subcate_dict = json.load(open(os.path.join(data_path, "subcategory.json"), 'r', encoding='utf-8'))
user_dict = json.load(open(os.path.join(data_path, "user.json"), 'r', encoding='utf-8'))
news_dict = json.load(open(os.path.join(data_path, "news.json"), 'r', encoding='utf-8'))

user_emb_path = os.path.join(data_path, "user_emb.npy")

catgeory_number = 8
subcate_number = 4
subcate_news = 4
user_embedding_length = catgeory_number + catgeory_number * subcate_number + catgeory_number * subcate_number * subcate_news

user_embed = np.zeros((len(user_dict), user_embedding_length))
for uid, uinfo in tqdm(user_dict.items(), total=len(user_dict), desc='parse user embd'):
    
    ucate_dict = uinfo['cate_dict']
    cate_len = []
    for ck, cd in ucate_dict.items():
        total_num = 0
        for sbk, sbl in cd.items():
            total_num += len(sbl)
        cate_len.append((ck, total_num))
    
    cur_line = []
    
    sorted_cate_len = [t[0] for t in sorted(cate_len, key=lambda x: x[1], reverse=True)]
    if len(sorted_cate_len) < catgeory_number:
        sorted_cate_len += [cate_dict['<his>']] * (catgeory_number - len(sorted_cate_len))
    sorted_cate_len = sorted_cate_len[:catgeory_number]
    # insert category index 
    cur_line += sorted_cate_len
    assert(len(cur_line) == catgeory_number)
    # insert subcategory index
    for cindex in sorted_cate_len:
        if cindex == cate_dict['<his>']:
            cur_line += [subcate_dict['<his>']] * subcate_number
            continue
        
        sorted_subcate_len = sorted(ucate_dict[cindex], key=lambda x: len(ucate_dict[cindex][x]), reverse=True)
        if len(sorted_subcate_len) < subcate_number:
            sorted_subcate_len += [subcate_dict['<his>']] * (subcate_number - len(sorted_subcate_len))
        cur_line += sorted_subcate_len[:subcate_number]
    assert(len(cur_line) == catgeory_number + catgeory_number * subcate_number)
    # insert news index
    cur_cate_index = 0
    cur_subcate_index = catgeory_number
    subcate_cnt = 0
    while cur_cate_index < catgeory_number:
        
        cur_cate = cur_line[cur_cate_index]
        cur_subcate = cur_line[cur_subcate_index]
        if cur_subcate == subcate_dict['<his>']:
            cur_line += [news_dict['<his>']['idx']] * subcate_news
        else:
            news_list = ucate_dict[cur_cate][cur_subcate]
            if len(news_list) < subcate_news:
                news_list += [news_dict['<his>']['idx']] * (subcate_news - len(news_list))
            cur_line += news_list[:subcate_news]

        cur_subcate_index += 1
        subcate_cnt += 1
        if subcate_cnt >= subcate_number:
            cur_cate_index += 1
            subcate_cnt = 0
    
    user_embed[uinfo['idx']] = np.array(cur_line)

print('user embedding shape', user_embed.shape)
np.save(user_emb_path, user_embed)