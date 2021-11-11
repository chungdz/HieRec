import pandas as pd
from tqdm import tqdm
import os
import re
import json

def parse_ent_list(x):
    if x.strip() == "":
        return ''

    return [k["WikidataId"] for k in json.loads(x)]

punctuation = '!,;:?"\''
def removePunctuation(text):
    text = re.sub(r'[{}]+'.format(punctuation),'',text)
    return text.strip().lower()

def parse_his(user_dict, his_beh, desc, user_idx):
    
    for uid, hist in tqdm(his_beh[['uid', 'hist']].values, total=his_beh.shape[0], desc=desc):

        if uid not in user_dict:
            user_dict[uid] = {"hist": [], "idx": user_idx}
            user_idx += 1
        else:
            if len(user_dict[uid]['hist']) > 0:
                continue
        
        hist_list = str(hist).split(' ')
        if str(hist) == 'nan':
            continue
       
        user_dict[uid]["hist"] = hist_list
    
    return user_idx

data_path = 'adressa'
max_title_len = 30
max_entity_len = 5

print("Loading news info")
news_df = pd.read_csv(os.path.join(data_path, "news.tsv"), sep="\t", encoding="utf-8",
                        names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                        quoting=3)
all_news = news_df.drop_duplicates("newsid")

news_dict = {}
word_dict = {'<pad>': 0}
entity_dict = {'<pad>': 0}
category_dict = {'<pad>': 0, '<his>': 1}
subcategory_dict = {'<pad>': 0, '<his>': 1}

word_idx = 1
news_idx = 2
entity_idx = 1
category_idx = 2

for n, title, cate, title_ents in tqdm(all_news[['newsid', "title", "cate", "title_ents"]].values, total=all_news.shape[0], desc='parse news'):
    news_dict[n] = {}
    news_dict[n]['idx'] = news_idx
    news_idx += 1

    tarr = removePunctuation(title).split()
    wid_arr = []
    for t in tarr:
        if t not in word_dict:
            word_dict[t] = word_idx
            word_idx += 1
        wid_arr.append(word_dict[t])
    cur_len = len(wid_arr)
    if cur_len < max_title_len:
        for l in range(max_title_len - cur_len):
            wid_arr.append(word_dict['<pad>'])
    news_dict[n]['title'] = wid_arr[:max_title_len]
    
    entity_arr = parse_ent_list(title_ents)
    eidx_arr = set()
    for e in entity_arr:
        if e not in entity_dict:
            entity_dict[e] = entity_idx
            entity_idx += 1
        eidx_arr.add(entity_dict[e])
    eidx_n = list(eidx_arr)
    ent_len = len(eidx_n)
    if ent_len < max_entity_len:
        for l in range(max_entity_len - ent_len):
            eidx_n.append(entity_dict['<pad>'])
    news_dict[n]['entity'] = eidx_n[:max_entity_len]
    
    if cate not in category_dict:
        category_dict[cate] = category_idx
        category_idx += 1
    news_dict[n]['category'] = category_dict[cate]

## paddning news for impression
news_dict['<pad>']= {}
news_dict['<pad>']['idx'] = 0
tarr = removePunctuation("This is the title of the padding news").split()
wid_arr = []
for t in tarr:
    if t not in word_dict:
        word_dict[t] = word_idx
        word_idx += 1
    wid_arr.append(word_dict[t])
cur_len = len(wid_arr)
if cur_len < max_title_len:
    for l in range(max_title_len - cur_len):
        wid_arr.append(word_dict['<pad>'])
news_dict['<pad>']['title'] = wid_arr[:max_title_len]
news_dict['<pad>']['entity'] = [entity_dict['<pad>']] * max_entity_len
news_dict['<pad>']['category'] = category_dict['<pad>']

## paddning news for history
news_dict['<his>']= {}
news_dict['<his>']['idx'] = 1
news_dict['<his>']['title'] = [word_dict['<pad>']] * max_title_len
news_dict['<his>']['entity'] = [entity_dict['<pad>']] * max_entity_len
news_dict['<his>']['category'] = category_dict['<his>']

print('news_num', len(news_dict))
print('entity num', len(entity_dict))
print('category num', len(category_dict))
print('word num', len(word_dict))

user_dict = {}
user_idx = 0

f_his_beh = os.path.join(data_path, "train/behaviors.tsv")
his_beh = pd.read_csv(f_his_beh, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
user_idx = parse_his(user_dict, his_beh, 'parse train behaviors', user_idx)

f_his_beh = os.path.join(data_path, "dev/behaviors.tsv")
his_beh = pd.read_csv(f_his_beh, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
user_idx = parse_his(user_dict, his_beh, 'parse dev behaviors', user_idx)

f_his_beh = os.path.join(data_path, "test/behaviors.tsv")
his_beh = pd.read_csv(f_his_beh, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
user_idx = parse_his(user_dict, his_beh, 'parse test behaviors', user_idx)

cate_num_list = []
subcate_num_list = []
cate_news_list = []
subcate_news_list = []

for uid, uinfo in tqdm(user_dict.items(), total=len(user_dict), desc='parse user dict'):
    
    cate_dict = {}
    for h in uinfo['hist']:
        cate = news_dict[h]['category']
        idx = news_dict[h]['idx']
        if cate not in cate_dict:
            cate_dict[cate] = []
        cate_dict[cate].append(idx)
    
    uinfo['cate_dict'] = cate_dict
    
    cate_num_list.append(len(cate_dict))
    total_num = 0
    for ck, cd in cate_dict.items():
        total_num += len(cd)
    cate_news_list.append(total_num)

avg_cate_num = sum(cate_num_list) / len(cate_num_list)
avg_cate_news = sum(cate_news_list) / len(cate_news_list)
print('avg category per user', avg_cate_num)
print('avg news number per category', avg_cate_news)

print('save dicts')
json.dump(user_dict, open(os.path.join(data_path, "user.json"), 'w', encoding='utf-8'))
json.dump(news_dict, open(os.path.join(data_path, "news.json"), 'w', encoding='utf-8'))
json.dump(word_dict, open(os.path.join(data_path, "word.json"), 'w', encoding='utf-8'))
json.dump(entity_dict, open(os.path.join(data_path, "entity.json"), 'w', encoding='utf-8'))
json.dump(category_dict, open(os.path.join(data_path, "category.json"), 'w', encoding='utf-8'))



