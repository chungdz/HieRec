import numpy as np
import json
from tqdm import tqdm
import os

def build_entity_dict(emb_dict, lines, desc):
    error_line = 0
    for line in tqdm(lines, total=len(lines), desc=desc):
        row = line.strip().split()
        try:
            embedding = [float(w) for w in row[1: entity_embed_size + 1]]
            emb_dict[row[0]] = np.array(embedding)
        except:
            error_line += 1
    print("Error lines: {}".format(error_line))

# settings
word_embed_size = 300
entity_embed_size = 100

data_path = 'adressa'

print('build word embeddings')
word_dict = json.load(open(os.path.join(data_path, "word.json"), 'r', encoding='utf-8'))
glove_path = os.path.join(data_path, "glove.840B.300d.txt")
wemb_path = os.path.join(data_path, "word_emb.npy")

lines = open(glove_path, "r", encoding="utf8").readlines()
emb_dict = dict()
error_line = 0
for line in tqdm(lines, total=len(lines), desc='load Glove'):
    row = line.strip().split()
    try:
        embedding = [float(w) for w in row[1:]]
        emb_dict[row[0]] = np.array(embedding)
    except:
        error_line += 1
print("Error lines: {}".format(error_line))

weights_matrix = np.zeros((len(word_dict), word_embed_size))
words_found = 0

for k, v in tqdm(word_dict.items(), total=len(word_dict), desc='generate word embeddings'):
    try:
        weights_matrix[v] = emb_dict[k]
        words_found += 1
    except KeyError:
        weights_matrix[v] = np.random.normal(size=(word_embed_size,))
print("Totally find {} words in pre-trained embeddings.".format(words_found))
np.save(wemb_path, weights_matrix)
print('word embedding shape:', weights_matrix.shape)

print('build news title embeddings')
news_dict = json.load(open(os.path.join(data_path, "news.json"), 'r', encoding='utf-8'))
title_emb_path = os.path.join(data_path, "title_emb.npy")

title_len = len(news_dict['<pad>']['title'])
title_matrix = np.zeros((len(news_dict), title_len), dtype=int)

for k, v in news_dict.items():
    title_matrix[v['idx']] = np.array(v['title'])

print('news embedding shape:', title_matrix.shape)
np.save(title_emb_path, title_matrix)

print('build news entity embeddings')
news_entity_emb_path = os.path.join(data_path, "news_entity_emb.npy")

entity_len = len(news_dict['<pad>']['entity'])
news_entity_matrix = np.zeros((len(news_dict), entity_len), dtype=int)

for k, v in news_dict.items():
    news_entity_matrix[v['idx']] = np.array(v['entity'])

print('news entity embedding shape', news_entity_matrix.shape)
np.save(news_entity_emb_path, news_entity_matrix)
