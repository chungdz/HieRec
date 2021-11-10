import os
import argparse
from tqdm import tqdm
import json
import scipy.stats as ss
import numpy as np
import pandas as pd
import math
import torch
from utils.log_util import NEPTUNE_API_TOKEN
from utils.log_util import convert_omegaconf_to_dict
from utils.train_util import set_seed
from utils.train_util import save_checkpoint_by_epoch
from utils.eval_util import group_labels
from utils.eval_util import cal_metric



def gather(output_path, filenum, start_file=0, validate=False, save=True):
    preds = []
    labels = []
    imp_indexes = []

    for i in range(start_file, filenum):
        with open(output_path + 'tmp_{}.json'.format(i), 'r', encoding='utf-8') as f:
            cur_result = json.load(f)
        imp_indexes += cur_result['imp']
        labels += cur_result['labels']
        preds += cur_result['preds']

    print(len(preds))
    all_keys = list(set(imp_indexes))
    group_labels = {k: [] for k in all_keys}
    group_preds = {k: [] for k in all_keys}

    for l, p, k in zip(labels, preds, imp_indexes):
        group_labels[k].append(l)
        group_preds[k].append(p)
    
    if validate:
        all_labels = []
        all_preds = []
        for k in all_keys:
            all_labels.append(group_labels[k])
            all_preds.append(group_preds[k])
        
        metric_list = [x.strip() for x in "group_auc || mean_mrr || ndcg@5;10".split("||")]
        ret = cal_metric(all_labels, all_preds, metric_list)
        for metric, val in ret.items():
            print("Epoch: {}, {}: {}".format(1, metric, val))

    if save:
        final_arr = []
        for k in group_preds.keys():
            new_row = []
            new_row.append(k)
            new_row.append(','.join(list(map(str, np.array(group_labels[k]).astype(int)))))
            new_row.append(','.join(list(map(str, np.array(group_preds[k]).astype(float)))))
            
            rank = ss.rankdata(-np.array(group_preds[k])).astype(int).tolist()
            new_row.append('[' + ','.join(list(map(str, rank))) + ']')
            
            assert(len(rank) == len(group_labels[k]))
            
            final_arr.append(new_row)
        
        fdf = pd.DataFrame(final_arr, columns=['impression', 'labels', 'preds', 'ranks'])
        fdf.drop(columns=['labels', 'ranks']).to_csv(output_path + 'score.txt', sep=' ', index=False)
        fdf.drop(columns=['labels', 'preds']).to_csv(output_path + 'result.txt', header=None, sep=' ', index=False)