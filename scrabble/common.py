import json
import os
import argparse
import random
from functools import reduce, partial
import logging
import re
from collections import defaultdict, OrderedDict
import pdb
import sys
import requests
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as hier

from .data_model import *

POINT_POSTFIXES = ['sensor', 'setpoint', 'command', 'alarm', 'status', 'meter']

#SCRABBLE_METADATA_DIR = str(os.environ['SCRABBLE_METADATA_DIR'])
SCRABBLE_METADATA_DIR = os.path.dirname(os.path.realpath(__file__)) + \
    '/../metadata'

def elem2list(elem):
    if isinstance(elem, str):
        return elem.split('_')
    else:
        return []


def csv2json(df, key_idx, value_idx):
    keys = df[key_idx].tolist()
    values = df[value_idx].tolist()
    return {k: elem2list(v) for k, v in zip(keys, values)}

def sub_dict_by_key_set(d, ks):
    return dict((k,v) for k, v in d.items() if k in ks)

def leave_one_word(s, w):
    if w in s:
        s = s.replace(w, '')
        s = w + '-' + s
    return s

def find_keys(tv, d, crit=lambda x,y:x==y):
    keys = list()
    for k, v in d.items():
        if crit(tv, v):
            keys.append(k)
    return keys

def check_in(x,y):
    return x in y

def joiner(s):
    return ''.join(s)

def get_word_clusters(sentence_dict):
    srcids = list(sentence_dict.keys())
    sentences = []
    for srcid in srcids:
        sentence = []
        for metadata_type, sent in sentence_dict[srcid].items():
            sentence.append(''.join(sent))
        sentence = '\n'.join(sentence)
        sentence = ' '.join(re.findall('[a-z]+', sentence))
        sentences.append(sentence)
    vect = TfidfVectorizer()
    #vect = CountVectorizer()
    bow = vect.fit_transform(sentences).toarray()
    z = linkage(bow, metric='cityblock', method='complete')
    dists = list(set(z[:,2]))
    thresh = (dists[1] + dists[2]) /2
    print("Threshold: ", thresh)
    b = hier.fcluster(z,thresh, criterion='distance')
    cluster_dict = defaultdict(list)

    for srcid, cluster_id in zip(srcids, b):
        cluster_dict[cluster_id].append(srcid)
    return dict(cluster_dict)

def select_random_samples(building,
                          srcids,
                          n,
                          use_cluster_flag,
                          sentence_dict=None,
                          token_type='justseparate',
                          reverse=True,
                          cluster_dict=None,
                          shuffle_flag=True,
                          unique_clusters_flag=False,
                         ):
    #if not cluster_dict:
    #    cluster_filename = 'model/%s_word_clustering_%s.json' % (building, token_type)
    #    with open(cluster_filename, 'r') as fp:
    #        cluster_dict = json.load(fp)
    assert sentence_dict or cluster_dict
    if not cluster_dict:
        cluster_dict = get_word_clusters(sentence_dict)

    # Learning Sample Selection
    sample_srcids = set()
    length_counter = lambda x: len(x[1])
    ander = lambda x, y: x and y
    if use_cluster_flag:
        sample_cnt = 0
        sorted_cluster_dict = OrderedDict(
            sorted(cluster_dict.items(), key=length_counter, reverse=reverse))
        #n = len(sorted_cluster_dict) #TODO: Remove if not working well
        while len(sample_srcids) < n:
            cluster_dict_items = list(sorted_cluster_dict.items())
            if shuffle_flag:
                random.shuffle(cluster_dict_items)
            for cluster_num, srcid_list in cluster_dict_items:
                valid_srcid_list = set(srcid_list)\
                        .intersection(set(srcids))\
                        .difference(set(sample_srcids))
                if len(valid_srcid_list) > 0:
                    sample_srcids.add(\
                            random.choice(list(valid_srcid_list)))
                if len(sample_srcids) >= n:
                    break
            if unique_clusters_flag:
                break
    else:
        sample_srcids = random.sample(srcids, n)
    return list(sample_srcids)

def splitter(s):
    return s.split('_')

def alpha_tokenizer(s): 
    return re.findall('[a-zA-Z]+', s)

def adder(x, y):
    return x+y

def bilou_tagset_phraser(sentence, token_labels, keep_alltokens=False):
    phrase_labels = list()
    curr_phrase = ''
    for i, (c, label) in enumerate(zip(sentence, token_labels)):
        if label[2:] in ['rightidentifier', 'leftidentifier'] \
                and not keep_alltokens:
            continue
        tag = label[0]
        if tag=='B':
            if curr_phrase:
            # Below is redundant if the other tags handles correctly.       
                phrase_labels.append(curr_phrase)
            curr_phrase = label[2:]
        elif tag == 'I':
            if curr_phrase != label[2:] and curr_phrase:
                phrase_labels.append(curr_phrase)
                curr_phrase = label[2:]
        elif tag == 'L':
            if curr_phrase != label[2:] and curr_phrase:
                # Add if the previous label is different                    
                phrase_labels.append(curr_phrase)
            # Add current label                                             
            phrase_labels.append(label[2:])
            curr_phrase = ''
        elif tag == 'O':
            # Do nothing other than pushing the previous label
            if not keep_alltokens:
                if curr_phrase:
                    phrase_labels.append(curr_phrase)
                curr_phrase = ''
            else:
                if curr_phrase == 'O':
                    pass
                else:
                    if curr_phrase:
                        phrase_labels.append(curr_phrase)
                    curr_phrase = 'O'

        elif tag == 'U':
            if curr_phrase:
                phrase_labels.append(curr_phrase)
            phrase_labels.append(label[2:])
        else:
            print('Tag is incorrect in: {0}.'.format(label))
            pdb.set_trace()
        if len(phrase_labels)>0:
            if phrase_labels[-1] == '':
                pdb.set_trace()
    if curr_phrase != '':
        phrase_labels.append(curr_phrase)
    phrase_labels = [leave_one_word(\
                         leave_one_word(phrase_label, 'leftidentifier'),\
                            'rightidentifier')\
                        for phrase_label in phrase_labels]
    phrase_labels = list(reduce(adder, map(splitter, phrase_labels), []))
    return phrase_labels

def make_phrase_dict(sentence_dict, token_label_dict, keep_alltokens=False):
    #phrase_dict = OrderedDict()
    phrase_dict = dict()
    for srcid, token_labels_dict in token_label_dict.items():
        phrases = []
        for metadata_type, token_labels in token_labels_dict.items():
            if srcid not in sentence_dict:
                #pdb.set_trace()
                pass
            sentence = sentence_dict[srcid][metadata_type]
            phrases += bilou_tagset_phraser(
                sentence, token_labels, keep_alltokens)
        remove_indices = list()
        for i, phrase in enumerate(phrases):
            #TODO: Below is heuristic. Is it allowable?
            #if phrase.split('-')[0] in ['building', 'networkadapter',\
            #                            'leftidentifier', 'rightidentifier']:
            if phrase.split('-')[0] in ['leftidentifier', 'rightidentifier']\
                    and not keep_alltokens:
                pdb.set_trace()
                remove_indices.append(i)
                pass
        phrases = [phrase for i, phrase in enumerate(phrases)\
                   if i not in remove_indices]
        #phrase_dict[srcid] = phrases + phrases # TODO: Why did I put this before?
        phrase_dict[srcid] = phrases
    return phrase_dict

def hier_clustering(d, threshold=3):
    srcids = d.keys()
    tokenizer = lambda x: x.split()
    vectorizer = TfidfVectorizer(tokenizer=tokenizer)
    assert isinstance(d, dict)
    assert isinstance(list(d.values())[0], list)
    assert isinstance(list(d.values())[0][0], str)
    doc = [' '.join(d[srcid]) for srcid in srcids]
    vect = vectorizer.fit_transform(doc)
    #TODO: Make vect aligned to the required format
    z = linkage(vect.toarray(), metric='cityblock', method='complete')
    dists = list(set(z[:,2]))
#    threshold = 3
    #threshold = (dists[2] + dists[3]) / 2
    b = hier.fcluster(z, threshold, criterion='distance')
    cluster_dict = defaultdict(list)
    for srcid, cluster_id in zip(srcids, b):
        cluster_dict[str(cluster_id)].append(srcid)
    value_lengther = lambda x: len(x[1])
    return OrderedDict(\
               sorted(cluster_dict.items(), key=value_lengther, reverse=True))

def set_logger(logfile=None):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(message)s')

    # Console Handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    # File Handler
    if logfile:
        fh = logging.FileHandler(logfile, mode='w+')
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    return logger


def parallel_func(orig_func, return_idx, return_dict, *args):
    return_dict[return_idx] = orig_func(*args)


def iteration_wrapper(iter_num, func, prev_data=None, *params):
    step_datas = list()
    if not prev_data:
        prev_data = {'iter_num':0,
                     'learning_srcids': [],
                     'model_uuid': None}
    for i in range(0, iter_num):
        print('{0} th stage started'.format(prev_data['iter_num']))
        step_data = func(prev_data, *params)
        print('{0} th stage finished'.format(prev_data['iter_num']))
        step_datas.append(step_data)
        prev_data = step_data
        prev_data['iter_num'] += 1
    return step_datas


def replace_num_or_special(word):
    if re.match('\d+', word):
        return 'NUMBER'
    elif re.match('[a-zA-Z]+', word):
        return word
    else:
        return 'SPECIAL'

def adder(x, y):
    return x + y


def get_random_srcids_dep(building_list, source_sample_num_list):
    srcids = list()
    for building, source_sample_num in zip(building_list, 
                                           source_sample_num_list): 
                                        #  Load raw sentences of a building
        with open("metadata/%s_char_sentence_dict.json" % (building), 
                      "r") as fp:
            sentence_dict = json.load(fp)
        sentence_dict = dict((srcid, [char for char in sentence]) 
                             for (srcid, sentence) in sentence_dict.items())

        # Load character label mappings.
        with open('metadata/{0}_char_label_dict.json'.format(building), 
                      'r') as fp:
            label_dict = json.load(fp)

        # Select learning samples.
        # Learning samples can be chosen from the previous stage.
        # Or randomly selected.
        sample_srcid_list = select_random_samples(building, \
                                                  label_dict.keys(), \
                                                  source_sample_num, \
                                                  use_cluster_flag=True)
        srcids += sample_srcid_list
    return srcids

def get_cluster_dict(building):
    cluster_filename = 'model/%s_word_clustering_justseparate.json' % \
                           (building)
    with open(cluster_filename, 'r') as fp:
        cluster_dict = json.load(fp)
    return cluster_dict

def get_label_dict(building):
    filename = 'metadata/%s_label_dict_justseparate.json' % \
                           (building)
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data

def slack_notifier(msg):
    webhook_url = 'https://hooks.slack.com/services/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'
    body = {
        'text': msg
       }
    res = requests.post(webhook_url, json=body)


def load_data(target_building, source_buildings, bacnettype_flag=False):
    building_sentence_dict = dict()
    building_label_dict = dict()
    building_tagsets_dict = dict()
    known_tags_dict = defaultdict(list)

    units = csv2json(pd.read_csv(SCRABBLE_METADATA_DIR + '/unit_mapping.csv'),
                     'unit', 'word')
    units[None] = []
    units[''] = []
    bacnettypes = csv2json(pd.read_csv(SCRABBLE_METADATA_DIR +
                                       '/bacnettype_mapping.csv'),
                           'bacnet_type_str', 'candidates')
    bacnettypes[None] = []
    bacnettypes[''] = []
    for building in source_buildings:
        true_tagsets = {}
        label_dict = {}
        for labeled in LabeledMetadata.objects(building=building):
            srcid = labeled.srcid
            true_tagsets[srcid] = labeled.tagsets
            fullparsing = labeled.fullparsing
            labels = {}
            for metadata_type, pairs in fullparsing.items():
                labels[metadata_type] = [pair[1] for pair in pairs]
            label_dict[srcid] = labels

        building_tagsets_dict[building] = true_tagsets
        building_label_dict[building] = label_dict
        sentence_dict = dict()
        for raw_point in RawMetadata.objects(building=building):
            srcid = raw_point.srcid
            #if srcid in true_tagsets:
            metadata = raw_point['metadata']
            sentences = {}
            for clm in column_names:
                if clm in metadata:
                    sentences[clm] = [c for c in metadata[clm].lower()]
            sentence_dict[srcid]  = sentences
            bacnet_unit = metadata.get('BACnetUnit')
            if bacnet_unit:
                known_tags_dict[srcid] += units[bacnet_unit]
            if bacnettype_flag:
                known_tags_dict[srcid] += bacnettypes[metadata.get('BACnetTypeStr')]
        building_sentence_dict[building] = sentence_dict
    target_srcids = list(building_label_dict[target_building].keys())
    return building_sentence_dict, target_srcids, building_label_dict,\
        building_tagsets_dict, known_tags_dict


def calc_acc_sub_tagsets(true, pred, srcids):
    if not pred:
        return None, None
    tot_acc = 0
    tot_point_acc = 0
    for srcid in srcids:
        true_set = set(true[srcid])
        pred_set = set(pred[srcid])
        tot_acc += len(true_set.intersection(pred_set)) /\
            len(true_set.union(pred_set))
        pred_points = find_points(pred_set)
        true_points = find_points(true_set)
        if not pred_points and not true_points:
            tot_point_acc += 1
        else:
            tot_point_acc += len(true_points.intersection(pred_points)) /\
                len(true_points.union(pred_points))
    tot_acc /= len(srcids)
    tot_point_acc /= len(srcids)
    return tot_acc, tot_point_acc

def calc_acc_sub_fullparsing(true, pred, srcids):
    if not pred:
        return None
    tot_acc = 0
    tot_point_acc = 0
    for srcid in srcids:
        true_set = true[srcid]
        pred_set = pred[srcid]
        sent_len = 0
        curr_acc_cnt = 0
        for key, labels in pred_set.items():
            assert len(true_set[key]) == len(labels)
            curr_acc_cnt = sum([t==p for t, p in zip(true_set[key], labels)])
            sent_len += len(labels)
        tot_acc += curr_acc_cnt / sent_len
    tot_acc /= len(srcids)
    return tot_acc


def calc_acc(true, pred, true_crf, pred_crf, srcids, learning_srcids):
    tot_acc, tot_point_acc = calc_acc_sub_tagsets(true, pred, srcids)
    learning_acc, learning_point_acc = calc_acc_sub_tagsets(true,
                                                            pred,
                                                            learning_srcids)
    crf_tot_acc = calc_acc_sub_fullparsing(true_crf, pred_crf, srcids)
    crf_learning_acc = calc_acc_sub_fullparsing(true_crf,
                                                pred_crf,
                                                learning_srcids)

    return crf_tot_acc, crf_learning_acc, \
        tot_acc, tot_point_acc, \
        learning_acc, learning_point_acc


def print_status(scrabble, tot_acc, tot_point_acc,
                 learning_acc, learning_point_acc,
                 tot_crf_acc, learning_crf_acc):
    print('-----------------')
    print('srcids: {0}'.format(len(set(scrabble.learning_srcids))))
    if tot_acc:
        print('curr total accuracy: {0}'.format(tot_acc))
    if tot_point_acc:
        print('curr total point accuracy: {0}'.format(tot_point_acc))
    if learning_acc:
        print('curr learning accuracy: {0}'.format(learning_acc))
    if learning_point_acc:
        print('curr learning point accuracy: {0}'.format(learning_point_acc))
    if tot_crf_acc:
        print('curr CRF accuracy: {0}'.format(tot_crf_acc))
    if learning_crf_acc:
        print('curr learning CRF accuracy: {0}'.format(learning_crf_acc))

def find_points(tagsets):
    points = []
    for tagset in tagsets:
        postfix = tagset.split('_')[-1]
        if postfix in POINT_POSTFIXES:
            points.append(tagset)
    if not points:
        points = ['none']
    return set(points)


## Argparser

def str2slist(s):
    s.replace(' ', '')
    return s.split(',')

def str2ilist(s):
    s.replace(' ', '')
    return [int(c) for c in s.split(',')]

def str2bool(v):
    if v in ['true', 'True']:
        return True
    elif v in ['false', 'False']:
        return False
    else:
        assert(False)

argparser = argparse.ArgumentParser()
argparser.register('type','bool',str2bool)
argparser.register('type','slist', str2slist)
argparser.register('type','ilist', str2ilist)
argparser.add_argument('-task',
                       type=str,
                       help='Learning model among ["char2ir", "ir2tagsets", "scrabble"]',
                       choices=['char2ir', 'ir2tagsets', 'scrabble', 'tagsets2entities'],
                       dest='task',
                       required=True
                       )
argparser.add_argument('-bl',
                       type='slist',
                       help='Learning source building name list',
                       dest='source_building_list')
argparser.add_argument('-nl',
                       type='ilist',
                       help='A list of the number of learning sample',
                       dest='sample_num_list')
argparser.add_argument('-t',
                       type=str,
                       help='Target buildling name',
                       dest='target_building')
argparser.add_argument('-ub',
                       type='bool',
                       help='Use Brick when learning',
                       default=False,
                       dest='use_brick_flag')
argparser.add_argument('-ut',
                       type='bool',
                       help='Use Known Tags',
                       default=False,
                       dest='use_known_tags')
argparser.add_argument('-iter',
                       type=int,
                       help='Number of iteration for the given work',
                       dest='iter_num',
                       default=20)
argparser.add_argument('-nj',
                       type=int,
                       help='Number of processes for multiprocessing',
                       dest='n_jobs',
                       default=10)
argparser.add_argument('-inc',
                       type=int,
                       help='Inc num in each strage',
                       dest='inc_num',
                       default=10)
argparser.add_argument('-ct',
                       type=str,
                       help='Tagset classifier type. one of RandomForest, \
                          StructuredCC, MLP.',
                       dest='tagset_classifier_type',
                       default='MLP')
argparser.add_argument('-ts',
                       type='bool',
                       help='Flag to use time series features too',
                       dest='ts_flag',
                       default=False)
argparser.add_argument('-neg',
                       type='bool',
                       help='Negative Samples augmentation',
                       dest='negative_flag',
                       default=True)
argparser.add_argument('-post',
                       type=str,
                       help='postfix of result filename',
                       default='0',
                       dest = 'postfix')
argparser.add_argument('-crfqs',
                       type=str,
                       help='Query strategy for CRF',
                       default='confidence',
                       dest = 'crfqs')
argparser.add_argument('-entqs',
                       type=str,
                       help='Query strategy for CRF',
                       default='phrase_util',
                       dest = 'entqs')
argparser.add_argument('-g',
                       type='bool',
                       help='Graphize the result or not',
                       dest='graphize',
                       default=False)

def query_result(query):
    res = ResultHistory.objects(**query).first().to_mongo()
    return res

def get_result_obj(params, clean_history):
    query_keys = [
        'use_brick_flag',
        'use_known_tags',
        'source_building_list',
        'sample_num_list',
        'target_building',
        'negative_flag',
        'entqs',
        'crfqs',
        'tagset_classifier_type',
        'postfix',
        'task',
    ]
    query = {k: getattr(params, k) for k in query_keys}
    res_obj = ResultHistory.objects(**query).upsert_one(**query)
    #res_obj = ResultHistory.objects(**query)\
    #    .update_one(**query, upsert=True)
    if not res_obj.history or clean_history:
        res_obj.history = []
    res_obj.save()
    return res_obj