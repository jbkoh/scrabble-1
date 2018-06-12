import arrow
import pdb
import sys
import json
from copy import deepcopy
import os
import random

from scrabble import Scrabble
from scrabble.data_model import *
from scrabble.common import *


args =argparser.parse_args()

t0 = arrow.get()

res_obj = get_result_obj(args, True)

source_buildings = args.source_building_list
target_building = args.target_building
source_sample_num_list = args.sample_num_list
framework_type = args.task

building_sentence_dict, target_srcids, building_label_dict,\
    building_tagsets_dict, known_tags_dict = load_data(target_building,
                                                       source_buildings)
tot_tagsets_dict = {}
for building, tagsets_dict in building_tagsets_dict.items():
    tot_tagsets_dict.update(tagsets_dict )

tot_labels_dict = {}
for building, labels_dict in building_label_dict.items():
    tot_labels_dict.update(labels_dict)

t1 = arrow.get()
print(t1-t0)
config = {
    'use_known_tags': args.use_known_tags,
    'n_jobs': args.n_jobs,
    'tagset_classifier_type': args.tagset_classifier_type,
    'use_brick_flag': args.use_brick_flag,
    'crfqs': args.crfqs,
    'entqs': args.entqs,
    'negative_flag': args.negative_flag
}

learning_srcid_file = 'metadata/test'
for building, source_sample_num in zip(source_buildings,
                                       source_sample_num_list):
    learning_srcid_file += '_{0}_{1}'.format(building, source_sample_num)
learning_srcid_file += '_srcids.json'

if os.path.isfile(learning_srcid_file):
    with open(learning_srcid_file, 'r') as fp:
        predefined_learning_srcids = json.load(fp)
else:
    predefined_learning_srcids = []
    for building, source_sample_num in zip(source_buildings,
                                           source_sample_num_list):
        predefined_learning_srcids += select_random_samples(
            building,
            building_tagsets_dict[building].keys(),
            True,
            source_sample_num,
            building_sentence_dict[building],
        )
    with open(learning_srcid_file, 'w') as fp:
        json.dump(predefined_learning_srcids, fp)

scrabble = Scrabble(target_building,
                    target_srcids,
                    building_label_dict,
                    building_sentence_dict,
                    building_tagsets_dict,
                    source_buildings,
                    source_sample_num_list,
                    known_tags_dict,
                    config=config,
                    learning_srcids=predefined_learning_srcids
                    )
if framework_type == 'char2ir':
    framework = scrabble.char2ir
elif framework_type == 'ir2tagsets':
    framework = scrabble.ir2tagsets
elif framework_type == 'tagsets2entities':
    framework = scrabble.tagsets2entities
    entities_dict = framework.map_tags_tagsets()
    framework.graphize(entities_dict)
    sys.exit(1)
elif framework_type == 'scrabble':
    framework = scrabble

framework.update_model([])
history = []
curr_learning_srcids = []
for i in range(0, args.iter_num):
    t2 = arrow.get()
    new_srcids = framework.select_informative_samples(args.inc_num)
    framework.update_model(new_srcids)
    if framework_type == 'char2ir':
        pred_tags = framework.predict(target_srcids)
        pred = None
    elif framework_type == 'ir2tagsets':
        pred = framework.predict(target_srcids + scrabble.learning_srcids)
        pred_tags = None
    elif framework_type == 'scrabbe':
        pred = framework.predict(target_srcids + scrabble.learning_srcids)
        pred_tags = framework.predict_tags(target_srcids)

    tot_crf_acc, learning_crf_acc, tot_acc, tot_point_acc,\
        learning_acc, learning_point_acc = calc_acc(
            true      = tot_tagsets_dict,
            pred      = pred,
            true_crf  = tot_labels_dict,
            pred_crf  = pred_tags,
            srcids    = target_srcids,
            learning_srcids = framework.learning_srcids)
    print_status(framework, tot_acc, tot_point_acc,
                 learning_acc, learning_point_acc,
                 tot_crf_acc, learning_crf_acc)
    new_srcids = [srcid for srcid in set(framework.learning_srcids)
                  if srcid not in curr_learning_srcids]
    hist = {
        'pred': pred,
        'pred_tags': pred_tags,
        'new_srcids': new_srcids,
        'learning_srcids': len(list(set(framework.learning_srcids)))
    }
    curr_learning_srcids = list(set(framework.learning_srcids))
    t3 = arrow.get()
    res_obj.history.append(hist)
    res_obj.save()
    print('{0}th took {1}'.format(i, t3 - t2))
