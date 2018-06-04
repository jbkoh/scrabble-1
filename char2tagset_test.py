import arrow
import pdb
import json
from copy import deepcopy

from scrabble import Scrabble
from data_model import *
from common import *
import random

t0 = arrow.get()

connect('oracle')

target_building = 'ap_m'
source_buildings = ['ebu3b', 'ap_m']
source_sample_num_list = [200, 10]

building_sentence_dict, target_srcids, building_label_dict,\
    building_tagsets_dict, known_tags_dict = load_data(target_building,
                                                       source_buildings)
tot_label_dict = {}
for building, tagsets_dict in building_tagsets_dict.items():
    tot_label_dict.update(tagsets_dict )

t1 = arrow.get()
print(t1-t0)
config = {
    'use_known_tags': True,
    'n_jobs':30,
    'tagset_classifier_type': 'MLP',
    'use_brick_flag': True,
}
scrabble = Scrabble(target_building,
                    target_srcids,
                    building_label_dict,
                    building_sentence_dict,
                    building_tagsets_dict,
                    source_buildings,
                    source_sample_num_list,
                    known_tags_dict,
                    config=config
                    )

scrabble.update_model([])
history = []
for i in range(0, 20):
    t2 = arrow.get()
    new_srcids = scrabble.select_informative_samples(10)
    scrabble.update_model(new_srcids)
    pred = scrabble.predict(target_srcids + scrabble.learning_srcids)
    pred_tags = scrabble.predict_tags(target_srcids)
    tot_acc, tot_point_acc, learning_acc, learning_point_acc = \
        calc_acc(tot_label_dict, pred, target_srcids, scrabble.learning_srcids)
    print_status(scrabble, tot_acc, tot_point_acc,
                 learning_acc, learning_point_acc)
    hist = {
        'pred': pred,
        'pred_tags': pred_tags,
        'learning_srcids': list(set(deepcopy(scrabble.learning_srcids)))
    }
    history.append(hist)

    t3 = arrow.get()
    print('{0}th took {1}'.format(i, t3 - t2))

    with open('result/scrabble_history_debug_{0}.json'.format(target_building), 'w') as fp:
        json.dump(history, fp)
