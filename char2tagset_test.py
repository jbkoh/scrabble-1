import arrow
import pdb
import json

from scrabble import Scrabble
from data_model import *
from common import *
import random

def calc_acc(true, pred, srcids, learning_srcids):
    tot_acc = 0
    for srcid in srcids:
        true_set = set(true[srcid])
        pred_set = set(pred[srcid])
        tot_acc += len(true_set.intersection(pred_set)) /\
            len(true_set.union(pred_set))
    tot_acc /= len(srcids)

    learning_acc = 0
    for srcid in learning_srcids:
        true_set = set(true[srcid])
        pred_set = set(pred[srcid])
        learning_acc += len(true_set.intersection(pred_set)) /\
            len(true_set.union(pred_set))
    learning_acc /= len(learning_srcids)
    return tot_acc, learning_acc

def print_status(scrabble, tot_acc, learning_acc):
    print('char2ir srcids: {0}'.format(len(scrabble.char2ir.learning_srcids)))
    print('ir2tagsets srcids: {0}'.format(len(scrabble.ir2tagsets.learning_srcids)))
    print('curr total accuracy: {0}'.format(tot_acc))
    print('curr learning accuracy: {0}'.format(learning_acc))


t0 = arrow.get()

connect('oracle')



column_names = ['VendorGivenName',
                 'BACnetName',
                 'BACnetDescription']

target_building = 'ap_m'
source_buildings = ['ap_m']
source_sample_num_list = [10]
#source_buildings = ['ap_m', 'ebu3b']
#source_sample_num_list = [200, 10]
#source_sample_num_list = [5, 0]

building_sentence_dict = dict()
building_label_dict = dict()
building_tagsets_dict = dict()
for building in source_buildings:
    true_tagsets = {}
    label_dict = {}
    for labeled in LabeledMetadata.objects(building=building):
        srcid = labeled.srcid
        true_tagsets[srcid] = labeled.tagsets
        fullparsing = None
        for clm in column_names:
            one_fullparsing = [i[1] for i in labeled.fullparsing[clm]]
            if not fullparsing:
                fullparsing = one_fullparsing
            else:
                fullparsing += ['O'] + one_fullparsing
                #  This format is alinged with the sentence
                #  configormation rule.
        label_dict[srcid] = fullparsing

    building_tagsets_dict[building] = true_tagsets
    building_label_dict[building] = label_dict
    sentence_dict = dict()
    for raw_point in RawMetadata.objects(building=building):
        srcid = raw_point.srcid
        if srcid in true_tagsets:
            metadata = raw_point['metadata']
            sentence = None
            for clm in column_names:
                if not sentence:
                    sentence = [c for c in metadata[clm].lower()]
                else:
                    sentence += ['\n'] + \
                                [c for c in metadata[clm].lower()]
            sentence_dict[srcid]  = sentence
    building_sentence_dict[building] = sentence_dict

target_srcids = random.sample(list(building_label_dict[target_building].keys()), 1000)
t1 = arrow.get()
print(t1-t0)
scrabble = Scrabble(target_building,
                    target_srcids,
                    building_label_dict,
                    building_sentence_dict,
                    building_tagsets_dict,
                    source_buildings,
                    source_sample_num_list
                    )

scrabble.update_model([])
history = []
for i in range(0, 20):
    t2 = arrow.get()
    new_srcids = scrabble.select_informative_samples(10)
    scrabble.update_model(new_srcids)
    pred = scrabble.predict(target_srcids + scrabble.learning_srcids)
    pred_tags = scrabble.predict_tags(target_srcids)
    tot_acc, learning_acc = calc_acc(building_tagsets_dict[target_building],
                                     pred,
                                     target_srcids, scrabble.learning_srcids)
    print_status(scrabble, tot_acc, learning_acc)
    slack_notifier('an iteration done')
    hist = {
        'pred': pred,
        'pred_tags': pred_tags,
        'learning_srcids': scrabble.learning_srcids}
    history.append(hist)

    t3 = arrow.get()
    print('{0}th took {1}'.format(i, t3 - t2))

    with open('result/scrabble_history_{0}.json'.format(target_building), 'w') as fp:
        json.dump(history, fp)
