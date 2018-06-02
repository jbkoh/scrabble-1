import arrow
import pdb
import json
import pandas as pd
from collections import defaultdict
from copy import deepcopy

from ir2tagsets import Ir2Tagsets
from data_model import *

t0 = arrow.get()

connect('oracle')

def elem2list(elem):
    if isinstance(elem, str):
        return elem.split('_')
    else:
        return []


def csv2json(df, key_idx, value_idx):
    keys = df[key_idx].tolist()
    values = df[value_idx].tolist()
    return {k: elem2list(v) for k, v in zip(keys, values)}

units = csv2json(pd.read_csv('metadata/unit_mapping.csv'),
                 'unit', 'word')
units[None] = []
units[''] = []
bacnettypes = csv2json(pd.read_csv('metadata/bacnettype_mapping.csv'),
                       'bacnet_type_str', 'candidates')
bacnettypes[None] = []
bacnettypes[''] = []

column_names = ['VendorGivenName',
                 'BACnetName', 
                 'BACnetDescription']

known_tags_dict = defaultdict(list)

target_building = 'ap_m'
source_buildings = ['ap_m']
source_sample_num_list = [10]

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
        metadata = raw_point['metadata']
        if srcid in true_tagsets:
            sentence = None
            for clm in column_names:
                if not sentence:
                    sentence = [c for c in metadata[clm].lower()]
                else:
                    sentence += ['\n'] + \
                                [c for c in metadata[clm].lower()]
            sentence_dict[srcid]  = sentence
        known_tags_dict[srcid] += units[metadata.get('BACnetUnit')]
        #known_tags_dict[srcid] += bacnettypes[metadata.get('BACnetTypeStr')]
    building_sentence_dict[building] = sentence_dict

known_tags_dict = dict(known_tags_dict)
target_srcids = list(building_label_dict[target_building].keys())
t1 = arrow.get()
print(t1-t0)
ir2tagsets = Ir2Tagsets(target_building,
                        target_srcids,
                        building_label_dict,
                        building_sentence_dict,
                        building_tagsets_dict,
                        source_buildings,
                        source_sample_num_list,
                        known_tags_dict=known_tags_dict,
                        conf={
                            'use_known_tags': False,
                            'n_jobs':24
                        }
                        )

history = []
ir2tagsets.update_model([])
for i in range(0, 20):
    t2 = arrow.get()
    new_srcids = ir2tagsets.select_informative_samples(10)
    ir2tagsets.update_model(new_srcids)
    pred = ir2tagsets.predict(target_srcids + ir2tagsets.learning_srcids)
    proba = ir2tagsets.predict_proba(target_srcids)
    t3 = arrow.get()
    hist = {
        'sricds': deepcopy(ir2tagsets.learning_srcids),
        'pred': pred
    }
    history.append(hist)
    print('{0}th took {1}'.format(i, t3 - t2))
    with open('result/test_tagsonly.json', 'w') as fp:
        json.dump(history, fp, indent=2)
