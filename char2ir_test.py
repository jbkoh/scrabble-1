import arrow
import pdb

from char2ir import Char2Ir
from data_model import *

t0 = arrow.get()

column_names = ['VendorGivenName', 
                 'BACnetName', 
                 'BACnetDescription']


target_building = 'ebu3b'
source_buildings = ['ap_m', 'ebu3b']
source_sample_num_list = [100, 0]


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

target_srcids = list(building_label_dict[target_building].keys())
t1 = arrow.get()
print(t1-t0)

char2ir = Char2Ir(target_building, 
                  target_srcids,
                  building_label_dict,
                  building_sentence_dict,
                  source_buildings,
                  source_sample_num_list,
                  conf={
                      'use_cluster_flag': True,
                      #'use_brick_flag': False
                  })

char2ir.update_model([])

for i in range(0,5):
    t1 = arrow.get()
    new_srcids = char2ir.select_informative_samples(10)
    char2ir.update_model(new_srcids)
    t2 = arrow.get()
    print('{0}th took: {1}'.format(i, t2-t1))

pred = char2ir.predict(target_srcids)
proba = char2ir.predict_proba(target_srcids)
pdb.set_trace()
