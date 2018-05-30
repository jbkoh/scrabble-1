import json
from common import *
from collections import Counter, defaultdict
from functools import reduce
import pdb
from data_model import *

source_buildings = ['ap_m', 'bml']
target_building = 'bml'

with open('result/scrabble_test_notransfer.json', 'r') as fp:
    res = json.load(fp)

point_accuracy_cnt = 0
set_accuracy_cnt = 0
unfound_point_cnts = defaultdict(int)

for srcid, pred_tagsets in res.items():
    labeled_metadata = LabeledMetadata.objects(srcid=srcid).first().to_mongo()
    point_tagset = labeled_metadata['point_tagset']
    true_tagsets = labeled_metadata['tagsets']
    if point_tagset in pred_tagsets:
        point_accuracy_cnt += 1
    else:
        raw_metadata = RawMetadata.objects(srcid=srcid).first()\
            .to_mongo()['metadata']
        #print('Name: {0}'.format(raw_metadata['VendorGivenName']))
        #print('Pred: {0}'.format(pred_tagsets))
        #print('True: {0}'.format(true_tagsets))
        unfound_point_cnts[point_tagset] += 1

    true_tagsets_set = set(true_tagsets)
    pred_tagsets_set = set(pred_tagsets)
    set_accuracy_cnt += len(true_tagsets_set.intersection(pred_tagsets_set)) / \
        len(true_tagsets_set.union(pred_tagsets_set))


print('point accuracy: {0}'.format(point_accuracy_cnt / len(res)))
print('general accuracy: {0}'.format(set_accuracy_cnt / len(res)))
print('unfound tagstes: ')
print(unfound_point_cnts)
pdb.set_trace()
