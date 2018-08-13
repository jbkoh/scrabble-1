import arrow
import pdb
import sys
import json
from copy import deepcopy
import os
import random

from scrabble.data_model import *
from scrabble.common import *
#from scrabble.char2ir import Char2Ir

argparser = argparse.ArgumentParser()
argparser.register('type','bool',str2bool)
argparser.register('type','slist', str2slist)
argparser.register('type','ilist', str2ilist)

argparser.add_argument('-task',
                       type=str,
                       help='Types of the learning task. One of ``scrabble``, ``char2ir`` and ``ir2tagsets``. ``scrabble`` tests the entire model and each of the others does for each step (1st and 2nd.)',
                       choices=['char2ir', 'ir2tagsets', 'scrabble', 'tagsets2entities'],
                       dest='task',
                       required=True
                       )
argparser.add_argument('-bl',
                       type='slist',
                       help='Learning source building name list: list of source building names deliminated by comma (e.g., ``-bl ebu3b,bml``).',
                       dest='source_building_list',
                       required=True
                       )
argparser.add_argument('-nl',
                       type='ilist',
                       help='Sample number list: list of sample numbers per building. The order should be same as bl. (e.g., ``-nl 200,1``)',
                       dest='sample_num_list',
                       required=True,
                       )
argparser.add_argument('-t',
                       type=str,
                       help='Target building name: Name of the target building. (e.g., ``-t bml``)',
                       dest='target_building',
                       required=True,
                       )
argparser.add_argument('-ub',
                       type='bool',
                       help='Use Brick when learning',
                       default=False,
                       dest='use_brick_flag')
argparser.add_argument('-ut',
                       type='bool',
                       help='Whether to use Brick when learning. (default: ``true``).',
                       default=False,
                       dest='use_known_tags')
argparser.add_argument('-iter',
                       type=int,
                       help='Total number of iterations. (default: ``20``)',
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
argparser.add_argument('-st',
                       type=str,
                       help='Sequential Model type, either crf or lstm',
                       dest='sequential_type',
                       default='crf')
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
                       help='Use sample augmentation using negative sample synthesis. (default: ``true``)',
                       dest='negative_flag',
                       default=True)
argparser.add_argument('-post',
                       type=str,
                       help='postfix of result filename',
                       default='0',
                       dest = 'postfix')
argparser.add_argument('-crfqs',
                       type=str,
                       help='Active learning query strategy for ``char2ir`` (default: ``confidence``).',
                       default='confidence',
                       dest = 'crfqs')
argparser.add_argument('-entqs',
                       type=str,
                       help='Active learning query strategy for ``ir2tagsets`` (default: ``phrase_util``).',
                       default='phrase_util',
                       dest = 'entqs')
argparser.add_argument('-g',
                       type='bool',
                       help='Graphize the result or not',
                       dest='graphize',
                       default=False)
args =argparser.parse_args()

t0 = arrow.get()

res_obj = get_result_obj(args, True)
full_history_filename = 'result/{0}_{1}_{2}.json'.format(
    res_obj.task, res_obj.sequential_type, res_obj.postfix)
full_history = []

source_buildings = args.source_building_list
target_building = args.target_building
source_sample_num_list = args.sample_num_list
framework_type = args.task

building_sentence_dict, target_srcids, building_label_dict,\
    building_tagsets_dict, known_tags_dict = load_data(target_building,
                                                       source_buildings,
                                                       bacnettype_flag=True,
                                                       )
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
    'negative_flag': args.negative_flag,
    'ts_flag': args.ts_flag,
    'sequential_type': args.sequential_type,
}

#learning_srcid_file = 'metadata/test'
#for building, source_sample_num in zip(source_buildings,
#                                       source_sample_num_list):
#    learning_srcid_file += '_{0}_{1}'.format(building, source_sample_num)
#learning_srcid_file += '_srcids.json'

if False and os.path.isfile(learning_srcid_file):
    # Disable this
    with open(learning_srcid_file, 'r') as fp:
        predefined_learning_srcids = json.load(fp)
else:
    predefined_learning_srcids = []
    for building, source_sample_num in zip(source_buildings,
                                           source_sample_num_list):
        predefined_learning_srcids += select_random_samples(
            building = building,
            srcids = building_tagsets_dict[building].keys(),
            n=source_sample_num,
            use_cluster_flag = True,
            sentence_dict = building_sentence_dict[building],
            shuffle_flag = False
        )
    #with open(learning_srcid_file, 'w') as fp:
    #    json.dump(predefined_learning_srcids, fp)


if framework_type == 'char2ir':
    """
    framework = Char2Ir(target_building,
                        target_srcids,
                        building_label_dict,
                        building_sentence_dict,
                        source_buildings,
                        source_sample_num_list,
                        learning_srcids=predefined_learning_srcids,
                        config=config
                        )
    """
    from scrabble import Scrabble
    scrabble = Scrabble(target_building,
                        target_srcids,
                        building_label_dict,
                        building_sentence_dict,
                        building_tagsets_dict,
                        source_buildings,
                        source_sample_num_list,
                        known_tags_dict={},
                        config=config,
                        learning_srcids=predefined_learning_srcids
                        )
    framework = scrabble.char2ir
elif framework_type == 'ir2tagsets':
    from scrabble import Scrabble
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
    framework = scrabble.ir2tagsets
elif framework_type == 'tagsets2entities':
    from scrabble import Scrabble
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
    framework = scrabble.tagsets2entities
    entities_dict = framework.map_tags_tagsets()
    framework.graphize(entities_dict)
    sys.exit(1)
elif framework_type == 'scrabble':
    from scrabble import Scrabble
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
    framework = scrabble

#framework.update_model([])
history = []
curr_learning_srcids = []
for i in range(0, args.iter_num):
    t2 = arrow.get()
    if i == 0:
        new_srcids = []
    else:
        new_srcids = framework.select_informative_samples(args.inc_num)
    framework.update_model(new_srcids)
    if framework_type == 'char2ir':
        pred_tags = framework.predict(target_srcids + framework.learning_srcids)
        """
        pred_phrases = defaultdict(dict)
        for srcid, tags_dict in pred_tags.items():
            for metadata_type, tags in tags_dict.items():
                pred_phrases[srcid][metadata_type] = list(set(
                    bilou_tagset_phraser(tags)
                ))
        """
        pred = None
        pred_phrases = make_phrase_dict(token_label_dict=pred_tags)
    elif framework_type == 'ir2tagsets':
        pred = framework.predict(target_srcids + scrabble.learning_srcids)
        pred_tags = None
        pred_phrases = None
    elif framework_type == 'scrabble':
        pred = framework.predict(target_srcids + scrabble.learning_srcids)
        pred_tags = None
        pred_phrases = None
        #pred_tags = framework.predict_tags(target_srcids)

    tot_crf_acc, learning_crf_acc, tot_f1, tot_macro_f1,\
        tot_acc, tot_point_acc, learning_acc, learning_point_acc = calc_acc(
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
    if framework_type == 'char2ir':
        hist = {
            'acc': tot_crf_acc,
            'f1': tot_f1,
            'macrof1': tot_macro_f1,
            'new_srcids': new_srcids,
            'learning_srcids': len(list(set(framework.learning_srcids)))
        }
    else:
        hist = {
            'pred': pred,
            'pred_tags': pred_tags,
            'new_srcids': new_srcids,
            'learning_srcids': len(list(set(framework.learning_srcids)))
        }
    curr_learning_srcids = list(set(framework.learning_srcids))
    t3 = arrow.get()
    res_obj['history'].append(hist)
    res_obj.save()
    full_history.append({
        'summary': hist,
        'full': {
            'pred_tags': pred_tags,
            'pred_phrases': pred_phrases
        }
    })
    with open(full_history_filename, 'w') as fp:
        json.dump(full_history, fp, indent=2)
    print('{0}th took {1}'.format(i, t3 - t2))
