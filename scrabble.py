import argparse
import pdb
from operator import itemgetter
from copy import deepcopy

import arrow

from common import *
from char2ir import Char2Ir
from ir2tagsets import Ir2Tagsets
from base_scrabble import BaseScrabble

def str2bool(v):
    if v in ['true', 'True']:
        return True
    elif v in ['false', 'False']:
        return False
    else:
        assert(False)

def str2slist(s): 
    s.replace(' ', '')
    return s.split(',')

def str2ilist(s):
    s.replace(' ', '')
    return [int(c) for c in s.split(',')]


class Scrabble(BaseScrabble):
    """docstring for Scrabble"""
    def __init__(self, 
                 source_building_list, 
                 target_building, 
                 sample_num_list,
                 building_sentence_dict,
                 building_label_dict,
                 building_tagsets_dict,
                 conf):
        self.source_building_list = source_building_list
        self.target_building = target_building
        self.sample_num_list = sample_num_list
        self.building_sentence_dict = building_sentence_dict
        self.building_label_dict = building_label_dict
        self.building_tagsets_dict = building_tagsets_dict
        self.conf = conf
        
        self.building_srcid_dict = dict()
        for building, sentences in self.building_sentence_dict.items():
            self.building_srcid_dict[building] = list(sentences.keys())


        self.char2ir = Char2Ir(source_building_list,
                          target_building,
                          sample_num_list,
                          building_sentence_dict,
                          building_label_dict,
                          conf)
        self.ir2tagsets = Ir2Tagsets(source_building_list, 
                                     target_building, 
                                     sample_num_list, 
                                     building_sentence_dict,
                                     building_label_dict,
                                     building_tagsets_dict,
                                     conf)
    def char2tagset_onestep(self,
                            step_data,
                            building_list,
                            target_building,
                            sample_num_list,
                            use_cluster_flag=False,
                            use_brick_flag=False,
                            crftype='crfsuite',
                            eda_flag=False,
                            negative_flag=True,
                            debug_flag=False,
                            n_jobs=8, # TODO parameterize
                            ts_flag=False,
                            inc_num=10,
                            crfqs='confidence',
                            entqs='phrase_util'):
        begin_time = arrow.get()
        step_data = deepcopy(step_data)
        step_data['learning_srcids'] = step_data['next_learning_srcids']

        step_data = self.char2ir.char2ir_onestep(
                                    step_data,
                                    building_list,
                                    sample_num_list,
                                    target_building,
                                    inc_num / 2,
                                    crfqs)

        pdb.set_trace()
        step_data = self.ir2tagsets.ir2tagset_onestep(step_data,
                                      building_list,
                                      sample_num_list,
                                      target_building,
                                      use_cluster_flag,
                                      use_brick_flag,
                                      eda_flag,
                                      negative_flag,
                                      debug_flag,
                                      n_jobs, # TODO parameterize
                                      ts_flag,
                                      inc_num / 2,
                                      entqs)
        end_time = arrow.get()
        logging.info('An iteration takes ' + str(end_time - begin_time))
        return step_data

    def char2tagset_iteration(self, iter_num, custom_postfix='', *params):
        """
        params: 
            building_list,
            source_sample_num_list,
            target_building,
            use_cluster_flag=False,
            use_brick_flag=False,
            crftype='crfsuite'
            eda_flag=False,
            negative_flag=True,
            debug_flag=True,
            n_jobs=8, # TODO parameterize
            ts_flag=False)
        """
        begin_time = arrow.get()
        building_list = params[0]
        source_sample_num_list = params[1]
        prev_data = {'iter_num':0,
                     'next_learning_srcids': get_random_srcids(
                                            building_list,
                                            source_sample_num_list),
                     'model_uuid': None}
        step_datas = self.iteration_wrapper(iter_num, self.char2tagset_onestep, 
                                       prev_data, *params)

        building_list = params[0]
        target_building = params[2]
        postfix = 'char2tagset_iter' 
        if custom_postfix:
            postfix += '_' + custom_postfix
        with open('result/crf_entity_iter_{0}_{1}.json'\
                .format(''.join(building_list+[target_building]), postfix), 'w') as fp:
            json.dump(step_datas, fp, indent=2)
        end_time = arrow.get()
        print(iter_num, " iterations took: ", end_time - begin_time)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.register('type','bool',str2bool)
    parser.register('type','slist', str2slist)
    parser.register('type','ilist', str2ilist)

    parser.add_argument(choices=['learn_crf', 'predict_crf', 'entity', 'crf_entity', \
                                 'init', 'result', 'iter_crf'],
                        dest = 'prog')

    parser.add_argument('predict',
                         action='store_true',
                         default=False)

    """
    parser.add_argument('-b',
                        type=str,
                        help='Learning source building name',
                        dest='source_building')
    parser.add_argument('-n', 
                        type=int, 
                        help='The number of learning sample',
                        dest='sample_num') """ 
    parser.add_argument('-bl',
                        type='slist',
                        help='Learning source building name list',
                        dest='source_building_list')
    parser.add_argument('-nl',
                        type='ilist',
                        help='A list of the number of learning sample',
                        dest='sample_num_list')
    parser.add_argument('-l',
                        type=str,
                        help='Label type (either label or category',
                        default='label',
                        dest='label_type')
    parser.add_argument('-c',
                        type='bool',
                        help='flag to indicate use hierarchical cluster \
                                to select learning samples.',
                        default=True,
                        dest='use_cluster_flag')
    parser.add_argument('-d',
                        type='bool',
                        help='Debug mode flag',
                        default=False,
                        dest='debug_flag')
    parser.add_argument('-t',
                        type=str,
                        help='Target buildling name',
                        dest='target_building')
    parser.add_argument('-crftype',
                        type=str,
                        help='CRF Package Name',
                        default='crfsuite',
                        dest='crftype')
    parser.add_argument('-eda',
                        type='bool',
                        help='Flag to use Easy Domain Adapatation',
                        default=False,
                        dest='eda_flag')
    parser.add_argument('-ub',
                        type='bool',
                        help='Use Brick when learning',
                        default=False,
                        dest='use_brick_flag')
    parser.add_argument('-avg',
                        type=int,
                        help='Number of exp to get avg. If 1, ran once',
                        dest='avgnum',
                        default=1)
    parser.add_argument('-iter',
                        type=int,
                        help='Number of iteration for the given work',
                        dest='iter_num',
                        default=1)
    parser.add_argument('-wk',
                        type=int,
                        help='Number of workers for high level MP',
                        dest='worker_num',
                        default=2)
    parser.add_argument('-nj',
                        type=int,
                        help='Number of processes for multiprocessing',
                        dest='n_jobs',
                        default=4)
    parser.add_argument('-inc',
                        type=int,
                        help='Inc num in each strage',
                        dest='inc_num',
                        default=10)
    parser.add_argument('-ct',
                        type=str,
                        help='Tagset classifier type. one of RandomForest, \
                              StructuredCC.',
                        dest='tagset_classifier_type',
                        default='StructuredCC')
    parser.add_argument('-ts',
                        type='bool',
                        help='Flag to use time series features too',
                        dest='ts_flag',
                        default=False)
    parser.add_argument('-neg',
                        type='bool',
                        help='Negative Samples augmentation',
                        dest='negative_flag',
                        default=True)
    parser.add_argument('-exp', 
                        type=str,
                        help='type of experiments for result output',
                        dest = 'exp_type')
    parser.add_argument('-post', 
                        type=str,
                        help='postfix of result filename',
                        default='0',
                        dest = 'postfix')
    parser.add_argument('-crfqs', 
                        type=str,
                        help='Query strategy for CRF',
                        default='confidence',
                        dest = 'crfqs')
    parser.add_argument('-entqs',
                        type=str,
                        help='Query strategy for CRF',
                        default='phrase_util',
                        dest = 'entqs')
                        

    args = parser.parse_args()

    tagset_classifier_type = args.tagset_classifier_type
    
    building_sentence_dict = dict()
    building_label_dict = dict()
    building_tagsets_dict = dict()
    for building in args.source_building_list + [args.target_building]:
        # Load character label mappings.
        with open('metadata/{0}_char_label_dict.json'.format(building), 'r')\
            as fp:
            one_label_dict = json.load(fp)

        with open('metadata/{0}_true_tagsets.json'.format(building), 'r') \
            as fp:
            one_tagsets_dict = json.load(fp)

        one_sentence_dict = {}
        for srcid in one_label_dict.keys():
            one_sentence_dict[srcid] = list(map(itemgetter(0), 
                                            one_label_dict[srcid]))
            one_label_dict[srcid] = list(map(itemgetter(1), 
                                         one_label_dict[srcid]))
        building_sentence_dict[building] = one_sentence_dict
        building_label_dict[building] = one_label_dict
        building_tagsets_dict[building] = one_tagsets_dict

    # Init objects
    if args.prog in ['learn_crf', 'iter_crf', 'predict_crf']:
        char2ir = Char2Ir(args.source_building_list,
                          args.target_building,
                          args.sample_num_list,
                          building_sentence_dict,
                          building_label_dict,
                          {
                              'use_cluster_flag': args.use_cluster_flag,
                              'use_brick_flag': args.use_brick_flag
                          })
    if args.prog in ['entity']:
        ir2tagsets = Ir2Tagsets(args.source_building_list, 
                                args.target_building, 
                                args.sample_num_list, 
                                building_sentence_dict,
                                building_label_dict,
                                conf={
                                    'use_cluster_flag': False,
                                    'use_brick_flag': False
                                })
    if args.prog in ['crf_entity']:
        scrabble = Scrabble(args.source_building_list, 
                            args.target_building, 
                            args.sample_num_list, 
                            building_sentence_dict,
                            building_label_dict,
                            building_tagsets_dict,
                            conf={
                            'use_cluster_flag': True,
                            'use_brick_flag': True 
                            })


    if args.prog == 'learn_crf':
        char2ir.learn_crf_model(args.source_building_list,
                                args.sample_num_list)
        #learn_crf_model(building_list=args.source_building_list,
        #                source_sample_num_list=args.sample_num_list,
        #                use_cluster_flag=args.use_cluster_flag,
        #                use_brick_flag=args.use_brick_flag,
        #                crftype=args.crftype)
    elif args.prog == 'predict_crf':
        char2ir.crf_test(building_list=args.source_building_list,
                 source_sample_num_list=args.sample_num_list,
                 target_building=args.target_building)
    elif args.prog == 'iter_crf':
        params = (args.source_building_list,
                  args.sample_num_list,
                  args.target_building,
                  args.inc_num,
                  args.crfqs,
#                  args.n_jobs)
                 )
        char2ir.char2ir_iteration(args.iter_num, args.postfix, *params)
    elif args.prog == 'entity':
        if args.avgnum == 1:
            ir2tagsets.entity_recognition_iteration(args.iter_num,
                                         args.postfix,
                                         args.source_building_list,
                                         args.sample_num_list,
                                         args.target_building,
                                         args.use_cluster_flag,
                                         args.use_brick_flag,
                                         args.debug_flag,
                                         args.eda_flag,
                                         args.ts_flag,
                                         args.negative_flag,
                                         args.n_jobs,
                                         args.inc_num,
                                         args.entqs
                                        )
        elif args.avgnum>1:
            ir2tagsets.entity_recognition_from_ground_truth_get_avg(args.avgnum,
                building_list=args.source_building_list,
                source_sample_num_list=args.sample_num_list,
                target_building=args.target_building,
                use_cluster_flag=args.use_cluster_flag,
                use_brick_flag=args.use_brick_flag,
                eda_flag=args.eda_flag,
                ts_flag=args.ts_flag,
                negative_flag=args.negative_flag,
                n_jobs=args.n_jobs,
                worker_num=args.worker_num)
    elif args.prog == 'crf_entity':
        params = (args.source_building_list,
                  args.sample_num_list,
                  args.target_building,
                  args.use_cluster_flag,
                  args.use_brick_flag,
                  args.crftype,
                  args.eda_flag,
                  args.negative_flag,
                  args.debug_flag,
                  args.n_jobs,
                  args.ts_flag,
                  args.inc_num,
                  args.crfqs,
                  args.entqs
                  )
        scrabble.char2tagset_iteration(args.iter_num, args.postfix, *params)
    elif args.prog == 'result':
        assert args.exp_type in ['crf', 'entity', 'crf_entity', 'entity_iter',
                                 'etc', 'entity_ts', 'cls']
        if args.exp_type == 'crf':
            crf_result()
        elif args.exp_type == 'entity':
            entity_result()
        elif args.exp_type == 'crf_entity':
            crf_entity_result()
        elif args.exp_type == 'entity_iter':
            entity_iter_result()
        elif args.exp_type == 'entity_ts':
            entity_ts_result()
        elif args.exp_type == 'cls':
            cls_comp_result()
        elif args.exp_type == 'etc':
            etc_result()

    elif args.prog == 'init':
        init()
    else:
        #print('Either learn or predict should be provided')
        assert(False)


