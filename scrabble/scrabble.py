import pdb
from copy import deepcopy

from .ir2tagsets import Ir2Tagsets
from .char2ir import Char2Ir
from .base_scrabble import BaseScrabble
from .common import *




class Scrabble(BaseScrabble):
    def __init__(self,
                 target_building,
                 target_srcids,
                 building_label_dict,
                 building_sentence_dict,
                 building_tagsets_dict,
                 source_buildings,
                 source_sample_num_list,
                 known_tags_dict={},
                 config={},
                 learning_srcids=[]
                 ):
        super(Scrabble, self).__init__(
                 target_building,
                 target_srcids,
                 building_label_dict,
                 building_sentence_dict,
                 building_tagsets_dict,
                 source_buildings,
                 source_sample_num_list,
                 deepcopy(learning_srcids),
                 config)
        self.target_srcids = target_srcids
        self.known_tags_dict = known_tags_dict
        if 'use_cluster_flag' in config:
            self.use_cluster_flag = config['use_cluster_flag']
        else:
            self.use_cluster_flag = True

        self.init_data()
        self.char2ir = Char2Ir(target_building,
                               target_srcids,
                               building_label_dict,
                               building_sentence_dict,
                               source_buildings,
                               source_sample_num_list,
                               deepcopy(self.learning_srcids),
                               config
                               )
        self.ir2tagsets = Ir2Tagsets(target_building,
                                     target_srcids,
                                     building_label_dict,
                                     building_sentence_dict,
                                     building_tagsets_dict,
                                     source_buildings,
                                     source_sample_num_list,
                                     deepcopy(self.learning_srcids),
                                     known_tags_dict=known_tags_dict,
                                     config=config
                                     )

    def init_data(self):
        self.sentence_dict = {}
        self.label_dict = {}
        self.tagsets_dict = {}
        self.phrase_dict = {}
        #self.point_dict = {}

        for building, source_sample_num in zip(self.source_buildings,
                                               self.source_sample_num_list):
            self.sentence_dict.update(self.building_sentence_dict[building])
            one_label_dict = self.building_label_dict[building]
            self.label_dict.update(one_label_dict)

            if not self.learning_srcids:
                sample_srcid_list = select_random_samples(
                                        building,
                                        one_label_dict.keys(),
                                        source_sample_num,
                                        self.use_cluster_flag)
                self.learning_srcids += sample_srcid_list
            one_tagsets_dict = self.building_tagsets_dict[building]
            self.tagsets_dict.update(one_tagsets_dict)
            """
            for srcid, tagsets in one_tagsets_dict.items():
                point_tagset = 'none'
                for tagset in tagsets:
                    if tagset in point_tagsets:
                        point_tagset = tagset
                        break
                self.point_dict[srcid] = point_tagset
            """

        self.phrase_dict = make_phrase_dict(self.sentence_dict,
                                            self.label_dict)
        # validation
        for srcid in self.target_srcids:
            assert srcid in self.tagsets_dict


    def update_model(self, srcids):
        self.learning_srcids += srcids
        self.char2ir.update_model(srcids)
        #phrases_pred = self.char2ir.predict(srcids)
        #self.ir2tagsets.update_phrases(phrases_pred)
        self.ir2tagsets.update_model(srcids)


    def predict_tags(self, target_srcids=None):
        if not target_srcids:
            target_srcids = self.target_srcids
        pred_bios = self.char2ir.predict(target_srcids)
        return pred_bios

    def predict(self, target_srcids=None):
        if not target_srcids:
            target_srcids = self.target_srcids
        pred_bios = self.char2ir.predict(target_srcids)
        pred_phrases = make_phrase_dict(self.sentence_dict, pred_bios)
        phrases = {srcid: self.phrase_dict[srcid]
                          if srcid in self.learning_srcids
                          else pred_phrases[srcid]
                   for srcid in target_srcids}
        self.ir2tagsets.update_phrases(phrases)
        pred = self.ir2tagsets.predict(target_srcids)
        return pred

    def predict_proba(self, target_srcids=None):
        if not target_srcids:
            target_srcids = self.target_srcids
        phrases = self.char2ir.predict(target_srcids)
        phrases_proba = self.char2ir.predict_proba(target_srcids)
        self.ir2tagsets.update_phrases(phrases)
        proba = self.ir2tagsets.predict_proba(target_srcids)
        return proba

    def select_informative_samples(self, sample_num):
        char2ir_num = int(sample_num / 2)
        ir2tagsets_num = sample_num - char2ir_num
        char2ir_srcids = self.char2ir.select_informative_samples(char2ir_num)
        ir2tagsets_srcids = self.ir2tagsets.select_informative_samples(
                                ir2tagsets_num)
        new_srcids = set(char2ir_srcids + ir2tagsets_srcids)
        redundant_srcids = [srcid for srcid in new_srcids
                            if srcid in self.learning_srcids]
        if redundant_srcids:
            print('WARNING: redundant srcids from select_samples(): {0}'
                  .format(redundant_srcids))
        return new_srcids

    def evaluate(self, target_srcids=None):
        if not target_srcids:
            target_srcids = self.target_srcids
        pred = self.predict(target_srcids)
        pred_tags = self.predict_tags(target_srcids)
        hist = {
            'pred_tagsets': pred,
            'pred_tags': pred_tags,
            'learning_srcids': deepcopy(self.learning_srcids)
        }
        self.history.append(hist)












