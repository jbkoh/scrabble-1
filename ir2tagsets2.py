import os
from uuid import uuid4
from operator import itemgetter

import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, chi2, SelectPercentile, SelectKBest
from sklearn.pipeline import Pipeline
from scipy.sparse import vstack, csr_matrix, hstack, issparse, coo_matrix, \
    lil_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.problem_transform import LabelPowerset, ClassifierChain, \
    BinaryRelevance
from sklearn.metrics import precision_recall_fscore_support

from base_scrabble import BaseScrabble
from common import *
from brick_parser import pointTagsetList        as  point_tagsets,\
                         locationTagsetList     as  location_tagsets,\
                         equipTagsetList        as  equip_tagsets,\
                         pointSubclassDict      as  point_subclass_dict,\
                         equipSubclassDict      as  equip_subclass_dict,\
                         locationSubclassDict   as  location_subclass_dict,\
                         tagsetTree             as  tagset_tree

tagset_list = point_tagsets + location_tagsets + equip_tagsets
tagset_list.append('networkadapter')


def tree_flatter(tree, init_flag=True):
    branches_list = list(tree.values())
    d_list = list(tree.keys())
    for branches in branches_list:
        for branch in branches:
            added_d_list = tree_flatter(branch)
            d_list = [d for d in d_list if d not in added_d_list]\
                    + added_d_list
    return d_list

def extend_tree(tree, k, d):
    for curr_head, branches in tree.items():
        if k==curr_head:
            branches.append(d)
        for branch in branches:
            extend_tree(branch, k, d)

def calc_leaves_depth(tree, d=dict(), depth=0):
    curr_depth = depth + 1
    for tagset, branches in tree.items():
        if d.get(tagset):
            d[tagset] = max(d[tagset], curr_depth)
        else:
            d[tagset] = curr_depth
        for branch in branches:
            new_d = calc_leaves_depth(branch, d, curr_depth)
            for k, v in new_d.items():
                if d.get(k):
                    d[k] = max(d[k], v)
                else:
                    d[k] = v
    return d

def augment_tagset_tree(tagsets):
    global tagset_tree
    global subclass_dict
    for tagset in set(tagsets):
        if '-' in tagset:
            classname = tagset.split('-')[0]
            #tagset_tree[classname].append({tagset:[]})
            extend_tree(tagset_tree, classname, {tagset:[]})
            subclass_dict[classname].append(tagset)
            subclass_dict[tagset] = []
        else:
            if tagset not in subclass_dict.keys():
                classname = tagset.split('_')[-1]
                subclass_dict[classname].append(tagset)
                subclass_dict[tagset] = []
                extend_tree(tagset_tree, classname, {tagset:[]})


# TODO: Initialize tagset_list

class Ir2Tagsets(BaseScrabble):
    """docstring for Ir2Tagsets"""

    def __init__(self,
                 target_building,
                 target_srcids,
                 building_label_dict,
                 building_sentence_dict,
                 building_tagsets_dict,
                 source_buildings=[],
                 source_sample_num_list=[],
                 conf={
                     'use_cluster_flag': False,
                     # 'use_brick_flag': False
                 }):
        self.source_buildings = source_buildings
        self.target_building = target_building
        if self.target_building not in self.source_buildings:
            self.source_buildings.append(self.target_building)
            self.source_sample_num_list += 0
        self.source_sample_num_list = source_sample_num_list
        self.use_cluster_flag = conf['use_cluster_flag']
        # self.use_brick_flag = conf['use_brick_flag']
        self.building_tagsets_dict = building_tagsets_dict
        self.use_brick_flag = False  # Temporarily disable it
        self.building_sentence_dict = building_sentence_dict
        self.building_label_dict = building_label_dict
        self.target_srcids = target_srcids
        self._init_data()
        self.ts2ir = None
        self.ts_feature_filename = 'TS_Features/features.pkl'

        if 'eda_flag' in conf:
            self.eda_flag = conf['eda_flag'], 
        else:
            self.eda_flag = False
        if 'use_brick_flag' in conf:
            self.use_brick_flag = conf['use_brick_flag']
        else:
            self.use_brick_flag = True
        if 'n_jobs' in conf:
            self.n_jobs = conf['n_jobs']
        else:
            n_jobs = 6
        if 'ts_flag' in conf:
            self.ts_flag = conf['ts_flag']
        else:
            self.ts_flag = False
        if 'negative_flag' in conf:
            self.negative_flag = conf['negative_flag']
        else:
            self.negative_flag = True
        if 'tagset_classifier_type' in conf:
            self.tagset_classifier_type = conf['tagset_classifier_type']
        else: 
            self.tagset_classifier_type = 'StructuredCC'
        if 'n_estimators' in conf:
            self.n_estimators = conf['n_estimators']
        else:
            self.n_estimators = 50 # TODO: Find the proper value
        if 'vectorizer_type' in conf:
            self.vectorizer_type = conf['vectorizer_type']
        else:
            self.vectorizer_type = 'tfidf'
        self._init_brick()

    def _init_brick(self):
        self.tagset_list = point_tagsets + location_tagsets + equip_tagsets
        self.tagset_list.append('networkadapter')

        self.subclass_dict = dict()
        self.subclass_dict.update(point_subclass_dict)
        self.subclass_dict.update(equip_subclass_dict)
        self.subclass_dict.update(location_subclass_dict)
        self.subclass_dict['networkadapter'] = list()
        self.subclass_dict['unknown'] = list()
        self.subclass_dict['none'] = list()
        

    def _init_data(self):
        self.learning_srcids = []
        self.sentence_dict = {}
        self.label_dict = {}
        self.tagsets_dict = {}
        self.phrase_dict = {}
        self.point_dict = {}

        for building, source_sample_num in zip(self.source_buildings,
                                               self.source_sample_num_list):
            self.sentence_dict.update(self.building_sentence_dict[building])
            one_label_dict = self.building_label_dict[building]
            self.label_dict.update(one_label_dict)

            sample_srcid_list = select_random_samples(building,
                                                      one_label_dict.keys(),
                                                      source_sample_num,
                                                      self.use_cluster_flag)
            self.learning_srcids += sample_srcid_list
            one_tagsets_dict = self.building_tagsets_dict[building]
            self.tagsets_dict.update(one_tagsets_dict)
            for srcid, tagsets in one_tagsets_dict:
                point_tagset = 'none'
                for tagset in tagsets:
                    if tagset in point_tagsets:
                        point_tagset = tagset
                        break
                self.point_dict[srcid] = point_tagset

        self.phrase_dict = make_phrase_dict(self.sentence_dict, 
                                            self.label_dict)
    def _extend_tagset_list(self, new_tagsets):
        self.tagset_list += new_tagsets
        self.tagset_list = list(set(self.tagset_list))

    def update_model(self, srcids):
        self.learning_srcids += srcids
        self._extend_tagset_list([self.tagsets_dict[srcid] 
                            for srcid in self.learning_srcids + 
                                         self.target_srcids])
        augment_tagset_tree(tagset_list)
        self._build_tagset_classifier(self.learning_srcids,
                                      self.target_srcids,
                                      validation_srcids=[])

    # ESSENTIAL
    def select_informative_samples(self, sample_num):
        pass

    # ESSENTIAL
    def learn_auto(self, iter_num=1):
        """Learn from the scratch to the end.
        """
        pass

    def _augment_phrases_with_ts(self, phrase_dict, srcids, ts2ir):
        with open(ts_feature_filename, 'rb') as fp:
            ts_features = pickle.load(fp, encoding='bytes')
        ts_tags_pred = ts2ir.predict(ts_features, srcids)

        tag_binarizer = ts2ir.get_binarizer()
        pred_tags_list = tag_binarizer.inverse_transform(ts_tags_pred)

        for srcid, pred_tags in zip(srcids, pred_tags_list):
            phrase_dict[srcid] += list(pred_tags)
        return phrase_dict

    def _predict_and_proba(self, target_srcids):
        #return self.tagset_classifier, self.tagset_vectorizer, self.tagset_binarizer, self.ts2ir
        phrase_dict = {srcid: self.phrase_dict[srcid] 
                       for srcid in target_srcids}
        if ts2ir:
            phrase_dict = self._augment_phrases_with_ts(phrase_dict, srcids, ts2ir)
        doc = [' '.join(phrase_dict[srcid]) for srcid in srcids]
        vect_doc = self.tagset_vectorizer.transform(doc) # should this be fit_transform?

        certainty_dict = dict()
        tagsets_dict = dict()
        pred_mat = self.tagset_classifier.predict(vect_doc)
        prob_mat = self.tagset_classifier.predict_proba(vect_doc)
        if not isinstance(pred_mat, np.ndarray):
            try:
                pred_mat = pred_mat.toarray()
            except:
                pred_mat = np.asarray(pred_mat)
        pred_tagsets_dict = dict()
        pred_certainty_dict = dict()
        pred_point_dict = dict()
        for i, (srcid, pred) in enumerate(zip(srcids, pred_mat)):
        #for i, (srcid, pred, point_pred) \
                #in enumerate(zip(srcids, pred_mat, point_mat)):
            pred_tagsets_dict[srcid] = binarizer.inverse_transform(\
                                            np.asarray([pred]))[0]
            #pred_tagsets_dict[srcid] = list(binarizer.inverse_transform(pred)[0])
            #pred_point_dict[srcid] = point_tagsets[point_pred]
            #pred_vec = [prob[i][0] for prob in prob_mat]
            #pred_certainty_dict[srcid] = pred_vec
            pred_certainty_dict[srcid] = 0
        pred_certainty_dict = OrderedDict(sorted(pred_certainty_dict.items(), \
                                                 key=itemgetter(1), reverse=True))
        logging.info('Finished prediction')
        return pred_tagsets_dict, pred_certainty_dict

    def predict(self, target_srcids=None):
        if not target_srcids:
            target_srcids =self.target_srcids
        pred, _ =self._predict_and_proba(target_srcids)
        return pred
    
    def predict_proba(self, target_srcids=None):
        if not target_srcids:
            target_srcids =self.target_srcids
        _, proba =self._predict_and_proba(target_srcids)
        return proba

    def _build_point_classifier(self):
        # TODO: Implement this later if needed
        #       Currently, just collected garbages.
        point_classifier = RandomForestClassifier(
                               n_estimators=self.n_estimators,
                               n_jobs=n_jobs)
        # Dataset only for points. Just for testing.
        learning_point_dict = dict()
        for srcid, tagsets in chain(learning_truths_dict.items(),
                                    validation_truths_dict.items()):
            point_tagset = 'none'
            for tagset in tagsets:
                if tagset in point_tagsets:
                    point_tagset = tagset
                    break
            learning_point_dict[srcid] = point_tagset
        learning_point_dict['dummy'] = 'unknown'
        point_truths_dict = dict()
        point_srcids = list()
        for srcid in learning_srcids:
            truths = learning_truths_dict[srcid]
            point_tagset = None
            for tagset in truths:
                if tagset in point_tagsets:
                    point_tagset = tagset
                    break
            if point_tagset:
                point_truths_dict[srcid] = point_tagset
                point_srcids.append(srcid)

        try:
            point_truth_mat = [point_tagsets.index(point_truths_dict[srcid]) \
                               for srcid in point_srcids]
            point_vect_doc = np.vstack([learning_vect_doc[learning_srcids.index(srcid)]
                                        for srcid in point_srcids])
        except:
            pdb.set_trace()


    def _augment_with_ts(self, test_phrases_dict,):
        # TODO: Implement below
        ts_learning_srcids = list()
        learning_tags_dict = dict([(srcid, splitter(tagset)) for srcid, tagset
                                   in learning_point_dict.items()])
        tag_binarizer = MultiLabelBinarizer()
        tag_binarizer.fit(map(splitter, learning_point_dict.values()))
        with open(self.ts_feature_filename, 'rb') as fp:
            ts_features = pickle.load(fp, encoding='bytes')
        new_ts_features = list()
        for ts_feature in ts_features:
            feats = ts_feature[0]
            srcid = ts_feature[2]
            if srcid in learning_srcids + validation_srcids:
                point_tagset = learning_point_dict[srcid]
                point_tags = point_tagset.split('_')
                point_vec = tag_binarizer.transform([point_tags])
                new_feature = [feats, point_vec, srcid]
                new_ts_features.append(new_feature)
            elif srcid in test_srcids:
                new_ts_features.append(ts_feature)
        ts_features = new_ts_features

        self.ts2ir = TimeSeriesToIR(mlb=tag_binarizer)
        self.ts2ir.fit(ts_features, learning_srcids, validation_srcids, learning_tags_dict)
        learning_ts_tags_pred = self.ts2ir.predict(ts_features, learning_srcids)
        for srcid, ts_tags in zip(learning_srcids, \
                                  tag_binarizer.inverse_transform(
                                      learning_ts_tags_pred)):
            #learning_phrase_dict[srcid] += list(ts_tags)
            ts_srcid = srcid + '_ts'
            learning_phrase_dict[ts_srcid] = learning_phrase_dict[srcid]\
                                                + list(ts_tags)
            ts_learning_srcids.append(ts_srcid)
            learning_truths_dict[ts_srcid] = learning_truths_dict[srcid]

        test_ts_tags_pred = self.ts2ir.predict(ts_features, test_srcids)
        for srcid, ts_tags in zip(test_srcids, \
                                  tag_binarizer.inverse_transform(
                                      test_ts_tags_pred)):
            #ts_srcid = srcid + '_ts'
            #test_phrase_dict[ts_srcid] = test_phrase_dict[srcid] + list(ts_tags)
            #test_srcids .append(ts_srcid) # TODO: Validate if this works.
            test_phrase_dict[srcid] += list(ts_tags)

    def _augment_negative_examples(self):
        negative_doc = []
        negative_srcids = []
        negative_truths_dict = {}
        for srcid in learning_srcids:
            true_tagsets = list(set(learning_truths_dict[srcid]))
            sentence = learning_phrase_dict[srcid]
            for tagset in true_tagsets:
                negative_srcid = srcid + ';' + gen_uuid()
                removing_tagsets = set()
                new_removing_tagsets = set([tagset])
                removing_tags = []
                negative_tagsets = list(filter(tagset.__ne__, true_tagsets))
                i = 0
                while len(new_removing_tagsets) != len(removing_tagsets):
                    i += 1
                    if i>5:
                        pdb.set_trace()
                    removing_tagsets = deepcopy(new_removing_tagsets)
                    for removing_tagset in removing_tagsets:
                        removing_tags += removing_tagset.split('_')
                    for negative_tagset in negative_tagsets:
                        for tag in removing_tags:
                            if tag in negative_tagset.split('_'):
                                new_removing_tagsets.add(negative_tagset)
                negative_sentence = [tag for tag in sentence if\
                                     tag not in removing_tags]
                for tagset in removing_tagsets:
                    negative_tagsets = list(filter(tagset.__ne__,
                                                   negative_tagsets))

    #            negative_sentence = [word for word in sentence \
    #                                 if word not in tagset.split('_')]
                negative_doc.append(' '.join(negative_sentence))
                negative_truths_dict[negative_srcid] = negative_tagsets
                negative_srcids.append(negative_srcid)
        for i in range(0,50):
            # Add empty examples
            negative_srcid = gen_uuid()
            negative_srcids.append(negative_srcid)
            negative_doc.append('')
            negative_truths_dict[negative_srcid] = []
            
    def _augment_brick_samples(self):
        brick_truths_dict = dict()
        brick_srcids = []
        brick_doc = []
        logging.info('Start adding Brick samples')
        #brick_copy_num = int(len(learning_phrase_dict) * 0.04)
        #if brick_copy_num < 4:
        #brick_copy_num = 4
        #brick_copy_num = 2
        brick_copy_num = 6
        #brick_truths_dict = dict((gen_uuid(), [tagset]) \
        #                          for tagset in tagset_list\
        #                          for j in range(0, brick_copy_num))
        #for learning_srcid, true_tagsets in learning_truths_dict.items():
        #    for true_tagset in set(true_tagsets):
        #        brick_truths_dict[gen_uuid()] = [true_tagset]
#
        #brick_srcids = list(brick_truths_dict.keys())
        #brick_doc = [brick_truths_dict[tagset_id][0].replace('_', ' ')
        #                 for tagset_id in brick_srcids]
        brick_truths_dict = dict()
        brick_doc = list()
        brick_srcids = list()
        for tagset in tagset_list:
            for j in range(0, brick_copy_num):
                #multiplier = random.randint(2, 6)
                srcid = 'brick;' + gen_uuid()
                brick_srcids.append(srcid)
                brick_truths_dict[srcid] = [tagset]
                tagset_doc = list()
                for tag in tagset.split('_'):
                    tagset_doc += [tag] * random.randint(1,2)
                brick_doc.append(' '.join(tagset_doc))

        """
        if eda_flag:
            for building in set(building_list + [target_building]):
                for i in range(0, brick_copy_num):
                    for tagset in tagset_list:
                        brick_srcid = gen_uuid()
                        brick_srcids.append(brick_srcid)
                        brick_truths_dict[brick_srcid] = [tagset]
                        tags  = tagset.split('_') + \
                                [building + '#' + tag for tag in tagset.split('_')]
                        brick_doc.append(' '.join(tags))
        """
        logging.info('Finished adding Brick samples')

    def _augment_eda(self):
        if eda_flag:
            unlabeled_phrase_dict = make_phrase_dict(\
                                        test_sentence_dict, \
                                        test_token_label_dict, \
                                        {target_building:test_srcids},\
                                        False)
            prefixer = build_prefixer(target_building)
            unlabeled_target_doc = [' '.join(\
                                    map(prefixer, unlabeled_phrase_dict[srcid]))\
                                    for srcid in test_srcids]
#        unlabeled_vect_doc = - tagset_vectorizer\
#                               .transform(unlabeled_target_doc)
            unlabeled_vect_doc = np.zeros((len(test_srcids), \
                                           len(tagset_vectorizer.vocabulary_)))
            test_doc = [' '.join(unlabeled_phrase_dict[srcid])\
                             for srcid in test_srcids]
            test_vect_doc = tagset_vectorizer.transform(test_doc).toarray()
            for building in source_target_buildings:
                if building == target_building:
                    added_test_vect_doc = - test_vect_doc
                else:
                    added_test_vect_doc = test_vect_doc
                unlabeled_vect_doc = np.hstack([unlabeled_vect_doc,\
                                                added_test_vect_doc])

        if eda_flag:
            learning_vect_doc = tagset_vectorizer.transform(learning_doc +
                                                            negative_doc).todense()
            learning_srcids += negative_srcids
            new_learning_vect_doc = deepcopy(learning_vect_doc)
            for building in source_target_buildings:
                building_mask = np.array([1 if find_key(srcid.split(';')[0],\
                                                       total_srcid_dict,\
                                                       check_in) == building
                                          else 0 for srcid in learning_srcids])
                new_learning_vect_doc = np.hstack([new_learning_vect_doc] \
                                     + [np.asmatrix(building_mask \
                                        * np.asarray(learning_vect)[0]).T \
                                    for learning_vect \
                                        in learning_vect_doc.T])
            learning_vect_doc = new_learning_vect_doc
            if use_brick_flag:
                new_brick_srcids = list()
                new_brick_vect_doc = np.array([])\
                        .reshape((0, len(tagset_vectorizer.vocabulary) \
                                  * (len(source_target_buildings)+1)))
                brick_vect_doc = tagset_vectorizer.transform(brick_doc).todense()
                for building in source_target_buildings:
                    prefixer = lambda srcid: building + '-' + srcid
                    one_brick_srcids = list(map(prefixer, brick_srcids))
                    for new_brick_srcid, brick_srcid\
                            in zip(one_brick_srcids, brick_srcids):
                        brick_truths_dict[new_brick_srcid] = \
                                brick_truths_dict[brick_srcid]
                    one_brick_vect_doc = deepcopy(brick_vect_doc)
                    for b in source_target_buildings:
                        if b != building:
                            one_brick_vect_doc = np.hstack([
                                one_brick_vect_doc,
                                np.zeros((len(brick_srcids),
                                          len(tagset_vectorizer.vocabulary)))])
                        else:
                            one_brick_vect_doc = np.hstack([
                                one_brick_vect_doc, brick_vect_doc])
                    new_brick_vect_doc = np.vstack([new_brick_vect_doc,
                                                one_brick_vect_doc])
                    new_brick_srcids += one_brick_srcids
                learning_vect_doc = np.vstack([learning_vect_doc,
                                               new_brick_vect_doc])
                brick_srcids = new_brick_srcids
                learning_srcids += brick_srcids
        
    
    def _build_tagset_classifier(self, 
                                 learning_srcids,
                                 target_srcids,
                                 validation_srcids):

        # Config variables
        # TODO:

        validation_srcids = list(validation_truths_dict.keys())
        learning_srcids = deepcopy(learning_srcids)

        # Update TagSet pool to include TagSets not in Brick.
        # TODO: Maybe this should be done in initialization stage.
        orig_sample_num = len(learning_srcids)
        new_tagset_list = tree_flatter(self.tagset_tree, [])
        new_tagset_list = new_tagset_list + [ts for ts in self.tagset_list \
                                             if ts not in new_tagset_list]
        self.tagset_list = new_tagset_list
        self.tagset_binarizer = MultiLabelBinarizer(self.tagset_list)
        self.tagset_binarizer.fit([self.tagset_list])
        assert self.tagset_list == self.tagset_binarizer.classes_.tolist()

        learning_tagsets_dict = {srcid: self.tagsets_dict[srcid] 
                                 for srcid in learning_srcids}


        ## Init brick tag_list
        # TODO: Maybe this should be done in initialization stage.
        self.tag_list = list(set(reduce(adder, map(splitter, tagset_list))))

        # All possible vocabularies.
        vocab_dict = dict([(tag, i) for i, tag in enumerate(self.tag_list)])

        # Define Vectorizer
        tokenizer = lambda x: x.split()
        # TODO: We could use word embedding like word2vec here instead.
        if self.vectorizer_type == 'tfidf':
            self.tagset_vectorizer = TfidfVectorizer(tokenizer=tokenizer,
                                                vocabulary=vocab_dict)
        elif self.vectorizer_type == 'meanbembedding':
            self.tagset_vectorizer = MeanEmbeddingVectorizer(tokenizer=tokenizer, 
                                                        vocabulary=vocab_dict)
        elif self.vectorizer_type == 'count':
            self.tagset_vectorizer = CountVectorizer(tokenizer=tokenizer,
                                                vocabulary=vocab_dict)
        else:
            raise Exception('Wrong vectorizer type: {0}'
                                .format(self.vectorizer_type))

        if ts_flag:
            pass
            #TODO: Run self._augment_with_ts()

        ## Transform learning samples
        learning_doc = [' '.join(self.phrase_dict[srcid])
                        for srcid in learning_srcids]
        test_doc = [' '.join(phrase_dict[srcid]) 
                    for srcid in test_srcids]

        ## Augment with negative examples.
        if self.negative_flag:
            pass
            #TODO: self._augment_negative_examples()


        ## Init Brick samples.
        if self.use_brick_flag:
            pass
            # TODO: self._augment_brick_samples()

        self.tagset_vectorizer.fit(learning_doc + test_doc)# + brick_doc)

        # Apply Easy-Domain-Adaptation mechanism. Not useful.
        if self.eda_flag:
            pass
            # TODO: self._augment_eda()
        else:
            # Make TagSet vectors.
            learning_vect_doc = self.tagset_vectorizer.transform(learning_doc)
                                    .todense()

        truth_mat = csr_matrix([self.tagset_binarizer.transform(
                                    [learning_tagsets_dict[srcid]])[0]
                                for srcid in learning_srcids])
        if eda_flag:
            zero_vectors = self.tagset_binarizer.transform(\
                        [[] for i in range(0, unlabeled_vect_doc.shape[0])])
            truth_mat = vstack([truth_mat, zero_vectors])
            learning_vect_doc = np.vstack([learning_vect_doc, unlabeled_vect_doc])

        logging.info('Start learning multi-label classifier')
        ## Learn the classifier. StructuredCC is the default model.
        if tagset_classifier_type == 'RandomForest':
            def meta_rf(**kwargs):
                #return RandomForestClassifier(**kwargs)
                return RandomForestClassifier(n_jobs=n_jobs, n_estimators=150)
            meta_classifier = meta_rf
            params_list_dict = {}
        elif tagset_classifier_type == 'StructuredCC_BACKUP':
            #feature_selector = SelectFromModel(LinearSVC(C=0.001))
            feature_selector = SelectFromModel(LinearSVC(C=0.01, penalty='l1', dual=False))
            base_base_classifier = PassiveAggressiveClassifier(loss='squared_hinge', C=0.1)
            #base_base_classifier = GradientBoostingClassifier()
            #base_base_classifier = RandomForestClassifier()
            base_classifier = Pipeline([('feature_selection',
                                         feature_selector),
                                        ('classification',
                                         base_base_classifier)
                                       ])
            tagset_classifier = StructuredClassifierChain(
                                    base_classifier,
                                    self.tagset_binarizer,
                                    subclass_dict,
                                    self.tagset_vectorizer.vocabulary,
                                    n_jobs,
                                    use_brick_flag)
        elif tagset_classifier_type == 'Project':
            def meta_proj(**kwargs):
                #base_classifier = LinearSVC(C=20, penalty='l1', dual=False)
                base_classifier = SVC(kernel='rbf', C=10, class_weight='balanced')
                #base_classifier = GaussianProcessClassifier()
                tagset_classifier = ProjectClassifier(base_classifier,
                                                               self.tagset_binarizer,
                                                               self.tagset_vectorizer,
                                                               subclass_dict,
                                                               n_jobs)
                return tagset_classifier
            meta_classifier = meta_proj
            params_list_dict = {}

        elif tagset_classifier_type == 'CC':
            def meta_cc(**kwargs):
                feature_selector = SelectFromModel(LinearSVC(C=1))
                #feature_selector = SelectFromModel(LinearSVC(C=0.01, penalty='l1', dual=False))
                base_base_classifier = GradientBoostingClassifier(**kwargs)
                #base_base_classifier = SGDClassifier(loss='modified_huber', penalty='elasticnet')
                #base_base_classifier = PassiveAggressiveClassifier(loss='squared_hinge', C=0.1)
                #base_base_classifier = LogisticRegression()
                #base_base_classifier = RandomForestClassifier(**kwargs)
                base_classifier = Pipeline([('feature_selection',
                                             feature_selector),
                                            ('classification',
                                             base_base_classifier)
                                           ])
                tagset_classifier = ClassifierChain(classifier=base_classifier)
                return tagset_classifier
            meta_classifier = meta_cc
            params_list_dict = {}

        elif tagset_classifier_type == 'StructuredCC':
            def meta_scc(**kwargs):
                feature_selector = SelectFromModel(LinearSVC(C=1))
                #feature_selector = SelectFromModel(LinearSVC(C=0.01, penalty='l1', dual=False))
                base_base_classifier = GradientBoostingClassifier(**kwargs)
                #base_base_classifier = SGDClassifier(loss='modified_huber', penalty='elasticnet')
                #base_base_classifier = PassiveAggressiveClassifier(loss='squared_hinge', C=0.1)
                #base_base_classifier = LogisticRegression()
                #base_base_classifier = RandomForestClassifier(**kwargs)
                base_classifier = Pipeline([('feature_selection',
                                             feature_selector),
                                            ('classification',
                                             base_base_classifier)
                                           ])
                tagset_classifier = StructuredClassifierChain(
                                    base_classifier,
                                    self.tagset_binarizer,
                                    subclass_dict,
                                    self.tagset_vectorizer.vocabulary,
                                    n_jobs,
                                    use_brick_flag,
                                    self.tagset_vectorizer)
                return tagset_classifier
            meta_classifier = meta_scc
            rf_params_list_dict = {
                'n_estimators': [10, 50, 100],
                'criterion': ['gini', 'entropy'],
                'max_features': [None, 'auto'],
                'max_depth': [1, 5, 10, 50],
                'min_samples_leaf': [2,4,8],
                'min_samples_split': [2,4,8]
            }
            gb_params_list_dict = {
                'loss': ['deviance', 'exponential'],
                'learning_rate': [0.1, 0.01, 1, 2],
                'criterion': ['friedman_mse', 'mse'],
                'max_features': [None, 'sqrt'],
                'max_depth': [1, 3, 5, 10],
                'min_samples_leaf': [1,2,4,8],
                'min_samples_split': [2,4,8]
            }
            params_list_dict = gb_params_list_dict
        elif tagset_classifier_type == 'StructuredCC_RF':
            base_classifier = RandomForest()
            tagset_classifier = StructuredClassifierChain(base_classifier,
                                                          self.tagset_binarizer,
                                                          subclass_dict,
                                                          self.tagset_vectorizer.vocabulary,
                                                          n_jobs)
        elif tagset_classifier_type == 'StructuredCC_LinearSVC':
            def meta_scc_svc(**kwargs):
                base_classifier = LinearSVC(loss='hinge', tol=1e-5,\
                                            max_iter=2000, C=2,
                                            fit_intercept=False,
                                            class_weight='balanced')
                tagset_classifier = StructuredClassifierChain(base_classifier,
                                                              self.tagset_binarizer,
                                                              subclass_dict,
                                                              self.tagset_vectorizer.vocabulary,
                                                              n_jobs)
                return tagset_classifier
            params_list_dict = {}
            meta_classifier = meta_scc_svc
        elif tagset_classifier_type == 'OneVsRest':
            base_classifier = LinearSVC(loss='hinge', tol=1e-5,\
                                        max_iter=2000, C=2,
                                        fit_intercept=False,
                                        class_weight='balanced')
            tagset_classifier = OneVsRestClassifier(base_classifier)
        elif tagset_classifier_type == 'Voting':
            def meta_voting(**kwargs):
                return VotingClassifier(self.tagset_binarizer, self.tagset_vectorizer,
                                        tagset_tree, tagset_list)
            meta_classifier = meta_voting
            params_list_dict = {}
        else:
            assert False

        if not isinstance(truth_mat, csr_matrix):
            truth_mat = csr_matrix(truth_mat)

        # TODO: This was for hyper-parameter optimization.
        #       But I disabled it because it's too slow.
        self.tagset_classifier = self._parameter_validation(learning_vect_doc[:orig_sample_num],
                             truth_mat[:orig_sample_num],
                             orig_learning_srcids,
                             params_list_dict, meta_classifier, self.tagset_vectorizer,
                             self.tagset_binarizer, source_target_buildings, eda_flag)

        # Actual fitting.
        if isinstance(self.tagset_classifier, StructuredClassifierChain):
            self.tagset_classifier.fit(learning_vect_doc, truth_mat.toarray(), \
                                  orig_sample_num=len(learning_vect_doc)
                                  - len(brick_srcids))
        else:
            self.tagset_classifier.fit(learning_vect_doc, truth_mat.toarray())
        point_classifier.fit(point_vect_doc, point_truth_mat)
        logging.info('Finished learning multi-label classifier')

    def _parameter_validation(vect_doc, truth_mat, srcids, params_list_dict,\
                             meta_classifier, vectorizer, binarizer, \
                             source_target_buildings, eda_flag):
        # TODO: This is not effective for now. Do I need one?
        #best_params = {'n_estimators': 50, 'criterion': 'entropy', 'max_features': 'auto', 'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 2}
        #best_params = {'criterion': 'entropy'}
        #best_params = {'loss': 'exponential', 'learning_rate': 0.01, 'criterion': 'friedman_mse', 'max_features': None, 'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 2}

            #tagset_classifier = RandomForestClassifier(n_estimators=100,
            #                                           random_state=0,\
            #                                           n_jobs=n_jobs)
        best_params = {'learning_rate':0.1, 'subsample':0.25}
        #best_params = {'C':0.4, 'solver': 'liblinear'}

        return meta_classifier(**best_params) # Pre defined setup.

        #best_params = {'n_estimators': 120, 'n_jobs':7}
        #return meta_classifier(**best_params)

        token_type = 'justseparate'
        results_dict = dict()
        for key, values in params_list_dict.items():
            results_dict[key] = {'ha': [0]*len(values),
                                 'a': [0]*len(values),
                                 'mf1': [0]*len(values)}
        avg_num = 3
        for i in range(0,avg_num):
            learning_indices = random.sample(range(0, len(srcids)),
                                             int(len(srcids)/2))
            validation_indices = [i for i in range(0, len(srcids))
                                  if i not in learning_indices]
            learning_srcids = [srcids[i] for i
                                        in learning_indices]
            validation_srcids = [srcids[i] for i
                                 in validation_indices]
            for key, values in params_list_dict.items():
                for j, value in enumerate(values):
                    params = {key: value}
                    classifier = meta_classifier(**params)
                    classifier.fit(vect_doc[learning_indices], \
                                   truth_mat[learning_indices].toarray())

                    validation_sentence_dict, \
                    validation_token_label_dict, \
                    validation_truths_dict, \
                    validation_phrase_dict = self.get_multi_buildings_data(\
                                                source_target_buildings, validation_srcids, \
                                                eda_flag, token_type)

                    validation_pred_tagsets_dict, \
                    validation_pred_certainty_dict, \
                    _ = tagsets_prediction(classifier, vectorizer, binarizer, \
                                       validation_phrase_dict, validation_srcids, \
                                       source_target_buildings, eda_flag, None,
                                           ts2ir=None)
                    validation_result = tagsets_evaluation(validation_truths_dict, \
                                                           validation_pred_tagsets_dict, \
                                                           validation_pred_certainty_dict,\
                                                           validation_srcids, \
                                                           None, \
                                                           validation_phrase_dict, \
                                                           debug_flag=False,
                                                           classifier=classifier, \
                                                           vectorizer=vectorizer)
                    results_dict[key]['ha'][j] += validation_result['hierarchy_accuracy']
                    results_dict[key]['a'][j] += validation_result['accuracy']
                    results_dict[key]['mf1'][j] += validation_result['macro_f1']
                    results_dict[key]['macro_f1'][j] += validation_result['macro_f1']
        best_params = dict()
        for key, results in results_dict.items():
            metrics = results_dict[key]['mf1']
            best_params[key] = params_list_dict[key][metrics.index(max(metrics))]
        classifier = meta_classifier(**best_params)
        classifier.fit(vect_doc[learning_indices], \
                       truth_mat[learning_indices].toarray())

        validation_sentence_dict, \
        validation_token_label_dict, \
        validation_truths_dict, \
        validation_phrase_dict = self.get_multi_buildings_data(\
                                    source_target_buildings, validation_srcids, \
                                    eda_flag, token_type)

        validation_pred_tagsets_dict, \
        validation_pred_certainty_dict, \
        _ = tagsets_prediction(classifier, vectorizer, binarizer, \
                           validation_phrase_dict, validation_srcids, \
                           source_target_buildings, eda_flag, None,
                               ts2ir=None)
        validation_result = tagsets_evaluation(validation_truths_dict, \
                                               validation_pred_tagsets_dict, \
                                               validation_pred_certainty_dict,\
                                               validation_srcids, \
                                               None, \
                                               validation_phrase_dict, \
                                               debug_flag=False,
                                               classifier=classifier, \
                                               vectorizer=vectorizer)
        best_ha = validation_result['hierarchy_accuracy']
        best_a = validation_result['accuracy']
        best_mf1 = validation_result['macro_f1']

        return meta_classifier(**best_params)

