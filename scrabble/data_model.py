from mongoengine import *
import json

with open('config/scrabble_config.json', 'r') as fp:
    config = json.load(fp)
    dbname = config['dbname']

connect(dbname)


class RawMetadata(Document):
    srcid = StringField(required=True)
    building = StringField(required=True)
    metadata = DictField()

class LabeledMetadata(Document):
    srcid = StringField(required=True)
    building = StringField(required=True)
    fullparsing = DictField()
    tagsets = ListField(StringField())
    point_tagset = StringField()
column_names = ['VendorGivenName', 
                 'BACnetName', 
                 'BACnetDescription']

class ResultHistory(Document):
    history = ListField()
    use_brick_flag = BooleanField()
    use_known_tags = BooleanField()
    sample_num_list = ListField()
    target_building = StringField(required=True)
    negative_flag = BooleanField()
    entqs = StringField()
    crfqs = StringField()
    tagset_classifier_type = StringField()
    postfix = StringField()

