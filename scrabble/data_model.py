from mongoengine import *
from .common import *

connect('plastering')

class RawMetadata(Document):
    srcid = StringField(required=True)
    building = StringField(required=True)
    metadata = DictField() # This may contain any kind of metadata as a dictionary. There can be mutiple columns of the raw metadata.
#    #example metadata
#    {
#        "VendorGivenName": "ZN",
#        "BACnetDescription": "Zone"
#    }

class LabeledMetadata(Document):
    srcid = StringField(required=True)
    building = StringField(required=True)
    fullparsing = DictField()
#   #example full parsing
#    {
#        "VendorGivenName": [
#            [ "Z", "B_zone"],
#            [ "N", "I_zone"]
#        ],
#        "BACnetDescription": [
#            [ "Z", "B_zone"],
#            [ "o", "I_zone"],
#            [ "n", "I_zone"],
#            [ "e", "I_zone"]
#        ]
#    }

    tagsets = ListField(StringField())
#   #example tagsets
#    ["zone", ...]

    point_tagset = StringField()
#   #example point tagset
#    "zone_temperature_sensor"

column_names = ['VendorGivenName',
                 'BACnetName',
                 'BACnetDescription']

class ResultHistory(Document):
    history = ListField()
    use_brick_flag = BooleanField()
    use_known_tags = BooleanField()
    sample_num_list = ListField()
    source_building_list = ListField()
    target_building = StringField(required=True)
    negative_flag = BooleanField()
    entqs = StringField()
    crfqs = StringField()
    tagset_classifier_type = StringField()
    postfix = StringField()
    task = StringField()
    sequential_type = StringField()
    ts_flag = BooleanField()

