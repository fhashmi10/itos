import json


class Config:
    def __init__(self, data, tokenize, train, model):
        self.data = data
        self.tokenize = tokenize
        self.train = train
        self.model = model

    @classmethod
    def from_json(cls, cfg):
        # Creates config from json
        params = json.loads(json.dumps(cfg), object_hook=HelperObject)
        return cls(params.data, params.tokenize, params.train, params.model)


class HelperObject(object):
    # Helper class to convert json into Python object
    def __init__(self, dict_):
        self.__dict__.update(dict_)
