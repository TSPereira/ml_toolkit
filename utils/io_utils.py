import json
from datetime import datetime


class JsonConvert(object):
    mappings = {}


    @classmethod
    def class_mapper(cls, d):
        for keys, cls_ in cls.mappings.items():
            if keys.issuperset(d.keys()) & bool(d):  # are all required arguments present?
                if cls_ == datetime:
                    return datetime(**d)
                else:
                    cls_ = cls_()
                    cls_.__dict__ = d
                    return cls_
        else:
            return d


    @classmethod
    def complex_handler(cls, obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            raise TypeError('Object of type %s with value of %s is not JSON serializable' % (type(obj), repr(obj)))


    @classmethod
    def register(cls, cls_):
        cls.mappings[frozenset(tuple(attr for attr in cls_().__dict__.keys()))] = cls_
        return cls_


    @classmethod
    def to_json(cls, obj):
        return json.dumps(obj.__dict__, default=cls.complex_handler, indent=4)


    @classmethod
    def from_json(cls, json_str):
        return json.loads(json_str, object_hook=cls.class_mapper)


    @classmethod
    def to_file(cls, obj, path):
        with open(path, 'w') as jfile:
            jfile.writelines([cls.to_json(obj)])
        return path


    @classmethod
    def from_file(cls, filepath):
        with open(filepath, 'r') as jfile:
            result = cls.from_json(jfile.read())
        return result


@JsonConvert.register
class datetime(datetime):
    def __new__(cls, year=1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0, tzinfo=None, *, fold=0):
        return super().__new__(cls, year=year, month=month, day=day, hour=hour, minute=minute, second=second,
                               microsecond=microsecond, tzinfo=tzinfo, fold=fold)


    @property
    def __dict__(self):
        return dict(year=self.year, month=self.month, day=self.day, hour=self.hour, minute=self.minute,
                    second=self.second, microsecond=self.microsecond, tzinfo=self.tzinfo, fold=self.fold)


    def __add__(self, other):
        dt = super().__add__(other)
        return datetime(year=dt.year, month=dt.month, day=dt.day, hour=dt.hour, minute=dt.minute,
                        second=dt.second, microsecond=dt.microsecond, tzinfo=dt.tzinfo, fold=dt.fold)
