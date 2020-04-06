import json
from datetime import datetime


class JsonConvert:
    """
    Class to allow other classes to be serializable to Json Format. Works as a wrapper class.
    Class to be turned serializable must only contain serializable objects. Objects that are not serializable must
    be replaced by a serializable version that is registered by this wrapper. As examples see the "datetime" class
    version in this file

    Object to be converted by this wrapper need to implement a __dict__ property (classes that inherit directly from
    "object" have this property built-in and to be initializable without arguments passed. All instance attributes
    should be initialized in the __init__ function

    Instances of classes that don't have attributes will not be able to be reconverted back into the original object.


    ATTENTION: Classes that implement same superset of attributes might get confused todo review way to distinguish

    Usage:
        To convert one instance of the class latter, one only needs to use the classmethods provided in the class
        >>> @JsonConvert.register
        >>> class Test:
        >>>     def __init__(self, a=1, b=2, c=3):
        >>>         return
        >>>
        >>> inst = Test()
        >>> inst_as_json = JsonConvert.to_json(inst)
        >>> JsonConvert.to_file(inst, '/path/to/file/')

        To load from a json variable or file one needs to load the class first (so it will be registered in the
        JsonConvert class in the new session) and then use the load classmethods included.
        >>> @JsonConvert.register
        >>> class Test:
        >>>     def __init__(self, a=1, b=2, c=3):
        >>>         return
        >>>
        >>> inst = JsonConvert.from_json(inst_as_json)
        >>> inst1 = JsonConvert.from_file('/path/to/file/')

    """

    mappings = {}

    @classmethod
    def _class_mapper(cls, d):
        """
        Maps attributes from dictionary like structure to registered class
        :param dict d: dictionary with attributes and their values to map back to class
        :return: Registered class filled or original dictionary passed
        """
        for keys, cls_ in cls.mappings.items():
            if keys.issuperset(d.keys()) and bool(d):
                if cls_ == datetime:
                    return datetime(**d)
                else:
                    cls_ = cls_()
                    cls_.__dict__ = d
                    return cls_
        else:
            return d

    @classmethod
    def _complex_handler(cls, obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            raise TypeError('Object of type %s with value of %s is not JSON serializable' % (type(obj), repr(obj)))

    @classmethod
    def register(cls, cls_):
        """
        Save a frozenset of class passed keys (when class is initialized) to class mappings
        :param cls_: Class to register in JsonConvert
        :return: Original class cls_
        todo find way to initialize any class without having to have only kwargs (get args from __code__ and assign
         None for example)
        """
        cls.mappings[frozenset(tuple(attr for attr in cls_().__dict__.keys()))] = cls_
        return cls_

    @classmethod
    def to_json(cls, obj):
        """
        Get object representation as json. obj must have a __dict__ property/attribute that represents the attributes
        of the object
        :param obj: object to get representation for
        :return: json string representation
        """
        return json.dumps(obj.__dict__, default=cls._complex_handler, indent=4)

    @classmethod
    def from_json(cls, json_str):
        """
        Load a json representation of a registered object back into that object structure
        :param str json_str: json string representation of instance of registered object
        :return: Instance of object registered
        """
        return json.loads(json_str, object_hook=cls._class_mapper)

    @classmethod
    def to_file(cls, obj, path):
        """
        Save object representation as json file. obj must have a __dict__ property/attribute that represents the
        attributes of the object
        :param obj: object to get representation for
        :param str path: full path of file to save to
        :return str: path of file
        """
        with open(path, 'w') as jfile:
            jfile.writelines([cls.to_json(obj)])
        return path

    @classmethod
    def from_file(cls, filepath):
        """
        Load a json representation of a registered object back into that object structure
        :param str filepath: full path of file to load from
        :return: Instance of object registered
        """
        with open(filepath, 'r') as jfile:
            result = cls.from_json(jfile.read())
        return result


"""
save mappings with name, attributes, required attrs to create new instance
add additional attribute to json _classname

on loading get _classname, retrieve class required attrs from mapping, use keys to init class and add remaining 
key, values to dict"""


@JsonConvert.register
class datetime(datetime):
    """
    Child class of built-in datetime to register it to JsonConvert
    """
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
