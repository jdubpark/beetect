class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    m.new_key = 'Hello world!' <==> m['new_key'] = 'Hello world!'
    del m.new_key <==> del m['new_key']

    m.sub_dict = Map({})
    m.sub_dict.new_key = 'Sub Hellow world!'
    """

    """ https://stackoverflow.com/a/23689767/13086908 """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    """ https://stackoverflow.com/a/32107024/13086908 """
    # def __init__(self, *args, **kwargs):
    #     super(Map, self).__init__(*args, **kwargs)
    #     for arg in args:
    #         if isinstance(arg, dict):
    #             for k, v in arg.items():
    #                 self[k] = v
    #
    #     if kwargs:
    #         for k, v in kwargs.items():
    #             self[k] = v
    #
    # def __getattr__(self, attr):
    #     return self.get(attr)
    #
    # def __setattr__(self, key, value):
    #     self.__setitem__(key, value)
    #
    # def __setitem__(self, key, value):
    #     super(Map, self).__setitem__(key, value)
    #     self.__dict__.update({key: value})
    #
    # def __delattr__(self, item):
    #     self.__delitem__(item)
    #
    # def __delitem__(self, key):
    #     super(Map, self).__delitem__(key)
    #     del self.__dict__[key]
