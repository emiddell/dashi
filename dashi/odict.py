

from collections import MutableMapping

class OrderedDict(MutableMapping):
    """
        a dictionary that keeps tracks of the order in which 
        keys are added

        based on: http://code.activestate.com/recipes/496761
    """ 
    
    def __init__(self, data=None):
        self._keys = []
        self._data = {}

        if data is not None:
            if hasattr(data, 'items'): # dictionary
                items = list(data.items())
            else:                      # list of tuples
                items = list(data)
            for i in range(len(items)):
                length = len(items[i])
                if length != 2:
                    raise ValueError('dictionary update sequence element '
                        '#%d has length %d; 2 is required' % (i, length))
                self._keys.append(items[i][0])
                self._data[items[i][0]] = items[i][1]
    
    def __len__(self):
        return len(self._keys)
        
        
    def __setitem__(self, key, value):
        if key not in self._data:
            self._keys.append(key)
        self._data[key] = value
        
        
    def __getitem__(self, key):
        return self._data[key]
    
    
    def __delitem__(self, key):
        del self._data[key]
        self._keys.remove(key)
        
        
    def keys(self):
        return list(self._keys)

    def __iter__(self):
        for key in self._keys:
            yield key

    def __repr__(self):
        result = []
        for key in self._keys:
            result.append('%s: %s' % (repr(key), repr(self._data[key])))
        if len(self._keys) < 4:
            return ''.join(['{', ', '.join(result), '}'])
        else:
            return ''.join(['{', ',\n '.join(result), '}'])

    
    
    def copy(self):
        copyDict = OrderedDict()
        copyDict._data = self._data.copy()
        copyDict._keys = self._keys[:]
        return copyDict


