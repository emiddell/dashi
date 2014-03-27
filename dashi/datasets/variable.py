
class Variable(object):
    def __init__(self, vardef=None, bins=None, label=None, transform=None):
        """
           bundles information that one want to store about a variable

           Parameters:
            vardef     specifies this variable, e.g. a path in a hdf fil
            bins       provide a range in which this variable is supposed to be intereting. e.g. can 
                       be passed to d.factory.hist1d
            label      a string decribing the variable. e.g. can be used to generate axes labels
            transform  a callable with one argument through which the data is transformed
        """

        self.bins = bins
        self.label=label # will become the histogram label
        self.vardef = vardef
        self.transform = transform

    def __repr__(self):
        repr = "<Variable "
        if self.bins is not None:
            repr += "bins: " + str(self.bins) 
        else:
            repr += "bins: None"

        if isinstance(self.vardef,str):
            repr += " def: '" + self.vardef + "'"
        elif callable(self.vardef):
            repr += " def: <calculated>" 
        else:
            repr += " def: <unknown>"
        repr += ">"
        return repr
