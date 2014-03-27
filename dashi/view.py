import weakref

class View(object):
    def __init__(self, model):
        self._mvc_model = weakref.ref(model)

    def update(self):
        raise NotImplementedError("update method has not been implemented")

class Model(object):
    def __init__(self):
        self._mvc_views = set()

    def _mvc_add_view(self, view):
        " add a view/listener "
        if not isinstance(view, View):
            raise ValueError("argument must be of type 'View'")
        self._mvc_views.add(weakref.ref(view))

    def _mvc_remove_view(self, view):
        " remove a view/listener"
        ref = weakref.ref(view)
        if ref in self._mvc_views:
            self._mvc_views.remove(ref)
        else:
            raise ValueError("given view is not registered with this model")

    def _mvc_notify(self):
        " notify each registered view"
        for viewref in self._mvc_views:
            viewref().notify()


