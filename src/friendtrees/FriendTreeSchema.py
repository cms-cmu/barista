# Based on https://github.com/CoffeaTeam/coffea/blob/7dd4f863837a6319579f078c9e445c61d9106943/coffea/nanoevents/schemas/nanoaod.py
from coffea.nanoevents.schemas.base import BaseSchema, zip_forms

class behavior_descriptor:
    def __get__(self, instance, owner):
        """Behaviors necessary to implement this schema (works as class property and instance property)"""
        from coffea.nanoevents.methods import nanoaod
        return nanoaod.behavior

class FriendTreeSchema(BaseSchema):
    """Basic multiclassifier friend tree schema"""
    behavior = behavior_descriptor()

    def __init__(self, base_form, name=''):
        super().__init__(base_form)
        self.mixins = {}
        if isinstance(self._form["contents"], dict):
            # coffea 0.7: contents is a dict {name: form}
            self._form["contents"] = self._build_collections(self._form["contents"])
        else:
            # coffea 2025: contents is a list parallel to fields
            branch_forms = dict(zip(self._form["fields"], self._form["contents"]))
            output = self._build_collections(branch_forms)
            self._form["fields"] = list(output.keys())
            self._form["contents"] = list(output.values())

    def _build_collections(self, branch_forms):
        for k in branch_forms:
            if k.endswith('event'):
                name = '_'.join(k.split('_')[:-1])

            elif k.startswith('w'):
                name = k

        mixin = self.mixins.get(name, "NanoCollection")

        # simple collection
        output = {}


        #
        #  Gets the branches with names "name_*"
        #
        _tmpDict = {
            k[len(name) + 1 :] : branch_forms[k]
            for k in branch_forms
            if k.startswith(name + "_")
        }

        #
        #  Add the branches with name "name"
        #
        if name in branch_forms:
            _tmpDict[ name ] = branch_forms[name]

        output[name] = zip_forms(
            _tmpDict,
            name,
            record_name=mixin,
        )
        output[name].setdefault("parameters", {})
        output[name]["parameters"].update({"collection_name": name})

        return output
