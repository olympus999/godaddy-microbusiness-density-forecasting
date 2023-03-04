import pandas as pd
from traitlets import TraitType

class DataFrame(TraitType):
    """A trait for pd.DataFrame."""

    info_text = "pd.DataFrame"

    def validate(self, obj, value):
        if type(value) == pd.DataFrame:
            return value
        self.error(obj, value)