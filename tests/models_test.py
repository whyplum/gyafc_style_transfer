"""
Validate that model's tf graph is successfully built
"""

from src.models import Model, Encoder, GruClassifier
import pytest
from src.utils import all_subclasses, clean_tf_graph


@pytest.mark.parametrize("model", all_subclasses(Model))
@clean_tf_graph
def test_individual_models(model):
    ins = model()
    _ = ins.test()

