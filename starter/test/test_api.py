import api_classes
import pytest
import pydantic


@pytest.fixture
def item_dict():
    """
    dict representing data row
    we know this one is correct because it is hardcoded in Item class
    """
    return api_classes.Item.Config.schema_extra["example"]


def test_validation_passes_for_correct_objects(item_dict):
    """
    correct item dict should be possible to parse as Item
    """
    item = api_classes.Item.parse_obj(item_dict)
    assert True


def test_item_handles_export_to_json_with_hypens(item_dict):
    """
    correct item when parsed to dict by alias should have correct hyphenated cols
    """
    item = api_classes.Item.parse_obj(item_dict)
    item_dict = item.dict(by_alias=True)

    expected_hyphen_cols = [
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "education-num",
        "marital-status"]
    for field in expected_hyphen_cols:
        assert field in item_dict.keys()


def test_validation_fails_for_incorrect_object(item_dict):
    """
    we should get an exception when passing invalid data
    """
    del item_dict["sex"]

    with pytest.raises(pydantic.error_wrappers.ValidationError) as exc:
        item = api_classes.Item.parse_obj(item_dict)
