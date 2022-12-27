import api_classes
import pytest
import pydantic
import main
from fastapi.testclient import TestClient
import pickle


@pytest.fixture
def item_dict():
    """
    dict representing data row
    we know this one is correct because it is hardcoded in Item class
    """
    return api_classes.Item.Config.schema_extra["example"]


@pytest.fixture
def client():
    return TestClient(main.app)


def test_validation_passes_for_correct_objects(item_dict):
    """
    correct item dict should be possible to parse as Item
    """
    item = api_classes.Item.parse_obj(item_dict)
    assert isinstance(item, api_classes.Item)


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
        "marital-status",
    ]
    for field in expected_hyphen_cols:
        assert field in item_dict.keys()


def test_validation_fails_for_incorrect_object(item_dict):
    """
    we should get an exception when passing invalid data
    """
    item_dict = item_dict.copy()
    del item_dict["sex"]

    with pytest.raises(pydantic.error_wrappers.ValidationError):
        api_classes.Item.parse_obj(item_dict)


def test_read_main(client):
    response = client.get("/root")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_post_incorrect(client):

    response = client.post("/predict/result")
    assert response.status_code == 422


def test_post_correct_proba(client, item_dict):
    response = client.post(
        "/predict/probability", json=item_dict, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 200
    classes = set(response.json().keys())
    assert classes == set(["<=50K", ">50K"])


def test_post_correct_result(client, item_dict):
    response = client.post(
        "/predict/result", json=item_dict, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 200
    assert response.json()["prediction"] in ["<=50K", ">50K"]


def test_predictions_coincide_with_model_predictions(client, item_dict):
    response = client.post(
        "/predict/result", json=item_dict, headers={"Content-Type": "application/json"}
    )
    with open("model.pkl", "rb") as f:
        model_wrapper = pickle.load(f)
    model_prediction = model_wrapper.predict_single(item_dict)
    assert response.json()["prediction"] == model_prediction
