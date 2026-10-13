
import pytest
import cv2 
import numpy as np 
from crackvision.image_utils import get_image_info, load_image
from crackvision.batch_utils import inspect_images
from crackvision.inference_utils import prepare_image_for_inference
from crackvision.inference_utils import (
    predict_image,
    prepare_image_for_inference,
)


def test_missing_image_raises_error(tmp_path):
    missing_path = tmp_path / "missing.jpg"

    with pytest.raises(FileNotFoundError):
        load_image(missing_path)


def test_unsupported_extension_raises_error(tmp_path):
    text_path = tmp_path / "notes.txt"
    text_path.write_text("not an image")

    with pytest.raises(ValueError, match="Unsupported image format"):
        load_image(text_path)

def test_load_valid_image(tmp_path):
    image_path = tmp_path / "steel.jpg"

    # Create a small synthetic image without needing the dataset.
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    image[:, :, 2] = 255

    assert cv2.imwrite(str(image_path), image)

    loaded_image = load_image(image_path)

    assert isinstance(loaded_image, np.ndarray)
    assert loaded_image.shape == (32, 48, 3)
    assert loaded_image.dtype == np.uint8



def test_get_image_info_for_color_image():
    image = np.zeros((32, 48, 3), dtype=np.uint8)

    info = get_image_info(image)

    assert info == {
        "width": 48,
        "height": 32,
        "channels": 3,
        "dtype": "uint8",
    }


def test_get_image_info_for_grayscale_image():
    image = np.zeros((20, 30), dtype=np.uint8)

    info = get_image_info(image)

    assert info["width"] == 30
    assert info["height"] == 20
    assert info["channels"] == 1


def test_get_image_info_rejects_invalid_dimensions():
    image = np.zeros((2, 3, 4, 5), dtype=np.uint8)

    with pytest.raises(ValueError, match="2D or 3D"):
        get_image_info(image)


def test_get_image_info_rejects_empty_image():
    image = np.zeros((0, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="non-zero width and height"):
        get_image_info(image)


def test_get_image_info_rejects_string_array():
    image = np.array([["not", "pixels"]])

    with pytest.raises(TypeError, match="numeric pixel values"):
        get_image_info(image)



def test_inspect_images_returns_metadata_in_order(tmp_path):
    first_path = tmp_path / "steel_01.jpg"
    second_path = tmp_path / "steel_02.png"

    first_image = np.zeros((32, 48, 3), dtype=np.uint8)
    second_image = np.zeros((20, 30), dtype=np.uint8)

    assert cv2.imwrite(str(first_path), first_image)
    assert cv2.imwrite(str(second_path), second_image)

    results = inspect_images([first_path, second_path])

    assert len(results) == 2
    assert results[0]["path"] == str(first_path)
    assert results[0]["width"] == 48
    assert results[0]["height"] == 32
    assert results[1]["path"] == str(second_path)
    assert results[1]["width"] == 30
    assert results[1]["height"] == 20
    assert results[1]["channels"] == 1


def test_inspect_images_fails_on_missing_image(tmp_path):
    valid_path = tmp_path / "steel.jpg"
    missing_path = tmp_path / "missing.jpg"

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    assert cv2.imwrite(str(valid_path), image)

    with pytest.raises(FileNotFoundError):
        inspect_images([valid_path, missing_path])




def test_prepare_grayscale_image_for_inference():
    image = np.zeros((20, 30), dtype=np.uint8)

    prepared = prepare_image_for_inference(image)

    assert prepared.shape == (20, 30, 3)
    assert prepared.dtype == np.uint8


def test_prepare_bgr_image_for_inference():
    image = np.zeros((20, 30, 3), dtype=np.uint8)

    prepared = prepare_image_for_inference(image)

    assert prepared.shape == (20, 30, 3)
    assert prepared is image


def test_prepare_bgra_image_for_inference():
    image = np.zeros((20, 30, 4), dtype=np.uint8)

    prepared = prepare_image_for_inference(image)

    assert prepared.shape == (20, 30, 3)
    assert prepared.dtype == np.uint8




def test_prepare_image_rejects_invalid_dimensions():
    image = np.zeros((10, 20, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="Expected a grayscale, BGR, or BGRA"):
        prepare_image_for_inference(image)


def test_prepare_image_rejects_empty_image():
    image = np.zeros((0, 20, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="non-zero width and height"):
        prepare_image_for_inference(image)


def test_predict_image_passes_expected_arguments():
    class FakeModel:
        def predict(self, **kwargs):
            self.received = kwargs
            return ["fake_prediction"]

    model = FakeModel()
    image = np.zeros((20, 30), dtype=np.uint8)

    result = predict_image(model, image)

    assert result == "fake_prediction"
    assert model.received["source"].shape == (20, 30, 3)
    assert model.received["imgsz"] == 640
    assert model.received["conf"] == 0.25
    assert model.received["save"] is False
    assert model.received["verbose"] is False


def test_predict_image_rejects_invalid_confidence():
    class FakeModel:
        def predict(self, **kwargs):
            raise AssertionError("Model should not be called")

    image = np.zeros((20, 30, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="confidence"):
        predict_image(FakeModel(), image, confidence=1.5)


def test_predict_image_rejects_empty_results():
    class FakeModel:
        def predict(self, **kwargs):
            return []

    image = np.zeros((20, 30, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="no prediction results"):
        predict_image(FakeModel(), image)



def test_predict_image_rejects_invalid_image_size():
    class FakeModel:
        def predict(self, **kwargs):
            raise AssertionError("Model should not be called")

    image = np.zeros((20, 30, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="image_size"):
        predict_image(FakeModel(), image, image_size=0)


def test_predict_image_rejects_non_numeric_image():
    class FakeModel:
        def predict(self, **kwargs):
            raise AssertionError("Model should not be called")

    image = np.array([["invalid pixels"]])

    with pytest.raises(TypeError, match="numeric pixel values"):
        predict_image(FakeModel(), image)
