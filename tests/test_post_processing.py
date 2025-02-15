import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical
from waste_classifier.post_processing import convert_to_trash


def test_convert_to_trash_returns_only_3_categories():
    # Given
    y_pred = np.random.randint(0, 6, 100)
    y = np.random.randint(0, 6, 100)
    y_pred_categorical = to_categorical(y_pred, 6)
    y_categorical = to_categorical(y, 6)

    # When
    y_pred_trash, y_trash = convert_to_trash(y_pred_categorical, y_categorical)

    # Then
    assert y_pred_trash.shape[1] == 3
    assert y_trash.shape[1] == 3


def test_convert_to_trash_converts_cardboard_to_recyclable():
    # Given
    y_pred = np.full((100, 1), 0)
    y = np.full((100, 1), 0)
    y_pred_categorical = to_categorical(y_pred, 6)
    y_categorical = to_categorical(y, 6)

    expected_result = to_categorical(np.full((100, 1), 0), 3)

    # When
    y_pred_trash, y_trash = convert_to_trash(y_pred_categorical, y_categorical)

    # Then
    assert np.array_equal(np.round(y_pred_trash), expected_result)
    assert np.array_equal(np.round(y_trash), expected_result)


def test_convert_to_trash_converts_glass_to_verre():
    # Given
    y_pred = np.full((100, 1), 1)
    y = np.full((100, 1), 1)
    y_pred_categorical = to_categorical(y_pred, 6)
    y_categorical = to_categorical(y, 6)

    expected_result = to_categorical(np.full((100, 1), 1), 3)

    # When
    y_pred_trash, y_trash = convert_to_trash(y_pred_categorical, y_categorical)

    # Then
    assert (np.array_equal(np.round(y_pred_trash), expected_result))
    assert (np.array_equal(np.round(y_trash), expected_result))


def test_convert_to_trash_converts_metal_to_recyclable():
    # Given
    y_pred = np.full((100, 1), 2)
    y = np.full((100, 1), 2)
    y_pred_categorical = to_categorical(y_pred, 6)
    y_categorical = to_categorical(y, 6)

    expected_result = to_categorical(np.full((100, 1), 0), 3)

    # When
    y_pred_trash, y_trash = convert_to_trash(y_pred_categorical, y_categorical)

    # Then
    assert np.array_equal(np.round(y_pred_trash), expected_result)
    assert np.array_equal(np.round(y_trash), expected_result)


def test_convert_to_trash_converts_paper_to_recyclable():
    # Given
    y_pred = np.full((100, 1), 3)
    y = np.full((100, 1), 3)
    y_pred_categorical = to_categorical(y_pred, 6)
    y_categorical = to_categorical(y, 6)

    expected_result = to_categorical(np.full((100, 1), 0), 3)

    # When
    y_pred_trash, y_trash = convert_to_trash(y_pred_categorical, y_categorical)

    # Then
    assert np.array_equal(np.round(y_pred_trash), expected_result)
    assert np.array_equal(np.round(y_trash), expected_result)


def test_convert_to_trash_converts_plastic_to_recyclable():
    # Given
    y_pred = np.full((100, 1), 4)
    y = np.full((100, 1), 4)
    y_pred_categorical = to_categorical(y_pred, 6)
    y_categorical = to_categorical(y, 6)

    expected_result = to_categorical(np.full((100, 1), 0), 3)

    # When
    y_pred_trash, y_trash = convert_to_trash(y_pred_categorical, y_categorical)

    # Then
    assert np.array_equal(np.round(y_pred_trash), expected_result)
    assert np.array_equal(np.round(y_trash), expected_result)


def test_convert_to_trash_converts_trash_to_non_recyclable():
    # Given
    y_pred = np.full((100, 1), 5)
    y = np.full((100, 1), 5)
    y_pred_categorical = to_categorical(y_pred, 6)
    y_categorical = to_categorical(y, 6)

    expected_result = to_categorical(np.full((100, 1), 2), 3)

    # When
    y_pred_trash, y_trash = convert_to_trash(y_pred_categorical, y_categorical)

    # Then
    assert np.array_equal(np.round(y_pred_trash), expected_result)
    assert np.array_equal(np.round(y_trash), expected_result)
