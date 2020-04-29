import post_processing
import numpy as np
from post_processing import to_categorical


def test_convert_to_trash_returns_only_3_categories():
    # Given
    y_pred = np.random.randint(0, 6, 100)
    y = np.random.randint(0, 6, 100)
    y_pred = to_categorical(y_pred, 6)
    y = to_categorical(y, 6)

    # When
    y_pred_trash, y_trash = post_processing.convert_to_trash(y_pred, y)

    # Then
    assert(y_pred_trash.shape[1] == 3)
    assert(y_trash.shape[1] == 3)
    

def test_convert_to_trash_cardboard_converted_to_recyclable():
    # Given
    y_pred = np.full((100, 1), 0)
    y =  np.full((100, 1), 0)
    y_pred = to_categorical(y_pred, 6)
    y = to_categorical(y, 6)

    expected_result = to_categorical(np.full((100, 1), 0), 3)

    # When
    y_pred_trash, y_trash = post_processing.convert_to_trash(y_pred, y)

    # Then
    assert(np.array_equal(np.round(y_pred_trash), expected_result))
    assert(np.array_equal(np.round(y_trash), expected_result))


def test_convert_to_trash_glass_converted_to_verre():
    # Given
    y_pred = np.full((100, 1), 1)
    y =  np.full((100, 1), 1)
    y_pred = to_categorical(y_pred, 6)
    y = to_categorical(y, 6)

    expected_result = to_categorical(np.full((100, 1), 1), 3)

    # When
    y_pred_trash, y_trash = post_processing.convert_to_trash(y_pred, y)

    # Then
    assert(np.array_equal(np.round(y_pred_trash), expected_result))
    assert(np.array_equal(np.round(y_trash), expected_result))


def test_convert_to_trash_metal_converted_to_recyclable():
    # Given
    y_pred = np.full((100, 1), 2)
    y =  np.full((100, 1), 2)
    y_pred = to_categorical(y_pred, 6)
    y = to_categorical(y, 6)

    expected_result = to_categorical(np.full((100, 1), 0), 3)

    # When
    y_pred_trash, y_trash = post_processing.convert_to_trash(y_pred, y)

    # Then
    assert(np.array_equal(np.round(y_pred_trash), expected_result))
    assert(np.array_equal(np.round(y_trash), expected_result))


def test_convert_to_trash_paper_converted_to_recyclable():
    # Given
    y_pred = np.full((100, 1), 3)
    y =  np.full((100, 1), 3)
    y_pred = to_categorical(y_pred, 6)
    y = to_categorical(y, 6)

    expected_result = to_categorical(np.full((100, 1), 0), 3)

    # When
    y_pred_trash, y_trash = post_processing.convert_to_trash(y_pred, y)

    # Then
    assert(np.array_equal(np.round(y_pred_trash), expected_result))
    assert(np.array_equal(np.round(y_trash), expected_result))


def test_convert_to_trash_plastic_converted_to_recyclable():
    # Given
    y_pred = np.full((100, 1), 4)
    y =  np.full((100, 1), 4)
    y_pred = to_categorical(y_pred, 6)
    y = to_categorical(y, 6)

    expected_result = to_categorical(np.full((100, 1), 0), 3)

    # When
    y_pred_trash, y_trash = post_processing.convert_to_trash(y_pred, y)

    # Then
    assert(np.array_equal(np.round(y_pred_trash), expected_result))
    assert(np.array_equal(np.round(y_trash), expected_result))


def test_convert_to_trash_trash_converted_to_non_recyclable():
    # Given
    y_pred = np.full((100, 1), 5)
    y =  np.full((100, 1), 5)
    y_pred = to_categorical(y_pred, 6)
    y = to_categorical(y, 6)

    expected_result = to_categorical(np.full((100, 1), 2), 3)

    # When
    y_pred_trash, y_trash = post_processing.convert_to_trash(y_pred, y)

    # Then
    assert(np.array_equal(np.round(y_pred_trash), expected_result))
    assert(np.array_equal(np.round(y_trash), expected_result))