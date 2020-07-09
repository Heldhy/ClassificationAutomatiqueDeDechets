import numpy as np

from waste_classifier.soft_recall import acceptable_dict_to_matrix, get_class, soft_recall_function


def get_prediction_array():
    return np.array(
        [
            [0.8, 0.3, 0.00076, 0.5, 0.1, 0.05],
            [0.8, 0.3, 0.00076, 0.5, 0.1, 0.05],
            [0.8, 0.3, 0.00076, 0.5, 0.1, 0.05],
            [0.8, 0.3, 0.00076, 0.5, 0.1, 0.05],
            [0.8, 0.3, 0.00076, 0.5, 0.1, 0.05],
            [0.02, 0.8, 0.00076, 0.5, 0.1, 0.05],
            [0.02, 0.8, 0.00076, 0.5, 0.1, 0.05],
            [0.02, 0.8, 0.00076, 0.5, 0.1, 0.05],
            [0.02, 0.8, 0.00076, 0.5, 0.1, 0.05],
            [0.02, 0.8, 0.00076, 0.5, 0.1, 0.05],
            [0.02, 0.8, 0.00076, 0.5, 0.1, 0.05],
            [0.02, 0.8, 0.00076, 0.5, 0.1, 0.05],
            [0.02, 0.8, 0.00076, 0.5, 0.1, 0.05],
            [0.02, 0.08, 0.76, 0.5, 0.1, 0.05],
            [0.02, 0.08, 0.76, 0.5, 0.1, 0.05],
            [0.02, 0.08, 0.76, 0.5, 0.1, 0.05],
            [0.02, 0.08, 0.76, 0.5, 0.1, 0.05],
            [0.02, 0.08, 0.76, 0.5, 0.1, 0.05],
            [0.02, 0.08, 0.76, 0.5, 0.1, 0.05],
            [0.02, 0.08, 0.076, 0.5, 0.1, 0.05],
            [0.02, 0.08, 0.076, 0.5, 0.1, 0.05],
            [0.02, 0.08, 0.076, 0.5, 0.1, 0.05],
            [0.02, 0.08, 0.076, 0.5, 0.1, 0.05],
            [0.02, 0.08, 0.076, 0.5, 0.1, 0.05],
            [0.02, 0.08, 0.076, 0.5, 0.1, 0.05],
            [0.02, 0.08, 0.076, 0.5, 0.1, 0.05],
            [0.02, 0.08, 0.076, 0.05, 0.91, 0.05],
            [0.02, 0.08, 0.076, 0.05, 0.91, 0.05],
            [0.02, 0.08, 0.076, 0.05, 0.91, 0.05],
            [0.02, 0.08, 0.076, 0.05, 0.91, 0.05],
            [0.02, 0.08, 0.076, 0.05, 0.91, 0.05],
            [0.02, 0.08, 0.076, 0.05, 0.91, 0.05],
            [0.02, 0.08, 0.076, 0.05, 0.091, 0.59],
            [0.02, 0.08, 0.076, 0.05, 0.091, 0.59],
            [0.02, 0.08, 0.076, 0.05, 0.091, 0.59],
            [0.02, 0.08, 0.076, 0.05, 0.091, 0.59],
            [0.02, 0.08, 0.076, 0.05, 0.091, 0.59],
        ]
    )


def get_true_array():
    return np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )


def test_acceptable_dict_to_matrix():
    # Given
    expected_dict = {
        'cardboard': {'cardboard': False, 'glass': False, 'metal': True, 'paper': True, 'plastic': True, 'trash': True},
        'glass': {'cardboard': False, 'glass': False, 'metal': False, 'paper': False, 'plastic': False, 'trash': True},
        'metal': {'cardboard': True, 'glass': False, 'metal': False, 'paper': True, 'plastic': True, 'trash': True},
        'paper': {'cardboard': True, 'glass': False, 'metal': True, 'paper': False, 'plastic': True, 'trash': True},
        'plastic': {'cardboard': True, 'glass': False, 'metal': True, 'paper': True, 'plastic': False, 'trash': True},
        'trash': {'cardboard': False, 'glass': False, 'metal': False, 'paper': False, 'plastic': False, 'trash': False}}
    an_table = {"paper": ["plastic", "cardboard", "metal", "trash"],
                "plastic": ["paper", "cardboard", "metal", "trash"],
                "cardboard": ["paper", "plastic", "metal", "trash"],
                "metal": ["paper", "plastic", "cardboard", "trash"],
                "glass": ["trash"], "trash": []}
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

    # When
    returned_dict = acceptable_dict_to_matrix(an_table, classes)

    # Then
    assert returned_dict == expected_dict


def test_get_class():
    # Given
    pseudo_prediction = np.array([0.01, 0.3, 0.00076, 0.5, 0.1, 0.05])
    expected_class = "paper"
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

    # When
    returned_class = get_class(pseudo_prediction, classes)

    # Then
    assert returned_class == expected_class


def test_soft_recall_function_weight_recall():
    # Given
    predictions = get_prediction_array()
    true = get_true_array()
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    an_table = {"paper": ["plastic", "cardboard", "metal", "trash"],
                "plastic": ["paper", "cardboard", "metal", "trash"],
                "cardboard": ["paper", "plastic", "metal", "trash"],
                "metal": ["paper", "plastic", "cardboard", "trash"],
                "glass": ["trash"], "trash": []}
    expected_value = 0.1891891891891892

    # When
    returned_value = soft_recall_function(predictions, true, an_table, classes).weight_recall

    # Then
    assert returned_value == expected_value


def test_soft_recall_function_soft_recall():
    # Given
    predictions = get_prediction_array()
    true = get_true_array()
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    an_table = {"paper": ["plastic", "cardboard", "metal", "trash"],
                "plastic": ["paper", "cardboard", "metal", "trash"],
                "cardboard": ["paper", "plastic", "metal", "trash"],
                "metal": ["paper", "plastic", "cardboard", "trash"],
                "glass": ["trash"], "trash": []}
    expected_value = 0.30791666666666667

    # When
    returned_value = soft_recall_function(predictions, true, an_table, classes).soft_recall

    # Then
    assert returned_value == expected_value


def test_soft_recall_function_recall():
    # Given
    predictions = get_prediction_array()
    true = get_true_array()
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    an_table = {"paper": ["plastic", "cardboard", "metal", "trash"],
                "plastic": ["paper", "cardboard", "metal", "trash"],
                "cardboard": ["paper", "plastic", "metal", "trash"],
                "metal": ["paper", "plastic", "cardboard", "trash"],
                "glass": ["trash"], "trash": []}
    expected_value = 0.19464285714285715

    # When
    returned_value = soft_recall_function(predictions, true, an_table, classes).recall

    # Then
    assert returned_value == expected_value


def test_soft_recall_function_weight_soft_recall():
    # Given
    predictions = get_prediction_array()
    true = get_true_array()
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    an_table = {"paper": ["plastic", "cardboard", "metal", "trash"],
                "plastic": ["paper", "cardboard", "metal", "trash"],
                "cardboard": ["paper", "plastic", "metal", "trash"],
                "metal": ["paper", "plastic", "cardboard", "trash"],
                "glass": ["trash"], "trash": []}
    expected_value = 0.2972972972972973

    # When
    returned_value = soft_recall_function(predictions, true, an_table, classes).weight_soft_recall

    # Then
    assert returned_value == expected_value
