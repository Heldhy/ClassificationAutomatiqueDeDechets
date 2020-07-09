from use_model import return_trash_label


def test_return_trash_label_returns_zero_for_label_zero():
    # Given
    label = 0
    expected_label = 0

    # When
    returned_label = return_trash_label(label)

    # Then
    assert expected_label == returned_label


def test_return_trash_label_returns_zero_for_label_two():
    # Given
    label = 2
    expected_label = 0

    # When
    returned_label = return_trash_label(label)

    # Then
    assert expected_label == returned_label


def test_return_trash_label_returns_zero_for_label_three():
    # Given
    label = 3
    expected_label = 0

    # When
    returned_label = return_trash_label(label)

    # Then
    assert expected_label == returned_label


def test_return_trash_label_returns_zero_for_label_four():
    # Given
    label = 4
    expected_label = 0

    # When
    returned_label = return_trash_label(label)

    # Then
    assert expected_label == returned_label


def test_return_trash_label_returns_one_for_label_one():
    # Given
    label = 1
    expected_label = 1

    # When
    returned_label = return_trash_label(label)

    # Then
    assert expected_label == returned_label


def test_return_trash_label_returns_two_for_label_five():
    # Given
    label = 5
    expected_label = 2

    # When
    returned_label = return_trash_label(label)

    # Then
    assert expected_label == returned_label

