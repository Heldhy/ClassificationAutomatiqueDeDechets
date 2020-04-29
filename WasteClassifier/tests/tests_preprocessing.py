import numpy as np
from waste_classifier import pre_processing


def test_make_square_of_a_squared_picture():
    # Given
    randnum = np.random.randint(0, 255, 7500)
    img = randnum.reshape((50, 50, 3))

    # When
    squared = pre_processing.make_square(img)

    # Then
    assert (squared.shape == (224, 224, 3))


def test_make_square_does_not_change_squared_picture():
    # Given
    randnum = np.random.randint(0, 255, 7500)
    img = randnum.reshape((50, 50, 3))

    # When
    squared = pre_processing.make_square(img, 50)

    # Then
    assert ((squared == img).all())


def test_make_square_of_a_rectangle_image():
    # Given
    randnum = np.random.randint(0, 255, 7200)
    img = randnum.reshape((60, 40, 3))

    # When
    squared = pre_processing.make_square(img)

    # Then
    assert (squared.shape == (224, 224, 3))


def test_make_square_not_only_one_color_for_a_rectangle():
    # Given
    randnum = np.random.randint(0, 255, 7200)
    img = randnum.reshape((60, 40, 3))

    # When
    squared = pre_processing.make_square(img)

    # Then
    assert ((np.mean(squared, axis=0) != squared[112, 112]).all())


def test_make_square_not_only_one_color_for_a_square():
    # Given
    randnum = np.random.randint(0, 255, 7500)
    img = randnum.reshape((50, 50, 3))

    # When
    squared = pre_processing.make_square(img)

    # Then
    assert ((np.mean(squared, axis=0) != squared[112, 112]).all())


def test_make_square_two_different_pictures_are_different():
    # Given
    first_array = np.random.randint(0, 255, 7500)
    first_img = first_array.reshape((50, 50, 3))
    second_array = np.random.randint(0, 255, 7500)
    second_img = second_array.reshape((50, 50, 3))

    # When
    first_img = pre_processing.make_square(first_img)
    second_img = pre_processing.make_square(second_img)

    # Then
    assert (not np.array_equal(first_img, second_img))


