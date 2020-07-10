import numpy as np

from waste_classifier.pre_processing import make_image_square


def test_make_image_square_returns_an_image_of_shape_224_224_3_for_a_random_squared_image():
    # Given
    randnum = np.random.randint(0, 255, 7500)
    img = randnum.reshape((50, 50, 3))

    # When
    squared = make_image_square(img)

    # Then
    assert squared.shape == (224, 224, 3)


def test_make_image_square_does_not_change_squared_picture_if_the_size_is_already_the_targeted_one():
    # Given
    randnum = np.random.randint(0, 255, 7500)
    img = randnum.reshape((50, 50, 3))

    # When
    squared = make_image_square(img, 50)

    # Then
    assert np.array_equal(squared, img)


def test_make_image_square_returns_an_image_of_shape_224_224_3_for_a_random_rectangle_image():
    # Given
    randnum = np.random.randint(0, 255, 7200)
    img = randnum.reshape((60, 40, 3))

    # When
    squared = make_image_square(img)

    # Then
    assert squared.shape == (224, 224, 3)


def test_make_image_square_two_different_pictures_are_different_after_being_made_squared():
    # Given
    first_array = np.random.randint(0, 255, 7500)
    first_img = first_array.reshape((50, 50, 3))
    second_array = np.random.randint(0, 255, 7500)
    second_img = second_array.reshape((50, 50, 3))

    # When
    first_img = make_image_square(first_img)
    second_img = make_image_square(second_img)

    # Then
    assert not np.array_equal(first_img, second_img)


