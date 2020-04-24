import matplotlib.pyplot as plt
import numpy as np
import preprocessing


def test_make_square_of_a_squared_picture():
    randnum = np.random.randint(0,255,7500)
    img = randnum.reshape((50,50,3))
    squared = preprocessing.make_square(img)
    assert(squared.shape == (224,224,3))

def test_squared_picture_not_changed():
    randnum = np.random.randint(0,255,7500)
    img = randnum.reshape((50,50,3))
    squared = preprocessing.make_square(img, 50)
    assert((squared == img).all())

def test_make_square_of_a_rectangle_image():
    randnum = np.random.randint(0, 255, 7200)
    img = randnum.reshape((60, 40, 3))
    squared = preprocessing.make_square(img)
    assert (squared.shape == (224, 224, 3))

def test_not_only_one_color_after_make_square_for_a_rectangle():
    randnum = np.random.randint(0, 255, 7200)
    img = randnum.reshape((60, 40, 3))
    squared = preprocessing.make_square(img)
    assert ((np.mean(squared, axis=0) != squared[112,112]).all())

def test_not_only_one_color_after_make_square_for_a_square():
    randnum = np.random.randint(0, 255, 7500)
    img = randnum.reshape((50, 50, 3))
    squared = preprocessing.make_square(img)
    assert ((np.mean(squared, axis=0) != squared[112,112]).all())
