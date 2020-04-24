from variables import *
import matplotlib.pyplot as plt
from preprocessing import make_square
from preprocessing import preprocess_input


def predict_image(model, path, open=False):
    if (not open):
        img = plt.imread(path)
    else:
        img = path
    img = make_square(img)
    plt.imshow(img)
    img = preprocess_input(img)
    pred = model.predict(img.reshape((1, HEIGHT, WIDTH, 3)))
    p = pred.tolist()[0]
    idx = p.index(max(p))
    idx2 = p.index(max(p[:idx] + p[idx + 1:]))
    trash = 0 if (idx in {0, 2, 3, 4}) else 1 if idx == 1 else 2
    print(classes[idx] + ": " + str(max(p) * 100)[:5] + "%")
    print(classes[idx2] + ": " + str(p[idx2] * 100)[:5] + "%")
    print("--")
    return typesFr[trash]
