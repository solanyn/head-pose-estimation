import numpy as np
from matplotlib import pyplot as plt
from itertools import cycle


def plotter(history_hold, metric = 'MeanAbsoluteError', ylim=[0.0, 1.0]):
    cycol = cycle('bgrcmk')
    for name, item in history_hold.items():
        y_train = item.history[metric]
        y_val = item.history['val_' + metric]
        x_train = np.arange(0,len(y_val))

        c=next(cycol)

        plt.plot(x_train, y_train, c+'-', label=name+'_train')
        plt.plot(x_train, y_val, c+'--', label=name+'_val')

    plt.legend()
    plt.xlim([1, max(plt.xlim())])
    plt.ylim(ylim)
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.grid(True)
    plt.show()

def plot_image(data_dir, predictions_array, path, class_names):
    img = plt.imread(data_dir + path)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)

    plt.xlabel("{} {:2.0f}%".format(class_names[predicted_label],
                                100*np.max(predictions_array)))
    plt.show()

def plot_value_array(predictions_array, class_names):
    plt.grid(False)
    plt.xticks(range(len(class_names)))
    plt.yticks([])
    thisplot = plt.bar(range(len(class_names)), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    plt.show()

