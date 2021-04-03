import numpy as np
import keras
import matplotlib.pyplot as plt


def horizontal_bar_plotting(data, labels, title, folder,
                            width=1920, height=1080, dpi=96, font_size=22,
                            color='b'):
    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.rcParams.update({'font.size': font_size})
    plt.barh((np.arange(len(data))), data.iloc[::-1],
             align='center', color=color)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.yticks(np.arange(len(data)), data.iloc[::-1].index)
    plt.xlim(xmin=0.0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(folder + "/" + title + ".png", dpi=dpi)
    plt.close()

    return


def vertical_bar_plotting(data, labels, title, folder,
                          width=1920, height=1080, dpi=96, font_size=22,
                          color='b'):
    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.rcParams.update({'font.size': font_size})
    plt.bar((np.arange(len(data))), data, align='center', color=color)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.xticks(np.arange(len(data)), data.index)
    plt.ylim(ymin=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(folder + "/" + title + ".png", dpi=dpi)
    plt.close()

    return


def box_plot_plotting(observed_values, categories, labels, title, folder,
                      width=1920, height=1080, dpi=96, font_size=22):
    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.rcParams.update({'font.size': font_size})
    plt.boxplot(observed_values, vert=False, labels=categories)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(folder + "/" + title + ".png", dpi=dpi)
    plt.close()

    return


def neural_network_plotting(model, folder):
    keras.utils.plot_model(model,
                           to_file=folder + "/model.png",
                           show_shapes=True)

    return
