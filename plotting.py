import numpy as np
import keras
import matplotlib.pyplot as plt


"""
    File for plotting functions.
"""


def horizontal_bar_plotting(data, labels, title, folder,
                            width=1920, height=1080, dpi=96, font_size=22,
                            color='b'):
    """
        Method to plot horizontal bar chart.
        param:
            1. data - pd.Series of data
            2. labels - list of string labels for axes
            3. title - string title of the plot
            4. folder - string path to folder
            5. width - width of the plot (1920 as default)
            6. height - height of the plot (1080 as default)
            7. dpi - DPI (96 as default)
            8. font_size - font size for plot (22 as default)
            9. color - color for the plot ('b' as default)
    """
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
    """
        Method to plot vertical bar chart.
        param:
            1. data - pd.Series of data
            2. labels - list of string labels for axes
            3. title - string title of the plot
            4. folder - string path to folder
            5. width - width of the plot (1920 as default)
            6. height - height of the plot (1080 as default)
            7. dpi - DPI (96 as default)
            8. font_size - font size for plot (22 as default)
            9. color - color for the plot ('b' as default)
    """
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
    """
        Method to plot box plots.
        param:
            1. observed_values - numpy array of observations
            2. categories - list of category names
            3. labels - list of string labels for axes
            4. title - string title of the plot
            5. folder - string path to folder
            6. width - width of the plot (1920 as default)
            7. height - height of the plot (1080 as default)
            8. dpi - DPI (96 as default)
            9. font_size - font size for plot (22 as default)
    """
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
    """
        Method to plot neural networks architecture.
        param:
            1. model - keras neural network model
            2. folder - string path to folder
    """
    keras.utils.plot_model(model,
                           to_file=folder + "/model.png",
                           show_shapes=True)

    return
