import pandas as pd
import numpy as np
import keras
import keras.backend
import sklearn.metrics
import os
import json
import data_management
import plotting


"""
    File for Hierarchical Tree Classifier, Neural Node,
        Multi-class Classifier and Binary Classifier.
"""


class TreeNeuralNetwork:
    """
        Class for Hierarchical Tree Classifier.
    """
    # Dictionary for tree
    tree = None
    tree_structure = None

    def __init__(self):
        """
            Method for initialization.
        """
        self.tree, self.tree_structure = {}, {}

        return

    def training_tree_initialization(self, data):
        """
            Method for tree creation for training.
            param:
                data - pd.DataFrame of data
        """
        # Root node initialization
        self.tree_structure["outlet"] = {"children": []}

        self.tree["outlet"] = \
            {"node": Node("outlet"),
             "children": []}
        self.tree["outlet"]["node"].training(
                    data.copy().drop(columns=["mcc_2", "mcc_3", "category2",
                                              "category3"]).
                    rename(columns={"mcc_1": "mcc",
                                    "category1": "category"}))

        # Nodes from level 1 categories layer initialization
        for category_1 in np.unique(data["category1"].values):
            local_first_data = data[data["category1"] == category_1]. \
                reindex().copy()

            local_first_tree_structure = {}

            local_first_tree = {}

            local_first_tree_structure[category_1] = \
                {"children":
                    list(np.unique(local_first_data["category2"].values))}

            local_first_tree[category_1] = \
                {"node":
                    Node("outlet/" + category_1),
                 "children": []}
            local_first_tree[category_1]["node"].\
                training(local_first_data.copy().drop(
                             columns=["mcc_1", "mcc_3", "category1",
                                      "category3"]).
                         rename(columns={"mcc_2": "mcc",
                                         "category2": "category"}))

            # Node from level 2 categories layer initialization
            for category_2 in np.unique(local_first_data["category2"].values):
                local_second_data = \
                    local_first_data[local_first_data["category2"] ==
                                     category_2].copy().reindex()

                local_second_tree = {}

                local_second_tree[category_2] = \
                    {"node":
                        Node("outlet/" + category_1 + "/" + category_2),
                     "children": None}
                local_second_tree[category_2]["node"].training(
                    local_second_data.copy().drop(
                                 columns=["mcc_1", "mcc_2", "category1",
                                          "category2"]).
                    rename(columns={"mcc_3": "mcc",
                                    "category3": "category"}))

                local_first_tree[category_1]["children"].append(
                    local_second_tree.copy())

            self.tree["outlet"]["children"].append(local_first_tree.copy())

            self.tree_structure["outlet"]["children"]. \
                append(local_first_tree_structure.copy())

        # Tree structure saving as JSON
        with open("data/model/tree_structure.json", "w",
                  encoding="utf-8") as file:
            json.dump(self.tree_structure, file, ensure_ascii=False)

        return

    def training(self, data):
        """
            Method for hierarchical tree classifier training.
            param:
                data - pd.DataFrame of data
        """
        self.training_tree_initialization(data)

        return

    def evaluation(self, validation_set):
        """
            Method to evaluate hierarchical tree classifier.
            param:
                validation_set - list of numpy arrays for input and output
        """
        validation_set_input = validation_set[0]
        validation_set_output = validation_set[1]

        # Output binarization
        model = sklearn.preprocessing.MultiLabelBinarizer()
        model = model.fit(validation_set_output)

        validation_set_output = \
            model.transform(validation_set_output)

        prediction_results = []

        # Prediction by rows
        for row in validation_set_input:
            prediction_result = []

            # Prediction on the root node
            neural_node = Node("outlet")
            category_1_prediction = neural_node.prediction(row[:518])
            prediction_result.append(category_1_prediction)

            # Prediction on the level 1 category layer node
            neural_node = Node("outlet/" + category_1_prediction)
            category_2_prediction = neural_node.prediction(
                np.concatenate([row[:517], np.array([row[518]])]))

            if category_2_prediction != "none":
                prediction_result.append(category_2_prediction)

            # Prediction on the level 3 category layer node
            neural_node = Node("outlet/" +
                               category_1_prediction + "/" +
                               category_2_prediction)
            category_3_prediction = neural_node.prediction(
                np.concatenate([row[:517], np.array([row[519]])]))

            if category_3_prediction != "none":
                prediction_result.append(category_3_prediction)

            prediction_results.append(prediction_result)

        # Prediction binarization
        prediction_output = model.transform(prediction_results)

        validation_report = sklearn.metrics.classification_report(
                validation_set_output, prediction_output,
                output_dict=True)

        # Metrics computing
        validation_scores = validation_report["samples avg"]
        validation_scores["accuracy"] =  \
            sklearn.metrics.accuracy_score(
                validation_set_output, prediction_output)
        validation_scores["auc"] = \
            sklearn.metrics.roc_auc_score(
                validation_set_output, prediction_output, average="macro")

        del validation_scores["support"]

        # Metrics plotting
        validation_scores = pd.Series(validation_scores)
        plotting.vertical_bar_plotting(validation_scores,
                                       ["Metrics", "Values"],
                                       "Hierarchical model metrics on "
                                       "validation set", "data/model",
                                       font_size=26)

        # Metrics output
        print("Hierarchical model validation scores: \n", validation_scores)

        return


class Node:
    """
        Class for Hierarchical Tree Classifier node.
    """
    # String folder path
    folder = None
    # Keras model
    model = None
    # Dictionary of output data embedding
    embedding_data = None

    # pd.DataFrame of training set
    training_set = None
    # pd.DataFrame of validation set
    validation_set = None

    def __init__(self, folder):
        """
            Method for class initialization.
            param:
                folder - string folder path
        """
        self.folder = "data/model/" + folder

        return

    def training_node_initialization(self, data):
        """
            Node initialization for training.
            param:
                data - pd.DataFrame of localized data
            return:
                categories_number - number of categories that
                    can be predicted
        """
        # Directory creation
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        # Statistics on the node computation
        data_management.node_data_statistics(data, self.folder)

        # Output embedding
        data, self.embedding_data = \
            data_management.node_data_embedding_creation(data, self.folder)

        # Computing number of categories
        categories_number = len(list(self.embedding_data.keys()))

        # Model selection and creation
        if categories_number == 1:
            print("No model is needed")
            self.model = list(self.embedding_data.keys())[0]
        else:
            data = data_management.node_data_distortion(data)

            self.training_set, self.validation_set = \
                data_management.node_sets_creation(data)

            if categories_number == 2:
                print("Binary classification task")
                self.model = BinaryClassifier(self.embedding_data,
                                              1, self.folder)
            else:
                print("Multi-class classification task")
                self.model = \
                    MultiClassClassifier(self.embedding_data,
                                         self.validation_set[1].shape[1],
                                         self.folder)

        return categories_number

    def training(self, data):
        """
            Method to train model on the node.
            param:
                data - pd.DataFrame of localized data
        """
        print("Training of node: '" + self.folder.split('/')[-1] + "'")

        # Training initialization
        training_flag = self.training_node_initialization(data)

        # Training if needed
        if training_flag == 1:
            return
        else:
            self.model.model_training_and_evaluation(self.training_set,
                                                     self.validation_set)

            return

    def prediction_node_initialization(self):
        """
            Node initialization for prediction.
        """
        # Embeddings import
        self.embedding_data = \
            data_management.node_data_embedding_reading(self.folder)

        # Categories number computing
        categories_number = len(list(self.embedding_data.keys()))

        # Model selection and initialization
        if categories_number == 1:
            self.model = list(self.embedding_data.keys())[0]
        elif categories_number == 2:
            self.model = BinaryClassifier(self.embedding_data,
                                          1, self.folder)
        else:
            self.model = MultiClassClassifier(self.embedding_data,
                                              categories_number, self.folder)

        return categories_number

    def prediction(self, data):
        """
            Method for prediction on the node.
            param:
                data - numpy array for prediction
            return:
                String prediction result
        """
        prediction_flag = self.prediction_node_initialization()

        if prediction_flag == 1:
            return list(self.embedding_data.keys())[0]
        else:
            return self.model.model_prediction(data)


class MultiClassClassifier:
    """
        Class for multi-class classifier.
    """
    # Dictionary of embeddings
    embedding_data = None
    # Number of neurons for the output layer
    output_shape = None
    # String folder path
    folder = None
    # Keras model
    model = None

    def __init__(self, embedding_data, output_shape, folder):
        """
            Method for class initialization.
            param:
                1. embedding_data - dictionary of embeddings
                2. output_shape - number of neurons for the output layer
                3. folder - string folder path
        """
        self.embedding_data = embedding_data
        self.output_shape = output_shape
        self.folder = folder

        return

    def model_training_and_evaluation(self, training_set, validation_set,
                                      batch_size=256, epochs=1000):
        """
            Method to train and evaluate neural network model.
            param:
                1. training_set - list of pd.DataFrames for training
                2. validation_set - list of pd.DataFrames for validation
                3 batch_size - batch size for training (256 as default)
                4. epochs - number of epochs for training (1000 as default)
        """
        self.model_creation()

        # Train model
        self.model.fit(x=training_set[0],
                       y=training_set[1],
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=1,
                       validation_data=(validation_set[0],
                                        validation_set[1]),
                       shuffle=True,
                       callbacks=[keras.callbacks.TerminateOnNaN(),
                                  keras.callbacks.EarlyStopping(
                                      patience=20, restore_best_weights=True),
                                  keras.callbacks.ReduceLROnPlateau(
                                      patience=2, min_lr=1e-15),
                                  keras.callbacks.TensorBoard(
                                      log_dir=self.folder + "/logs",
                                      batch_size=batch_size,
                                      write_grads=True,
                                      write_images=True)])

        # Save weights
        self.model.save_weights(self.folder + "/weights.h5")

        model_name = '-'.join(self.folder.split("/")[2:])

        # Model validation
        output_predicted = \
            np.rint(self.model.predict(x=validation_set[0],
                                       batch_size=batch_size))

        # Validation metrics computation
        validation_report = sklearn.metrics.classification_report(
            validation_set[1].values, output_predicted,
            target_names=list(self.embedding_data.keys()), output_dict=True)

        scores_validation = validation_report["macro avg"]
        scores_validation["accuracy"] = \
            sklearn.metrics.accuracy_score(validation_set[1].values,
                                           output_predicted)
        scores_validation["auc"] = \
            sklearn.metrics.roc_auc_score(validation_set[1].values,
                                          output_predicted)
        del scores_validation["support"]

        # Validation metrics plotting
        scores_validation = pd.Series(scores_validation)
        plotting.vertical_bar_plotting(scores_validation,
                                       ["Metrics", "Values"],
                                       model_name + " metrics validation set",
                                       self.folder, font_size=26)

        # Validation metrics output
        print("Model validation scores: \n", scores_validation)

        # Validation metrics output for every category
        for category in list(self.embedding_data.keys()):
            scores_validation = validation_report[category]
            del scores_validation["support"]

            scores_validation = pd.Series(scores_validation)
            plotting.vertical_bar_plotting(scores_validation,
                                           ["Metrics", "Values"],
                                           model_name +
                                           " metrics validation set (" +
                                           category + ")",
                                           self.folder, font_size=26)
            print("Model validation scores (" + category + "): \n",
                  scores_validation)

        keras.backend.clear_session()

        return

    def model_prediction(self, prediction_set):
        """
            Method for model prediction.
            param:
                prediction_set - numpy array for model prediction
            return:
                String model prediction
        """
        # Model creation
        self.model_creation()
        # Loading weights
        self.model.load_weights(self.folder + "/weights.h5")

        # Embeddings dictionary inverse
        inversed_embedding_data = \
            {np.argmax(category_embedded): category
             for category, category_embedded in
             self.embedding_data.items()}

        # Model prediction
        output = np.argmax(self.model.predict(prediction_set.reshape((1,
                                                                      518))))

        keras.backend.clear_session()

        return inversed_embedding_data[output]

    def model_creation(self):
        """
              Method to create compile neural network model.
              return:
                  model - keras neural network model
        """
        self.model = keras.Sequential()

        self.model.add(keras.layers.Dense(518, input_dim=518,
                                          activation='tanh'))
        self.model.add(keras.layers.Dense(518 // 2, activation="tanh"))
        self.model.add(keras.layers.Dense(518 // 4, activation="tanh"))
        self.model.add(keras.layers.Dense(self.output_shape,
                                          activation="softmax"))

        # Compile model
        self.model.compile(optimizer="Adam",
                           loss="categorical_crossentropy",
                           metrics=["accuracy"])

        plotting.neural_network_plotting(self.model, self.folder)

        return


class BinaryClassifier:
    """
        Class for binary classifier.
    """
    # Dictionary of embeddings
    embedding_data = None
    # Number of neurons for the output layer
    output_shape = None
    # String folder path
    folder = None
    # Keras model
    model = None

    def __init__(self, embedding_data, output_shape, folder):
        """
            Method for class initialization.
            param:
                1. embedding_data - dictionary of embeddings
                2. output_shape - number of neurons for the output layer
                3. folder - string folder path
        """
        self.embedding_data = embedding_data
        self.output_shape = output_shape
        self.folder = folder

        return

    def model_training_and_evaluation(self, training_set, validation_set,
                                      batch_size=256, epochs=1000):
        """
            Method to train and evaluate neural network model.
            param:
                1. training_set - list of pd.DataFrames for training
                2. validation_set - list of pd.DataFrames for validation
                3 batch_size - batch size for training (256 as default)
                4. epochs - number of epochs for training (1000 as default)
        """
        self.model_creation()

        # Train model
        self.model.fit(x=training_set[0],
                       y=training_set[1],
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=1,
                       validation_data=(validation_set[0],
                                        validation_set[1]),
                       shuffle=True,
                       callbacks=[keras.callbacks.TerminateOnNaN(),
                                  keras.callbacks.EarlyStopping(
                                      patience=20, restore_best_weights=True),
                                  keras.callbacks.ReduceLROnPlateau(
                                      patience=2, min_lr=1e-15),
                                  keras.callbacks.TensorBoard(
                                      log_dir=self.folder + "/logs",
                                      batch_size=batch_size,
                                      write_grads=True,
                                      write_images=True)])

        # Save weights
        self.model.save_weights(self.folder + "/weights.h5")

        model_name = '-'.join(self.folder.split("/")[2:])

        # Model validation
        output_predicted = \
            np.rint(self.model.predict(x=validation_set[0],
                                       batch_size=batch_size)).flatten()

        # Metrics computation
        validation_report = sklearn.metrics.classification_report(
            validation_set[1].values.flatten(), output_predicted,
            target_names=list(self.embedding_data.keys()), output_dict=True)

        scores_validation = validation_report["macro avg"]
        scores_validation["accuracy"] = \
            sklearn.metrics.accuracy_score(validation_set[1].values,
                                           output_predicted)
        scores_validation["auc"] = \
            sklearn.metrics.roc_auc_score(validation_set[1].values.flatten(),
                                          output_predicted)
        del scores_validation["support"]

        # Metrics plotting
        scores_validation = pd.Series(scores_validation)
        plotting.vertical_bar_plotting(scores_validation,
                                       ["Metrics", "Values"],
                                       model_name + " metrics validation set",
                                       self.folder, font_size=26)

        # Metrics output
        print("Model validation scores: \n", scores_validation)

        # Metrics computation and plotting for every category
        for category in list(self.embedding_data.keys()):
            scores_validation = validation_report[category]
            del scores_validation["support"]

            scores_validation = pd.Series(scores_validation)
            plotting.vertical_bar_plotting(scores_validation,
                                           ["Metrics", "Values"],
                                           model_name +
                                           " metrics validation set (" +
                                           category + ")",
                                           self.folder, font_size=26)
            print("Model validation scores (" + category + "): \n",
                  scores_validation)

        keras.backend.clear_session()

        return

    def model_prediction(self, prediction_set):
        """
            Method for model prediction.
            param:
                prediction_set - numpy array for model prediction
            return:
                String model prediction
        """
        # Model creation
        self.model_creation()
        # Weights loading
        self.model.load_weights(self.folder + "/weights.h5")

        # Embeddings dictionary inverse
        inversed_embedding_data = \
            {category_embedded[0]: category
             for category, category_embedded in self.embedding_data.items()}

        # Model prediction
        output = \
            np.rint(self.model.predict(prediction_set.reshape((1, 518))))[0, 0]

        keras.backend.clear_session()

        return inversed_embedding_data[output]

    def model_creation(self):
        """
              Method to create compile neural network model.
              return:
                  model - keras neural network model
        """
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(518, input_dim=518,
                                          activation='tanh'))
        self.model.add(keras.layers.Dense(518 // 2, activation="tanh"))
        self.model.add(keras.layers.Dense(self.output_shape,
                                          activation="sigmoid"))

        # Compile model
        self.model.compile(optimizer="Adam",
                           loss="binary_crossentropy",
                           metrics=["accuracy"])

        plotting.neural_network_plotting(self.model, self.folder)

        return
