import pandas as pd
import numpy as np
import keras
import keras.backend
import sklearn.metrics
import os
import json
import data_management
import plotting


class TreeNeuralNetwork:
    tree = None
    tree_structure = None

    def __init__(self):
        self.tree, self.tree_structure = {}, {}

        return

    def training_tree_initialization(self, data):
        self.tree_structure["outlet"] = {"children": []}

        self.tree["outlet"] = \
            {"node": Node("outlet"),
             "children": []}
        self.tree["outlet"]["node"].training(
                    data.copy().drop(columns=["mcc_2", "mcc_3", "category2",
                                              "category3"]).
                    rename(columns={"mcc_1": "mcc",
                                    "category1": "category"}))

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

        with open("data/model/tree_structure.json", "w",
                  encoding="utf-8") as file:
            json.dump(self.tree_structure, file, ensure_ascii=False)

        return

    def training(self, data):
        self.training_tree_initialization(data)

        return

    def evaluation(self, validation_set):
        validation_set_input = validation_set[0]
        validation_set_output = validation_set[1]

        model = sklearn.preprocessing.MultiLabelBinarizer()
        model = model.fit(validation_set_output)

        validation_set_output = \
            model.transform(validation_set_output)

        prediction_results = []

        for row in validation_set_input:
            prediction_result = []

            neural_node = Node("outlet")
            category_1_prediction = neural_node.prediction(row[:518])
            prediction_result.append(category_1_prediction)

            neural_node = Node("outlet/" + category_1_prediction)
            category_2_prediction = neural_node.prediction(
                np.concatenate([row[:517], np.array([row[518]])]))

            if category_2_prediction != "none":
                prediction_result.append(category_2_prediction)

            neural_node = Node("outlet/" +
                               category_1_prediction + "/" +
                               category_2_prediction)
            category_3_prediction = neural_node.prediction(
                np.concatenate([row[:517], np.array([row[519]])]))

            if category_3_prediction != "none":
                prediction_result.append(category_3_prediction)

            prediction_results.append(prediction_result)

        prediction_output = model.transform(prediction_results)

        validation_report = sklearn.metrics.classification_report(
                validation_set_output, prediction_output,
                output_dict=True)

        validation_scores = validation_report["samples avg"]
        validation_scores["accuracy"] =  \
            sklearn.metrics.accuracy_score(
                validation_set_output, prediction_output)
        validation_scores["auc"] = \
            sklearn.metrics.roc_auc_score(
                validation_set_output, prediction_output, average="macro")

        del validation_scores["support"]

        validation_scores = pd.Series(validation_scores)
        plotting.vertical_bar_plotting(validation_scores,
                                       ["Metrics", "Values"],
                                       "Hierarchical model metrics on "
                                       "validation set", "data/model",
                                       font_size=26)
        print("Hierarchical model validation scores: \n", validation_scores)

        return


class Node:
    folder = None
    model = None
    embedding_data = None

    training_set = None
    validation_set = None

    def __init__(self, folder):
        self.folder = "data/model/" + folder

        return

    def training_node_initialization(self, data):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        data_management.node_data_statistics(data, self.folder)

        data, self.embedding_data = \
            data_management.node_data_embedding_creation(data, self.folder)

        categories_number = len(list(self.embedding_data.keys()))

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
        if self.folder.split('/')[-1] not in ["outlet"]:
            return

        print("Training of node: '" + self.folder.split('/')[-1] + "'")

        training_flag = self.training_node_initialization(data)

        if training_flag == 1:
            return
        else:
            self.model.model_training_and_evaluation(self.training_set,
                                                     self.validation_set)

            return

    def prediction_node_initialization(self):
        self.embedding_data = \
            data_management.node_data_embedding_reading(self.folder)

        categories_number = len(list(self.embedding_data.keys()))

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
        prediction_flag = self.prediction_node_initialization()

        if prediction_flag == 1:
            return list(self.embedding_data.keys())[0]
        else:
            return self.model.model_prediction(data)


class MultiClassClassifier:
    embedding_data = None
    output_shape = None
    folder = None
    model = None

    def __init__(self, embedding_data, output_shape, folder):
        self.embedding_data = embedding_data
        self.output_shape = output_shape
        self.folder = folder

        return

    def model_training_and_evaluation(self, training_set, validation_set,
                                      batch_size=256, epochs=1000):
        """
            Method to train and evaluate neural network model.
            param:
                1. model - keras neural network model
                2. training_set - pandas DataFrame of training data
                3. validation_set - pandas trained DataFrame of testing data
                4. folder - string name of folder, where data need to be saved
            return:
                model - keras neural network trained model
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

        output_predicted = \
            np.rint(self.model.predict(x=validation_set[0],
                                       batch_size=batch_size))

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

        scores_validation = pd.Series(scores_validation)
        plotting.vertical_bar_plotting(scores_validation,
                                       ["Metrics", "Values"],
                                       model_name + " metrics validation set",
                                       self.folder, font_size=26)
        print("Model validation scores: \n", scores_validation)

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
        self.model_creation()
        self.model.load_weights(self.folder + "/weights.h5")

        inversed_embedding_data = \
            {np.argmax(category_embedded): category
             for category, category_embedded in
             self.embedding_data.items()}

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
    embedding_data = None
    output_shape = None
    folder = None
    model = None

    def __init__(self, embedding_data, output_shape, folder):
        self.embedding_data = embedding_data
        self.output_shape = output_shape
        self.folder = folder

        return

    def model_training_and_evaluation(self, training_set, validation_set,
                                      batch_size=256, epochs=1000):
        """
            Method to train and evaluate neural network model.
            param:
                1. model - keras neural network model
                2. training_set - pandas DataFrame of training data
                3. validation_set - pandas trained DataFrame of testing data
                4. folder - string name of folder, where data need to be saved
            return:
                model - keras neural network trained model
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

        output_predicted = \
            np.rint(self.model.predict(x=validation_set[0],
                                       batch_size=batch_size)).flatten()

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

        scores_validation = pd.Series(scores_validation)
        plotting.vertical_bar_plotting(scores_validation,
                                       ["Metrics", "Values"],
                                       model_name + " metrics validation set",
                                       self.folder, font_size=26)
        print("Model validation scores: \n", scores_validation)

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
        self.model_creation()
        self.model.load_weights(self.folder + "/weights.h5")

        inversed_embedding_data = \
            {category_embedded[0]: category
             for category, category_embedded in self.embedding_data.items()}

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
