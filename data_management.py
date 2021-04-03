import pandas as pd
import numpy as np
import sklearn.preprocessing
import plotting


def data_loading(path):
    data = pd.read_csv(path)

    return data


def data_statistics(data):
    categories_1_distribution = data[['category1']].groupby('category1').\
        size().sort_values(ascending=False)

    print(categories_1_distribution)

    plotting.horizontal_bar_plotting(categories_1_distribution.iloc[:10],
                                     ["Number", "Categories"],
                                     "Top 10 categories 1",
                                     "data/output_data/plots",
                                     font_size=26)

    av_flow_bbox = []
    top_categories_1 = categories_1_distribution.iloc[:10].index.values

    for category_1 in top_categories_1:
        av_flow_bbox.append(
            data[data['category1'] == category_1]['av_flow'].values.flatten())

    plotting.box_plot_plotting(av_flow_bbox[::-1], top_categories_1[::-1],
                               ["Categories",
                                "Average customers observations"],
                               "Average customers observations for "
                               "top 10 categories 1",
                               "data/output_data/plots",
                               font_size=26)

    categories_2_distribution = data[['category2']].groupby('category2').\
        size().sort_values(ascending=False)

    print(categories_2_distribution)

    plotting.horizontal_bar_plotting(categories_2_distribution.iloc[:10],
                                     ["Number", "Categories"],
                                     "Top 10 categories 2",
                                     "data/output_data/plots",
                                     font_size=26)

    av_flow_bbox = []
    top_categories_2 = categories_2_distribution.iloc[:10].index.values

    for category_2 in top_categories_2:
        av_flow_bbox.append(
            data[data['category2'] == category_2]['av_flow'].values.flatten())

    plotting.box_plot_plotting(av_flow_bbox[::-1], top_categories_2[::-1],
                               ["Categories",
                                "Average customers observations"],
                               "Average customers observations for "
                               "top 10 categories 2",
                               "data/output_data/plots",
                               font_size=26)

    categories_3_distribution = data[['category3']].groupby('category3'). \
        size().sort_values(ascending=False)

    print(categories_3_distribution)

    plotting.horizontal_bar_plotting(categories_3_distribution.iloc[:10],
                                     ["Number", "Categories"],
                                     "Top 10 categories 3",
                                     "data/output_data/plots",
                                     font_size=26)

    av_flow_bbox = []
    top_categories_3 = categories_3_distribution.iloc[:10].index.values

    for category_3 in top_categories_3:
        av_flow_bbox.append(
            data[data['category3'] == category_3]['av_flow'].values.flatten())

    plotting.box_plot_plotting(av_flow_bbox[::-1], top_categories_3[::-1],
                               ["Categories",
                                "Average customers observations"],
                               "Average customers observations for "
                               "top 10 categories 3",
                               "data/output_data/plots",
                               font_size=26)

    mcc_1_distribution = (data[['mcc_1']] * 87).round(decimals=0).\
        groupby('mcc_1').size().sort_values(ascending=False)

    print(mcc_1_distribution)

    plotting.horizontal_bar_plotting(mcc_1_distribution.iloc[:10],
                                     ["Number", "MCC"],
                                     "Top 10 MCC 1", "data/output_data/plots",
                                     font_size=26)

    mcc_2_distribution = (data[['mcc_2']] * 87).round(decimals=0).\
        groupby('mcc_2').size().sort_values(ascending=False)

    print(mcc_2_distribution)

    plotting.horizontal_bar_plotting(mcc_2_distribution.iloc[:10],
                                     ["Number", "MCC"],
                                     "Top 10 MCC 2", "data/output_data/plots",
                                     font_size=26)

    mcc_3_distribution = (data[['mcc_3']] * 87).round(decimals=0).\
        groupby('mcc_3').size().sort_values(ascending=False)

    print(mcc_3_distribution)

    plotting.horizontal_bar_plotting(mcc_3_distribution.iloc[:10],
                                     ["Number", "MCC 3"],
                                     "Top 10 MCC 3", "data/output_data/plots",
                                     font_size=26)

    print("\n")

    for category_1 in np.unique(data["category1"].values):
        print(category_1, " --->")

        for category_2 in np.unique(
                data[data["category1"] == category_1]["category2"].values):
            print("\t", category_2, " --> ",
                  np.unique(data[(data["category1"] == category_1) &
                                 (data["category2"] == category_2)]
                            ["category3"].values))

        print("\n")

    print("\n")

    mcc_unique = np.unique(data[["mcc_1", "mcc_2", "mcc_3"]].values.flatten())

    for mcc in mcc_unique:
        categories_1 = \
            list(np.unique(data[data["mcc_1"] == mcc]["category1"].values))
        categories_2 = \
            list(np.unique(data[data["mcc_2"] == mcc]["category2"].values))
        categories_3 = \
            list(np.unique(data[data["mcc_3"] == mcc]["category3"].values))
        categories = \
            list(set(categories_1) | set(categories_2) | set(categories_3))
        print(np.round(mcc * 87), "-->", categories)

    return


def data_preprocessing(data):
    input_embedded_set = []
    output_embedded_set = []

    for index in range(len(data)):
        try:
            names_string = data[['org_name_embedded']].values[index][0]. \
                split("[")[1].split("]")[0].split(' ')

            names_value = []

            for value in names_string:
                if value == '':
                    continue

                names_value.append(
                    float(value.split("\n")[0].replace("e-0", "e-")))

            names_value = np.array(names_value)
        except:
            names_value = \
                np.array(data[['org_name_embedded']].values[index][0])

        try:
            tr_statistic_string = data[['tr_stat']].values[index][0]. \
                split("[")[1].split("]")[0].split(' ')

            tr_statistic_value = []

            for value in tr_statistic_string:
                if value == '':
                    continue

                tr_statistic_value.append(float(value.split("\n")[0]))

            tr_statistic_value = np.array(tr_statistic_value)
        except:
            tr_statistic_value = \
                np.array(data[['tr_stat']].values[index][0])

        categories_string = []

        for i in range(1, 3):
            category = data[["category" + str(i)]].values[index][0]

            if category != "none":
                categories_string.append(category)

        av_value = np.array([data[['av_flow']].values[index][0]])
        mcc_code_1 = np.array([data[['mcc_1']].values[index][0]])
        mcc_code_2 = np.array([data[['mcc_1']].values[index][0]])
        mcc_code_3 = np.array([data[['mcc_1']].values[index][0]])

        input_embedded_set.append(np.concatenate((names_value,
                                                  tr_statistic_value,
                                                  av_value,
                                                  mcc_code_1,
                                                  mcc_code_2,
                                                  mcc_code_3)))
        output_embedded_set.append(categories_string)

    return np.array(input_embedded_set), np.array(output_embedded_set)


def validation_set_creation(data):
    categories_1_names = data[['category1']].groupby('category1').\
        size().sort_values(ascending=False).index.to_list()

    validation_set = pd.DataFrame(columns=data.columns)

    for category_1_name in categories_1_names:
        data_temp = data[data['category1'] == category_1_name]

        sets = np.split(data_temp.sample(frac=1, random_state=1),
                        [int(.8 * len(data_temp))])

        validation_set = validation_set.append(sets[1])

    validation_set_input, validation_set_output = \
        data_preprocessing(validation_set)

    return (validation_set_input, validation_set_output)


def node_data_statistics(data, folder):
    categories_distribution = data[['category']].groupby('category'). \
        size().sort_values(ascending=False)

    plotting.horizontal_bar_plotting(categories_distribution.iloc[:10],
                                     ["Number", "Categories"],
                                     "Top 10 categories", folder,
                                     font_size=26)

    av_flow_bbox = []
    top_categories = categories_distribution.iloc[:10].index.values

    for category in top_categories:
        av_flow_bbox.append(
            data[data['category'] == category]['av_flow'].values.flatten())

    plotting.box_plot_plotting(av_flow_bbox[::-1], top_categories[::-1],
                               ["Categories",
                                "Average customers observations"],
                               "Average customers observations for "
                               "top 10 categories",
                               folder,
                               font_size=26)

    mcc_distribution = (data[['mcc']] * 87).round(decimals=0).groupby('mcc'). \
        size().sort_values(ascending=False)

    plotting.horizontal_bar_plotting(mcc_distribution.iloc[:10],
                                     ["Number", "MCC"],
                                     "Top 10 MCC", folder,
                                     font_size=26)

    return


def node_data_embedding_creation(data, folder):
    unique_categories = np.unique(data['category'].values)

    encoding_model = sklearn.preprocessing.LabelBinarizer()
    category_embedded = encoding_model.fit_transform(
        unique_categories.reshape(-1, 1))

    def apply_embedded_data(row):
        return category_embedded[row.name, :]

    embedding_data = \
        pd.DataFrame(unique_categories.reshape(-1, 1), columns=["category"])
    embedding_data["category_embedded"] = \
        embedding_data.apply(apply_embedded_data, raw=False, axis=1)

    embedding_data.to_csv(folder + "/embedding_data.csv", index=False)

    embedding_data_raw = {}

    for index in range(len(unique_categories)):
        embedding_data_raw[unique_categories[index]] = \
            category_embedded[index, :]

    data = data.merge(embedding_data, on="category")

    return data, embedding_data_raw


def node_data_embedding_reading(folder):
    embedding_data = pd.read_csv(folder + "/embedding_data.csv")
    unique_categories = embedding_data['category'].values

    embedding_data_raw = {}

    for index in range(len(unique_categories)):
        categories_embedded = \
            embedding_data[['category_embedded']].values[index][0].\
            split("[")[1].split("]")[0].split(' ')

        embedding_values = []

        for value in categories_embedded:
            if value == '':
                continue

            embedding_values.append(
                int(value.split("\n")[0].replace("e-0", "e-")))

        embedding_data_raw[unique_categories[index]] = \
            np.array(embedding_values)

    return embedding_data_raw


def node_data_distortion(data,
                         exemplary_number_of_samples=4000,
                         distortion_weight=1e-1):
    def transformation(row):
        return distortion_array[row.name, :]

    global_distortion_data = pd.DataFrame(columns=data.columns)

    for category in np.unique(data['category'].values):
        category_data = data[data['category'] == category].copy().reindex()

        embedded_set = []

        for index in range(len(category_data)):
            names_string = \
                category_data[['org_name_embedded']].values[index][0]. \
                split("[")[1].split("]")[0].split(' ')

            names_value = []

            for value in names_string:
                if value == '':
                    continue

                names_value.append(
                    float(value.split("\n")[0].replace("e-0", "e-")))

            embedded_set.append(np.array(names_value))

        embedded_set = np.array(embedded_set).reshape(len(category_data), 512)

        std_array = np.std(embedded_set, axis=0)
        mean_array = np.mean(embedded_set, axis=0)

        distortion_data = pd.DataFrame(columns=category_data.columns)
        distortion_number = \
            (exemplary_number_of_samples - len(category_data)) // \
            len(category_data)

        for index in range(len(category_data)):
            names_string = \
                category_data[['org_name_embedded']].values[index][0].\
                split("[")[1].split("]")[0].split(' ')

            names_value = []

            for value in names_string:
                if value == '':
                    continue

                names_value.append(
                    float(value.split("\n")[0].replace("e-0", "e-")))

            distortion_array = np.empty((distortion_number, 512))

            for i in range(len(names_value)):
                distortion = \
                    np.random.normal(mean_array[i], std_array[i],
                                     [distortion_number, ]) * \
                    distortion_weight
                distortion_array[:, i] = distortion

            for i in range(distortion_array.shape[0]):
                distortion_array[i, :] += np.array(names_value)

            local_distortion_data = \
                pd.DataFrame(index=np.arange(distortion_number),
                             columns=data.columns)

            local_distortion_data["org_name"] =\
                np.full((distortion_number, 1), category_data[
                    ['org_name']].values[index])
            local_distortion_data["mcc"] = \
                np.full((distortion_number, 1),
                        category_data[['mcc']].values[index])
            local_distortion_data["category"] = \
                np.full((distortion_number, 1),
                        category_data[['category']].values[index])
            local_distortion_data["av_flow"] = \
                np.full((distortion_number, 1),
                        category_data[['av_flow']].values[index])
            local_distortion_data["tr_stat"] = \
                np.full((distortion_number, 1),
                        category_data[['tr_stat']].values[index])
            local_distortion_data["category_embedded"] = \
                np.full((distortion_number, 1),
                        category_data[['category_embedded']].values[index])

            local_distortion_data["org_name_embedded"] = \
                local_distortion_data.apply(transformation, raw=False, axis=1)

            distortion_data = \
                distortion_data.append(local_distortion_data).reindex()

        global_distortion_data = \
            global_distortion_data.append(distortion_data).reindex()

    data = data.append(global_distortion_data).reindex()

    return data


def node_data_preprocessing(data):
    input_embedded_set = []
    output_embedded_set = []

    for index in range(len(data)):
        try:
            names_string = data[['org_name_embedded']].values[index][0].\
                    split("[")[1].split("]")[0].split(' ')

            names_value = []

            for value in names_string:
                if value == '':
                    continue

                names_value.append(
                    float(value.split("\n")[0].replace("e-0", "e-")))

            names_value = np.array(names_value)
        except:
            names_value = \
                np.array(data[['org_name_embedded']].values[index][0])

        try:
            tr_statistic_string = data[['tr_stat']].values[index][0]. \
                split("[")[1].split("]")[0].split(' ')

            tr_statistic_value = []

            for value in tr_statistic_string:
                if value == '':
                    continue

                tr_statistic_value.append(float(value.split("\n")[0]))

            tr_statistic_value = np.array(tr_statistic_value)
        except:
            tr_statistic_value = \
                np.array(data[['tr_stat']].values[index][0])

        try:
            category_string = data[['category_embedded']].values[index][0]. \
                split("[")[1].split("]")[0].split(' ')

            category_value = []

            for value in category_string:
                if value == '':
                    continue

                category_value.append(int(value.split("\n")[0]))

            category_value = np.array(category_value)
        except:
            category_value = \
                np.array(data[['category_embedded']].values[index][0])

        av_value = np.array([data[['av_flow']].values[index][0]])
        mcc_code = np.array([data[['mcc']].values[index][0]])

        input_embedded_set.append(np.concatenate((names_value,
                                                  tr_statistic_value,
                                                  av_value,
                                                  mcc_code)))
        output_embedded_set.append(category_value)

    return np.array(input_embedded_set), np.array(output_embedded_set)


def node_sets_creation(data):
    category_names = data[['category']].groupby('category'). \
        size().sort_values(ascending=False).index.to_list()

    training_set, validation_set = \
        pd.DataFrame(columns=data.columns), \
        pd.DataFrame(columns=data.columns)

    for category in category_names:
        data_temp = data[data['category'] == category]

        sets = np.split(data_temp.sample(frac=1, random_state=1),
                        [int(.7 * len(data_temp))])

        training_set = training_set.append(sets[0])
        validation_set = validation_set.append(sets[1])

    training_set_input, training_set_output = \
        node_data_preprocessing(training_set)

    training_set_input, training_set_output = \
        pd.DataFrame(training_set_input), pd.DataFrame(training_set_output)

    validation_set_input, validation_set_output = \
        node_data_preprocessing(validation_set)

    validation_set_input, validation_set_output = \
        pd.DataFrame(validation_set_input), pd.DataFrame(validation_set_output)

    return (training_set_input, training_set_output), \
           (validation_set_input, validation_set_output)
