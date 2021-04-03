import tree_neural_network
import data_management


def main():
    data = data_management.data_loading("data/input_data/CFD_final_4.csv")

    data_management.data_statistics(data)

    validation_set = data_management.validation_set_creation(data)

    neural_tree = tree_neural_network.TreeNeuralNetwork()

    neural_tree.training(data)

    neural_tree.evaluation(validation_set)

    return


if __name__ == "__main__":
    main()
