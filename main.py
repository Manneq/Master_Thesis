import tree_classifier
import data_management


def main():
    """
        Main function.
    """
    # Data import
    data = data_management.data_loading("data/input_data/CFD_final_4.csv")

    # Data statistics printing
    #data_management.data_statistics(data)

    # Validation set creation
    validation_set = data_management.validation_set_creation(data)

    # Hierarchical tree classifier creation
    neural_tree = tree_classifier.HierarchicalTreeClassifier()

    # Hierarchical tree training
    neural_tree.training(data)

    # Hierarchical tree evaluation
    neural_tree.evaluation(validation_set)

    return


if __name__ == "__main__":
    main()
