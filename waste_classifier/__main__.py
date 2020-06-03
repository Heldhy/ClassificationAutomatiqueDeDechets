from waste_classifier import pre_processing, training_extractor


def main():
    x_train, y_train, x_test, y_test = pre_processing(type_of_model="mobilenet")
    #train_generator, val_generator = create_new_generator(x_train, y_train, "mobilenet")
    model = training_extractor(x_train, y_train, x_test, y_test, chosen_model="svc", evaluate=True)
    #model, history = fine_tuning(train_generator, val_generator, x_test, y_test, True)
    #evaluate_post_processed_prediction(model, x_test, y_test)


if __name__ == "__main__":
    main()
