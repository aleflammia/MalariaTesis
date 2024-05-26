class Config:
    train_data_path = "Dataset2/train"
    valid_data_path = "Dataset2/valid"
    test_data_path = "Dataset2/test"
    train_txt_path = f"{train_data_path}/train.txt"
    valid_txt_path = f"{valid_data_path}/valid.txt"
    test_txt_path = f"{test_data_path}/test.txt"
    batch_size = 256
    num_classes = 3
    num_workers = 4
    max_epochs = 20
    lr = 0.0005  