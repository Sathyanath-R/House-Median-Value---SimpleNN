# config.py

model_config = {
    "input_dim": 13,
    "hidden_dim": 5,
    "output_dim": 1,
    "lr": 0.01,
    "epochs": 50,
    "batch_size": 2,
    "target_column": "MEDV"
}

data_config = {
    "file_path": "/data/",
    "test_size": 0.2,
    "random_state": 42
}
