# script.py

import torch
import torch.utils.data as data
from config import model_config, data_config
from utils import load_data, preprocess_data, SimpleNN, train_model, evaluate_model
import joblib

if __name__ == "__main__":
    X, y = load_data(data_config['file_path'], data_config['target_column'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=data_config['test_size'], random_state=data_config['random_state'])

    X_train_processed, y_train_processed, preprocessor = preprocess_data(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)
    y_test_processed = y_test

    X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_processed.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_processed.values, dtype=torch.float32).view(-1, 1)

    train_dataset = data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = data.DataLoader(train_dataset, batch_size=model_config['batch_size'], shuffle=True)

    model = SimpleNN(model_config['input_dim'], model_config['hidden_dim'], model_config['output_dim'])
    train_model(model, train_loader, lr=model_config['lr'], epochs=model_config['epochs'])
    evaluate_model(model, X_test_tensor, y_test_tensor)

    joblib.dump(preprocessor, 'preprocessor.pkl')
    torch.save(model.state_dict(), 'model.pth')