import torch
from torch.utils import DataLoader, TensorDataset
from src.neural_network import ValueFunctionNN

def train_neural_network(training_data, config):
    x_data, y_data = zip(*training_data)
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)
    
    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size = config["neural_network"]["nn_batch_size"], shuffle=True)
    
    model = ValueFunctionNN(
        input_dim = config["neural_network"]["nn_input_dim"],
        hidden_dim = config["neural_network"]["nn_hidden_dim"],
        output_dim =config["neural_network"]["nn_output_dim"]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr = config["neural_network"]["nn_learning_rate"])
    loss_fn = torch.nn.MSELoss()
    
    for epoch in range(config["neural_network"]["nn_epochs"]):
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            predictions = model(x_batch).squeeze()
            loss = loss_fn(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
    return model