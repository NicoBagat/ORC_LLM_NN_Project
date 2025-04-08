import torch
from torch.utils import DataLoader, TensorDataset
from src.neural_network import ValueFunctionNN

def train_neural_network(training_data, config):
    x_data, y_data = zip(*training_data)
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)
    
    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size = config["batch_size"], shuffle=True)
    
    model = ValueFunctionNN(
        input_dim = config["state_dim"],
        hidden_dim = config["nn_hidden_dim"],
        output_dim = 1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr = config["learning_rate"])
    loss_fn = torch.nn.MSELoss()
    
    for epoch in range(config["epochs"]):
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            predictions = model(x_batch).squeeze()
            loss = loss_fn(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
    return model