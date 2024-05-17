import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataset import Shakespeare
from model import CharRNN, CharLSTM
import matplotlib.pyplot as plt

# import some packages you need here


def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
    """

    model.train()
    trn_loss = 0
    for batch, (inputs, targets) in enumerate(trn_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        if isinstance(model, CharRNN):
            hidden = model.init_hidden(inputs.size(0)).to(device)
            outputs, _ = model(inputs, hidden)
        else:
            hidden, cell = model.init_hidden(inputs.size(0))
            hidden, cell = hidden.to(device), cell.to(device)
            outputs, _ = model(inputs, (hidden, cell))
        optimizer.zero_grad()
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        trn_loss += loss.item()
    return trn_loss / len(trn_loader)

def validate(model, val_loader, device, criterion):
    """ Validate function

    Args:
        model: network
        val_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        val_loss: average loss value
    """

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if isinstance(model, CharRNN):
                hidden = model.init_hidden(inputs.size(0)).to(device)
                outputs, _ = model(inputs, hidden)
            else:
                hidden, cell = model.init_hidden(inputs.size(0))
                hidden, cell = hidden.to(device), cell.to(device)
                outputs, _ = model(inputs, (hidden, cell))
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            val_loss += loss.item()
    return val_loss / len(val_loader)


def main():
    """ Main function

        Here, you should instantiate
        1) DataLoaders for training and validation. 
           Try SubsetRandomSampler to create these DataLoaders.
        3) model
        4) optimizer
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Shakespeare('./data/shakespeare_train.txt')
    split_ratio = 0.8
    num_samples = len(dataset)
    indices = list(range(num_samples))
    split = int(num_samples * split_ratio)
    trn_idx, val_idx = indices[:split], indices[split:]
    trn_sampler = SubsetRandomSampler(trn_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    trn_loader = DataLoader(dataset, batch_size=64, sampler=trn_sampler)
    val_loader = DataLoader(dataset, batch_size=64, sampler=val_sampler)
    
    input_size = len(dataset.chars)
    hidden_size = 32
    output_size = len(dataset.chars)
    num_layers = 3
    
    rnn_model = CharRNN(dataset.char_to_index, dataset.index_to_char, input_size, hidden_size, output_size, num_layers).to(device)
    lstm_model = CharLSTM(dataset.char_to_index, dataset.index_to_char, input_size, hidden_size, output_size, num_layers).to(device)
    
    criterion = nn.CrossEntropyLoss()
    rnn_optimizer = optim.AdamW(rnn_model.parameters(), lr=0.001, weight_decay=0.01)
    lstm_optimizer = optim.AdamW(lstm_model.parameters(), lr=0.001, weight_decay=0.01)

    best_rnn_val_loss = float('inf')
    best_lstm_val_loss = float('inf')
    best_rnn_model = None
    best_lstm_model = None

    rnn_trn_losses = []
    rnn_val_losses = []
    lstm_trn_losses = []
    lstm_val_losses = []

    num_epochs = 15
    for epoch in range(num_epochs):
        rnn_trn_loss = train(rnn_model, trn_loader, device, criterion, rnn_optimizer)
        rnn_val_loss = validate(rnn_model, val_loader, device, criterion)
        lstm_trn_loss = train(lstm_model, trn_loader, device, criterion, lstm_optimizer)
        lstm_val_loss = validate(lstm_model, val_loader, device, criterion)

        rnn_trn_losses.append(rnn_trn_loss)
        rnn_val_losses.append(rnn_val_loss)
        lstm_trn_losses.append(lstm_trn_loss)
        lstm_val_losses.append(lstm_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"RNN Train Loss: {rnn_trn_loss:.4f}, "
              f"RNN Val Loss: {rnn_val_loss:.4f}, "
              f"LSTM Train Loss: {lstm_trn_loss:.4f}, "
              f"LSTM Val Loss: {lstm_val_loss:.4f}")

        if rnn_val_loss < best_rnn_val_loss:
            best_rnn_val_loss = rnn_val_loss
            best_rnn_model = rnn_model

        if lstm_val_loss < best_lstm_val_loss:
            best_lstm_val_loss = lstm_val_loss
            best_lstm_model = lstm_model

    print(f"Best RNN Val Loss: {best_rnn_val_loss:.4f}")
    print(f"Best LSTM Val Loss: {best_lstm_val_loss:.4f}")
    torch.save(best_rnn_model.state_dict(), 'best_rnn_model.pt')
    torch.save(best_lstm_model.state_dict(), 'best_lstm_model.pt')

    # Plot train losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), rnn_trn_losses, label='RNN')
    plt.plot(range(1, num_epochs + 1), lstm_trn_losses, label='LSTM')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.legend()
    plt.title('Train Loss Comparison')
    plt.savefig('train_loss_plot.png')
    plt.close()

    # Plot validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), rnn_val_losses, label='RNN')
    plt.plot(range(1, num_epochs + 1), lstm_val_losses, label='LSTM')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.title('Validation Loss Comparison')
    plt.savefig('val_loss_plot.png')
    plt.close()

if __name__ == '__main__':
    main()