import torch
from dataset import Shakespeare
from model import CharRNN, CharLSTM

def generate(model, seed_characters, temperature, num_characters, *args):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
				temperature: T
				args: other arguments if needed

    Returns:
        samples: generated characters
    """

    model.eval()
    generated_chars = seed_characters
    input_seq = torch.tensor([model.char_to_index[ch] for ch in seed_characters]).unsqueeze(0).to(device)
    
    # 은닉 상태를 올바른 형태로 초기화
    if isinstance(model, CharRNN):
        hidden = model.init_hidden(1).to(device)
    else:  # CharLSTM
        hidden = (model.init_hidden(1)[0].to(device), model.init_hidden(1)[1].to(device))
    
    for _ in range(num_characters):
        output, hidden = model(input_seq, hidden)
        output_dist = output.data.view(-1).div(temperature).exp()
        top_char = torch.multinomial(output_dist, 1)[0]
        generated_chars += model.index_to_char[top_char.item()]
        input_seq = torch.tensor([[top_char]]).to(device)
    
    return generated_chars

if __name__ == '__main__':
    dataset = Shakespeare('./data/shakespeare_train.txt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rnn_model = CharRNN(dataset.char_to_index, dataset.index_to_char, input_size=len(dataset.chars),
                        hidden_size=32, output_size=len(dataset.chars), num_layers=3).to(device)
    lstm_model = CharLSTM(dataset.char_to_index, dataset.index_to_char, input_size=len(dataset.chars),
                          hidden_size=32, output_size=len(dataset.chars), num_layers=3).to(device)

    rnn_model.load_state_dict(torch.load('best_rnn_model.pt'))
    lstm_model.load_state_dict(torch.load('best_lstm_model.pt'))

    seed_chars = ['H', 'T', 'A', 'M', 'S']
    temperatures = [0.5, 1.0, 2.0]

    for seed in seed_chars:
        print(f"Seed character: {seed}")
        for temp in temperatures:
            rnn_generated_text = generate(rnn_model, seed, temp, 100, device)
            lstm_generated_text = generate(lstm_model, seed, temp, 100, device)

            print(f"Temperature: {temp:.1f}")
            print(f"RNN generated text: {rnn_generated_text}")
            print(f"LSTM generated text: {lstm_generated_text}")
            print()
        print()