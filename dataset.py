import torch
from torch.utils.data import Dataset

class Shakespeare(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
					 You need this dictionary to generate characters.
				2) Make list of character indices using the dictionary
				3) Split the data into chunks of sequence length 30. 
           You should create targets appropriately.
    """

    def __init__(self, input_file, sequence_length=30):
        with open(input_file, 'r') as f:
            self.data = f.read()
        self.chars = sorted(list(set(self.data)))
        self.char_to_index = {ch: i for i, ch in enumerate(self.chars)}
        self.index_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.indices = [self.char_to_index[ch] for ch in self.data]
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.indices) - self.sequence_length

    def __getitem__(self, idx):
        input_seq = torch.tensor(self.indices[idx:idx+self.sequence_length])
        target_seq = torch.tensor(self.indices[idx+1:idx+self.sequence_length+1])
        return input_seq, target_seq

if __name__ == '__main__':
    dataset = Shakespeare('shakespeare.txt')
    print(f"Dataset length: {len(dataset)}")
    input_seq, target_seq = dataset[0]
    print(f"Input sequence shape: {input_seq.shape}")
    print(f"Target sequence shape: {target_seq.shape}")