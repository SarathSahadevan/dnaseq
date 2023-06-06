
from typing import Sequence
from functools import partial
import random
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np



# Set seed for reproducibility
def set_seed(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(13)

# Use this for getting x label
def rand_sequence_var_len(n_seqs: int, lb: int=16, ub: int=128) -> Sequence[int]:
    for i in range(n_seqs):
        seq_len = random.randint(lb, ub)
        yield [random.randint(1, 5) for _ in range(seq_len)]


# Alphabet helpers   
alphabet = 'NACGT'
dna2int = {a: i for a, i in zip(alphabet, range(1, 6))}
int2dna = {i: a for a, i in zip(alphabet, range(1, 6))}
dna2int.update({"pad": 0})
int2dna.update({0: "<pad>"})

intseq_to_dnaseq = partial(map, int2dna.get)
dnaseq_to_intseq = partial(map, dna2int.get)



# Model
class CpGPredictor(torch.nn.Module):
    ''' Simple model that uses a LSTM to count the number of CpGs in a sequence '''

        # TODO complete model, you are free to add whatever layers you need here
        # We do need a lstm and a classifier layer here but you are free to implement them in your way
    def __init__(self, input_size, LSTM_HIDDEN, output_size):
        super(CpGPredictor, self).__init__()
        self.embedding = torch.nn.Embedding(LSTM_LAYER, LSTM_HIDDEN)
        self.lstm = torch.nn.LSTM(LSTM_HIDDEN, LSTM_HIDDEN, batch_first=True)
        self.fc = torch.nn.Linear(LSTM_HIDDEN, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (h_n, _) = self.lstm(embedded)
        logits = self.fc(h_n[-1])
        return logits




alphabet = 'NACGT'
dna2int = {a: i for a, i in zip(alphabet, range(1, 6))}

LSTM_LAYER, LSTM_HIDDEN, output_size = max(dna2int.values())+1 , 32,1
imodel = CpGPredictor(LSTM_LAYER, LSTM_HIDDEN, output_size)  # Instantiate the model

def load_inf(dna):
    tmodel = torch.load("model2.pth")
    imodel.load_state_dict(tmodel)

    imodel.eval()

    testList = list('NACGT')
    # x = 'GAANNNNNCGNAANATACTCGGCCANTNCTNCANTATATATTNCNGNTGCTGATTCGGAACTTACNTAGGAGATTCTANTNNAGNTGTGCN'
    contains_invalid_character = any(character not in testList for character in dna.upper())

    
    if contains_invalid_character:
        print(">>>>>",list(set(dna.upper())))
        res_pred = "DNA  have some issues please check!!"
    else:
        dna= "".join(c for c in dna if c.isalpha())
        dna = dna.upper()

        X = [list(dnaseq_to_intseq(dna))]
        Y = [torch.LongTensor(seq) for seq in X]
        Z = pad_sequence(Y, batch_first=True, padding_value=0)



        with torch.no_grad():
            outputs = imodel(Z)
            res_pred = (outputs.squeeze().tolist())

    return res_pred

