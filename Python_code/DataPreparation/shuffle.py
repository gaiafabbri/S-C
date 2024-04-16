import numpy as np

#fai una funzione che faccia shuffle e splitting: però attenzione meglio separarle perchè lo shuffle non serve sempre quindi pensaci un attimo magari 
    # data shuffle
def apply_shuffle(data):
    random_state=42
    np.random.seed(random_state)
    shuffle_index = np.random.permutation(len(data))
    data_shuffled=data[shuffle_index]
    
    return data_shuffled


