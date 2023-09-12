import numpy as np
import os
from tensorflow import keras
from keras_bert import get_base_dict, get_model, compile_model, gen_batch_inputs
import tensorflow as tf

### Parameters ###

# Data directory
filepath = 'test_model14' # The model name or the directory to save model
data_dir = 'train_data' # Directory where data is saved

# Model hyperparameters
head_num = 8
transformer_num = 12
embed_dim = 64
feed_forward_dim = 512
seq_len = 19
dropout_rate = 0.05


# Masking hyperparameters
mask_rate = 0.5
mask_mask_rate = 0.8
mask_random_rate = 0.1
swap_sentence_rate = 0.3

# Training hyperparameters
batch_size = 5000
n_epochs = 500
learning_rate = 1e-4 # Initial learning rate, decay is applied



### Load the data and sort them into decreasing PT pairs ###
data = np.load(f'{data_dir}/inputs/train.npz', allow_pickle=True)
seq = data['seq']
true_pt = data['true_pt']
print('>>> Number of sequences:', len(seq))
true_seq = []

for i in range(len(seq)):
    temp = []
    for j in range(len(seq[i])):
        temp.append(seq[i][j])
    true_seq.append(temp)
    
sentence_pairs = []
for i in range(0, len(true_seq)-1, 2):
    if true_pt[i] >= true_pt[i+1]:
        sentence_pairs.append([true_seq[i], true_seq[i+1]])
    else:
        sentence_pairs.append([true_seq[i+1], true_seq[i]])
    
### Build token dictionary ###
token_dict = get_base_dict()  # A dict that contains some special tokens
for pairs in sentence_pairs:
    for token in pairs[0] + pairs[1]:
        if token not in token_dict:
            token_dict[token] = len(token_dict)
token_list = list(token_dict.keys())  # Used for selecting a random word

### Build & train the model ###
model = get_model(
    token_num=len(token_dict),
    head_num=head_num,
    transformer_num=transformer_num,
    embed_dim=embed_dim,
    feed_forward_dim=feed_forward_dim,
    seq_len=seq_len,
    pos_num=seq_len,
    dropout_rate=dropout_rate,
)


steps_per_epoch = max(len(sentence_pairs) // batch_size, 1000)
decay_steps = steps_per_epoch * n_epochs
compile_model(model, decay_steps=decay_steps, learning_rate=learning_rate)

model.summary()
if os.path.exists(filepath):
    try:
        model.load_weights(filepath)
        print(">>> Loading model at", filepath)
    except OSError:
        print(">>> Creating new model at", filepath)
    
        
def _generator(batch_size=5000, mask_rate=0.3, seq_len=19):
    total_size = len(sentence_pairs)
    i = 0
    while True:
        yield gen_batch_inputs(
                     sentence_pairs[i*batch_size:],
                     token_dict,
                     token_list,
                     batch_size=batch_size,
                     seq_len=seq_len,
                     mask_rate=mask_rate,
                     mask_mask_rate=mask_mask_rate,
                     mask_random_rate=mask_random_rate,
                     swap_sentence_rate=swap_sentence_rate,
                     force_mask=True,
        )
        if (i+1) * batch_size >= total_size:
            i = 0
        else:
            i += 1


model.fit_generator(
    generator=_generator(batch_size=batch_size, mask_rate=mask_rate),
    steps_per_epoch=steps_per_epoch,
    epochs=n_epochs,
    validation_data=_generator(),
    validation_steps=100,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
        keras.callbacks.ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True),
        keras.callbacks.TensorBoard(log_dir=filepath)
    ],
)

model.save(filepath)