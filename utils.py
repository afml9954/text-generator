from tensorflow import keras, expand_dims, squeeze, random
import numpy as np


def generate_text(model, start_string, num):
    # file_dl = os.path.join('data/colombia_91.txt')
    file_dl = keras.utils.get_file('Quijote.txt',
                                      'https://gist.githubusercontent.com/jsdario/6d6c69398cb0c73111e49f1218960f79/raw/8d4fc4548d437e2a7203a5aeeace5477f598827d/el_quijote.txt')

    texto = open(file_dl, 'rb').read().decode(encoding='utf-8')
    # texto = open(file_dl, 'rb').read().decode(encoding='latin-1')
    vocab = sorted(set(texto))
    num_generate = num

    char2idx = {u: i for i, u in enumerate(vocab)}  # asignamos un número a cada vocablo
    idx2char = np.array(vocab)

    input_eval = [char2idx[s] for s in start_string]
    input_eval = expand_dims(input_eval, 0)
    text_generated = []

    temperature = 0.5
    model.reset_states()

    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)
