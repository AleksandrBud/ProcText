import numpy as np
import tensorflow as tf
import os
import pickle


class generate_text:


    def init_model(self, path_to_file, seq_length=100, BATCH_SIZE=64, BUFFER_SIZE=10000):
        text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
        self.vocab = sorted(set(text))
        self.char2idx = {u: i for i, u in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab)

        text_as_int = np.array([self.char2idx[c] for c in text])
        examples_per_epoch = len(text) // (seq_length + 1)

        char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
        sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)


        def split_input_target(chunk):
            input_text = chunk[:-1]
            target_text = chunk[1:]
            return input_text, target_text

        dataset = sequences.map(split_input_target)
        self.batch_size = BATCH_SIZE
        self.vocab_size = len(self.vocab)
        self.dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


    def build_model(self, i_vocab_size, i_embedding_dim, i_rnn_units, i_batch_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(i_vocab_size, i_embedding_dim,
                                      batch_input_shape=[i_batch_size, None]),

            tf.keras.layers.GRU(i_rnn_units,
                                return_sequences=True,
                                stateful=True,
                                recurrent_initializer='glorot_uniform'),

            tf.keras.layers.GRU(i_rnn_units,
                                return_sequences=True,
                                stateful=True,
                                recurrent_initializer='glorot_uniform'),

            tf.keras.layers.GRU(i_rnn_units,
                                return_sequences=True,
                                stateful=True,
                                recurrent_initializer='glorot_uniform'),

            tf.keras.layers.Dense(i_vocab_size)
        ])
        return model


    def create_model(self, epochs = 40, period=20, embedding_dim=300, rnn_units=512, checkpoint_dir='./train_model'):
        if not os.path.exists(checkpoint_dir):
          os.mkdir(checkpoint_dir)
        model = self.build_model(
            i_vocab_size=self.vocab_size,
            i_embedding_dim=embedding_dim,
            i_rnn_units=rnn_units,
            i_batch_size=self.batch_size)

        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.checkpoint_dir = checkpoint_dir


        def loss(labels, logits):
            return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

        model.compile(optimizer='adam', loss=loss)
        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            period=period,
            save_weights_only=True)
        history = model.fit(self.dataset, epochs=epochs, callbacks=[checkpoint_callback])

        self.load_model(embedding_dim, rnn_units, checkpoint_dir)


    def save_model(self, save_path):
      if not os.path.exists(save_path):
        os.mkdir(save_path)
      params = [str(self.batch_size), str(self.vocab_size), str(self.embedding_dim),
                str(self.rnn_units)]
      path_model = tf.train.latest_checkpoint(self.checkpoint_dir)
      model_name = str(path_model).split('/')[-1]
      comand = f'cp {path_model+ ".index"} {save_path + model_name + ".index"}'
      os.system(comand)
      comand = f'cp {path_model + ".data-00000-of-00001"} {save_path + model_name + ".data-00000-of-00001"}'
      os.system(comand)
      comand = f'cp {self.checkpoint_dir + "/checkpoint"} {save_path + "checkpoint"}'
      os.system(comand)
      with open(save_path + 'char2idx.pkl', 'wb') as f:
        pickle.dump(self.char2idx, f)
      with open(save_path + 'idx2char.pkl', 'wb') as f:
        pickle.dump(self.idx2char, f)
      with open(save_path + 'params.pkl', 'wb') as f:
        pickle.dump(' '.join(params), f)


    def download_model(self, load_path):
      with open(load_path + 'params.pkl', 'rb') as f:
        params = pickle.load(f).split(' ')
      with open(load_path + 'char2idx.pkl', 'rb') as f:
        self.char2idx = pickle.load(f)
      with open(load_path + 'idx2char.pkl', 'rb') as f:
        self.idx2char = pickle.load(f)
      self.batch_size = int(params[0])
      self.vocab_size = int(params[1])
      self.embedding_dim = int(params[2])
      self.rnn_units = int(params[3])
      self.load_model(self.embedding_dim, self.rnn_units, load_path)



    def load_model(self, embedding_dim=300, rnn_units=512, checkpoint_dir='./train_model'):
        model = self.build_model(
            i_vocab_size=self.vocab_size,
            i_embedding_dim=embedding_dim,
            i_rnn_units=rnn_units,
            i_batch_size=self.batch_size)

        model = self.build_model(self.vocab_size, embedding_dim, rnn_units, i_batch_size=1)
        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        model.build(tf.TensorShape([1, None]))
        self.model = model


    def generate_text(self, start_string, num_generate=500, temperature=0.1):
      # Converting our start string to numbers (vectorizing)
      input_eval = [self.char2idx[s] for s in start_string]
      input_eval = tf.expand_dims(input_eval, 0)
      # Empty string to store our results
      text_generated = []

      # Here batch size == 1
      self.model.reset_states()
      for i in range(num_generate):
          predictions = self.model(input_eval)
          predictions = tf.squeeze(predictions, 0)
          # using a categorical distribution to predict the character returned by the model
          predictions = predictions / temperature
          predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
          # Pass the predicted character as the next input to the model
          # along with the previous hidden state
          input_eval = tf.expand_dims([predicted_id], 0)
          text_generated.append(self.idx2char[predicted_id])
      return (start_string + ''.join(text_generated))