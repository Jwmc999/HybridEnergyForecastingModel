import pandas as pd
import tensorflow as tf

# Sample generator
def generator(dataset, input_width, label_width, dropnan=True):
  """
  split a multivariate sequence into multi-lag samples

  Attributes:
  input_width (int): number of lag observation as input (X)
  label_width (int): number of observations as output (y)
  n_features (int): number of variables

  Returns:
  dataX (tensor): input
  dataY (tensor): label
  """
  # number of variables
  n_vars = 1 if type(dataset) is list else dataset.shape[1]
  df = pd.DataFrame(dataset)
  cols, names = [], []
  # input sequence (t-n, ..., t-1)
  for i in range(input_width, 0, -1):
    cols.append(df.shift(i))
    names += [("var%d(t-%d)" % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ..., t+n)
  for i in range(0, label_width):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
      names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
  # put it all together
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  if dropnan:
    agg.dropna(inplace=True)
  return agg
  
def get_model(name, units, input_width, label_width, n_features, dropout):
  """
  Model of choice

  Attributes:
  name (str): name of model. {'linear': linearLayer, 'rnn': rnnLayer, 'lstm': lstmLayer, 'cnn-lstm': cnnlstmLayer}
  units (int): dimension of models
  input_width (int): length of input (x)
  label_width (int): length of output (y_pred)
  n_features (int): number of variables
  dropout (float): Dropout rate

  Returns:
  model (class): keras model

  """
  tf.keras.backend.clear_session()  
  inputs = tf.keras.Input(shape=(input_width, n_features))
  opt = tf.keras.optimizers.Adam(learning_rate=1e-6)

  if name=="linear":
    m = linLayer(label_width)
  elif name == "rnn":
    m = rnnLayer(units, label_width, dropout)
  elif name == "lstm":
    m = lstmLayer(units, label_width, dropout)
  elif name == "cnn-lstm":
    m = cnnlstmLayer(units, label_width, dropout)
  else:
    print("please put valid model")

  outputs = m(inputs)
  model = tf.keras.Model(inputs, outputs)
  model.compile(loss='mse', optimizer=opt)
  return model

## MLP: Simple Linear Regression
class linLayer(tf.keras.layers.Layer):
  """
  Simple Linear Regression layers
  
  We will use it as single-step

  Attributes:
  step (int): number of output features (label_width)
  """
  def __init__(self, step):
    super(linLayer, self).__init__()
    self.step = step
    self.linear_layer = tf.keras.layers.Dense(step, activation="linear")

  def call(self, inputs):
    outputs = self.linear_layer(inputs)
    return outputs

## Multistep RNN
class rnnLayer(tf.keras.layers.Layer):
  """
  Multistep RNN layers
  
  We will use it as single-step

  Attributes:
  units (int): Dimension of LSTM
  step (int): length of output sequence (label_width)
  dropout (float): Dropout rate
  """
  def __init__(self, units, step, dropout):
    super(rnnLayer, self).__init__()
    self.units = units
    self.step = step
    self.rnn_layer = tf.keras.layers.SimpleRNN(units)
    self.dropout = tf.keras.layers.Dropout(dropout)
    self.dense = tf.keras.layers.Dense(step, activation='relu')
  
  def call(self, inputs, training=None):
    outputs = self.rnn_layer(inputs)
    outputs = self.dropout(outputs)
    outputs = self.dense(outputs)
    outputs = tf.expand_dims(outputs, -1)
    return outputs

## Multistep LSTM
class lstmLayer(tf.keras.layers.Layer):
  """
  Multistep LSTM layers
  
  We will use it as single-step
  
  """
  def __init__(self, units, step, dropout):
    super(lstmLayer, self).__init__()
    self.units = units
    self.step = step
    self.lstm_layer = tf.keras.layers.LSTM(units)
    self.dropout = tf.keras.layers.Dropout(dropout)
    self.dense = tf.keras.layers.Dense(step, activation='relu')
  
  def call(self, inputs, training=None):
    outputs = self.lstm_layer(inputs)
    outputs = self.dropout(outputs)
    outputs = self.dense(outputs)
    outputs = tf.expand_dims(outputs, -1)
    return outputs

## Multistep CNN-LSTM
class cnnlstmLayer(tf.keras.layers.Layer):
  """
  Multistep CNN-LSTM layers
  
  We will use it as single-step

  """
  def __init__(self, units, step, dropout):
    super(cnnlstmLayer, self).__init__()
    self.units = units
    self.step = step
    self.cnn1_layer = tf.keras.layers.Conv1D(filters=128, kernel_size=1, activation='relu')
    self.cnn2_layer = tf.keras.layers.Conv1D(filters=128, kernel_size=1, activation='relu')   
    self.pool = tf.keras.layers.MaxPooling1D()
    self.lstm_layer = tf.keras.layers.LSTM(units)
    self.dropout = tf.keras.layers.Dropout(dropout)
    self.dense = tf.keras.layers.Dense(step, activation='relu')
  
  def call(self, inputs, training=None):
    outputs = self.cnn1_layer(inputs)
    outputs = self.cnn2_layer(outputs)
    outputs = self.pool(outputs)
    outputs = self.lstm_layer(outputs)
    outputs = self.dropout(outputs)
    outputs = self.dense(outputs)
    outputs = tf.expand_dims(outputs, -1)
    return outputs