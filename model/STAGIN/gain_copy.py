import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from tqdm import tqdm

from util.utils import normalization, renormalization, rounding
from util.utils import xavier_init
from util.utils import binary_sampler, uniform_sampler, sample_batch_index

def spatial_attention(inputs):
    # Fully connected layer followed by tanh and softmax
    fc_layer = tf.keras.layers.Dense(inputs.shape[-1], activation='tanh')(inputs)
    attention_weights = tf.nn.softmax(fc_layer, axis=1)
    return inputs * attention_weights

def temporal_attention(data):
    with tf.variable_scope('temporal_attention', reuse=tf.AUTO_REUSE):
        fc = tf.keras.layers.Dense(units=data.shape[-1], activation=tf.nn.tanh)(data)
        attention_weights = tf.nn.softmax(fc, axis=1)
        weighted_data = data * attention_weights
        weighted_sum = tf.reduce_sum(weighted_data, axis=1)
        weighted_sum = tf.expand_dims(weighted_sum, axis=1)  # Add a dimension to keep the shape consistent
    return weighted_sum

def gain(data_x, gain_parameters):
    '''Impute missing values in data_x

    Args:
      - data_x: original data with missing values
      - gain_parameters: GAIN network parameters:
        - batch_size: Batch size
        - hint_rate: Hint rate
        - alpha: Hyperparameter
        - iterations: Iterations

    Returns:
      - imputed_data: imputed data
    '''
    # Define mask matrix
    data_m = 1 - np.isnan(data_x)
    # System parameters
    batch_size = gain_parameters['batch_size']
    hint_rate = gain_parameters['hint_rate']
    alpha = gain_parameters['alpha']
    iterations = gain_parameters['iterations']
    # Other parameters
    N, T, dim = data_x.shape
    # Hidden state dimensions
    h_dim = int(dim)
    # Normalization
    norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)
    ## GAIN architecture   
    # Input placeholders
    # Data vector
    X = tf.placeholder(tf.float32, shape=[None, None, dim])
    # Mask vector 
    M = tf.placeholder(tf.float32, shape=[None, None, dim])
    # Hint vector
    H = tf.placeholder(tf.float32, shape=[None, None, dim])
    # Discriminator variables
    D_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))  # Data + Hint as inputs
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))
    D_W3 = tf.Variable(xavier_init([h_dim, dim]))
    D_b3 = tf.Variable(tf.zeros(shape=[dim]))  # Multi-variate outputs
    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
    # Generator variables
    # Data + Mask as inputs (Random noise is in missing components)
    G_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W3 = tf.Variable(xavier_init([h_dim, dim]))
    G_b3 = tf.Variable(tf.zeros(shape=[dim]))
    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    ## GAIN functions
    # Generator
    def generator(x, m):
        inputs = tf.concat(values=[x, m], axis=-1)
        spatial_attention_output = spatial_attention(inputs)
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=h_dim, activation=tf.nn.tanh)
        lstm_outputs, _ = tf.nn.dynamic_rnn(lstm_cell, spatial_attention_output, dtype=tf.float32)
        temporal_attention_output = temporal_attention(lstm_outputs)
        G_h1 = tf.nn.relu(tf.matmul(tf.reshape(temporal_attention_output, [-1, dim * 2]), G_W1) + G_b1)
        # G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
        # MinMax normalized output
        G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
        return G_prob

    # Discriminator
    def discriminator(x, h):
        # Concatenate Data and Hint
        inputs = tf.concat(values=[x, h], axis=-1)
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_logit = tf.matmul(D_h2, D_W3) + D_b3
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob

    ## GAIN structure
    # Generator
    G_sample = generator(X, M)
    G_sample = spatial_attention(G_sample)  # Apply spatial attention
    G_sample = tf.keras.layers.LSTM(h_dim, return_sequences=True)(G_sample)  # Apply LSTM
    G_sample = temporal_attention(G_sample)  # Apply temporal attention

    # Combine with observed data
    Hat_X = X * M + G_sample * (1 - M)

    # Discriminator
    D_prob = discriminator(Hat_X, H)

    ## GAIN loss
    D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) \
                                  + (1 - M) * tf.log(1. - D_prob + 1e-8))

    G_loss_temp = -tf.reduce_mean((1 - M) * tf.log(D_prob + 1e-8))

    MSE_loss = \
        tf.reduce_mean((M * X - M * G_sample) ** 2) / tf.reduce_mean(M)

    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss

    ## GAIN solver
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

    ## Iterations
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Start Iterations
    for it in tqdm(range(iterations)):

        # Sample batch
        batch_idx = sample_batch_index(N, batch_size)
        X_mb = norm_data_x[batch_idx, :, :]
        M_mb = data_m[batch_idx, :, :]
        # Sample random vectors  
        Z_mb = uniform_sampler(0, 0.01, batch_size, T, dim)
        # Sample hint vectors
        H_mb_temp = binary_sampler(data_x, batch_size, dim)
        H_mb_temp = H_mb_temp[batch_idx, :, :]
        H_mb = M_mb * H_mb_temp

        # Combine random vectors with observed vectors
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        _, D_loss_curr = sess.run([D_solver, D_loss_temp],
                                  feed_dict={M: M_mb, X: X_mb, H: H_mb})
        _, G_loss_curr, MSE_loss_curr = \
            sess.run([G_solver, G_loss_temp, MSE_loss],
                     feed_dict={X: X_mb, M: M_mb, H: H_mb})

    ## Return imputed data      
    Z_mb = uniform_sampler(0, 0.01, N, T, dim)
    M_mb = data_m
    X_mb = norm_data_x
    X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

    imputed_data = sess.run([G_sample], feed_dict={X: X_mb, M: M_mb})[0]

    imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data

    # Renormalization
    imputed_data = renormalization(imputed_data, norm_parameters)

    # Rounding
    imputed_data = rounding(imputed_data, data_x)

    return imputed_data

import tensorflow as tf
# IF USING TF 2 use the following import to still use TF < 2.0 functionalities
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from tqdm import tqdm

from util.utils import normalization, renormalization, rounding
from util.utils import xavier_init
from util.utils import binary_sampler, uniform_sampler, sample_batch_index

def temporal_attention(inputs):
    # Shape: [batch_size, T, dim]
    # Ensure that the input shape is compatible
    T =180
    dim =36
    # Learnable weights for each time step and each feature
    weights = tf.Variable(tf.ones([1, T, dim]), trainable=True)
    # Apply the weights to the input
    weighted_inputs = inputs * weights
    return weighted_inputs

def gain(data_x, gain_parameters):
    '''Impute missing values in data_x
    
    Args:
      - data_x: original data with missing values
      - gain_parameters: GAIN network parameters:
        - batch_size: Batch size
        - hint_rate: Hint rate
        - alpha: Hyperparameter
        - iterations: Iterations
        
    Returns:
      - imputed_data: imputed data
    '''
    # Define mask matrix
    data_m = 1 - np.isnan(data_x)
    # System parameters
    batch_size = gain_parameters['batch_size']
    hint_rate = gain_parameters['hint_rate']
    alpha = gain_parameters['alpha']
    iterations = gain_parameters['iterations']
    
    # Other parameters
    N, T, dim = data_x.shape
    # Hidden state dimensions
    h_dim = int(dim)
    
    # Normalization
    norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)
    
    ## GAIN architecture   
    # Input placeholders
    # Data vector
    X = tf.placeholder(tf.float32, shape=[None, None, dim])
    # Mask vector 
    M = tf.placeholder(tf.float32, shape=[None, None, dim])
    # Hint vector
    H = tf.placeholder(tf.float32, shape=[None, None, dim])
    
    # Discriminator variables
    D_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))  # Data + Hint as inputs
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    
    D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))
    
    D_W3 = tf.Variable(xavier_init([h_dim, dim]))
    D_b3 = tf.Variable(tf.zeros(shape=[dim]))  # Multi-variate outputs
    
    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
    
    # Generator variables
    # Data + Mask as inputs (Random noise is in missing components)
    G_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))  
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    
    G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))
    
    G_W3 = tf.Variable(xavier_init([h_dim, dim]))
    G_b3 = tf.Variable(tf.zeros(shape=[dim]))
    
    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
    
    ## GAIN functions
    # Generator with temporal attention
    def generator(x, m):
        inputs = tf.concat(values=[x, m], axis=-1) 
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
        
        # Apply temporal attention
        G_h2_attention = temporal_attention(G_h2)
        
        # MinMax normalized output
        G_prob = tf.nn.sigmoid(tf.matmul(G_h2_attention, G_W3) + G_b3)
        return G_prob

    # Discriminator
    def discriminator(x, h):
        # Concatenate Data and Hint
        inputs = tf.concat(values=[x, h], axis=-1) 
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)  
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_logit = tf.matmul(D_h2, D_W3) + D_b3
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob
    
    ## GAIN structure
    # Generator
    G_sample = generator(X, M)
   
    # Combine with observed data
    Hat_X = X * M + G_sample * (1 - M)
    
    # Discriminator
    D_prob = discriminator(Hat_X, H)
    
    ## GAIN loss
    D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) +
                                  (1 - M) * tf.log(1. - D_prob + 1e-8)) 
    
    G_loss_temp = -tf.reduce_mean((1 - M) * tf.log(D_prob + 1e-8))
    
    MSE_loss = tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)
    
    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss 
    
    ## GAIN solver
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
    
    ## Iterations
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
     
    # Start Iterations
    for it in tqdm(range(iterations)):    
        
        # Sample batch
        batch_idx = sample_batch_index(N, batch_size)
        X_mb = norm_data_x[batch_idx, :, :]  
        M_mb = data_m[batch_idx, :, :]  
        # Sample random vectors  
        Z_mb = uniform_sampler(0, 0.01, batch_size, T, dim) 
        # Sample hint vectors
        H_mb_temp = binary_sampler(data_x, batch_size, dim)
        H_mb_temp = H_mb_temp[batch_idx, :, :]
        H_mb = M_mb * H_mb_temp
        
        # Combine random vectors with observed vectors
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb 
        
        _, D_loss_curr = sess.run([D_solver, D_loss_temp], 
                                  feed_dict={M: M_mb, X: X_mb, H: H_mb})
        _, G_loss_curr, MSE_loss_curr = \
        sess.run([G_solver, G_loss_temp, MSE_loss],
                 feed_dict={X: X_mb, M: M_mb, H: H_mb})
              
    ## Return imputed data      
    Z_mb = uniform_sampler(0, 0.01, N, T, dim) 
    M_mb = data_m
    X_mb = norm_data_x          
    X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb 
        
    imputed_data = sess.run([G_sample], feed_dict={X: X_mb, M: M_mb})[0]
    
    imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data
    
    # Renormalization
    imputed_data = renormalization(imputed_data, norm_parameters)  
    
    # Rounding
    imputed_data = rounding(imputed_data, data_x)  
            
    return imputed_data