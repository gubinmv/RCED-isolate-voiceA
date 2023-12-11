# модуль функций ошибки
# автор: Губин М.В.

import tensorflow as tf

def my_loss_function(y_true, y_pred):
    my_loss = tf.nn.l2_loss(y_true - y_pred) / batch_size

    return my_loss


def my_get_SDR(y_true, y_pred):
    my_loss = tf.nn.l2_loss(y_true - y_pred)
    my_loss2 = tf.nn.l2_loss(y_true)
    my_loss= 10*tf.math.log(my_loss2/my_loss)/2.30258

    return my_loss

def my_get_SDR_1(y_true, y_pred):
    my_loss = tf.nn.l2_loss(y_true - y_pred)
    my_loss2 = tf.nn.l2_loss(y_true)
    my_loss= 10*tf.math.log(my_loss2/my_loss)/2.30258

    return -my_loss

def my_get_SDR_2(y_true, y_pred, eps=1e-8):
    loss_ = tf.nn.l2_loss(y_true)*2
    loss_vt = tf.math.multiply(y_true, y_pred) / loss_ + eps
    loss_vt = loss_vt * y_true
    loss_et = y_pred - loss_vt
    loss_vtn = tf.nn.l2_loss(loss_vt)*2
    loss_etn = tf.nn.l2_loss(loss_et)*2
    my_loss= 10 * tf.math.log(loss_/loss_etn + eps)/2.30258

    return -my_loss

def my_get_SDR_3(y_true, y_pred, eps=1e-8):
    loss_ = tf.nn.l2_loss(y_true)*2
    loss_vt = tf.math.multiply(y_true, y_pred) / loss_ + eps

    my_loss = tf.nn.l2_loss(y_true - y_pred)
    my_loss2 = tf.nn.l2_loss(y_true)
    #my_loss= 10*tf.math.log(my_loss2/my_loss)/2.30258 + 10*tf.math.log(loss_vt**2)/2.30258
    my_loss= 10*tf.math.log(loss_vt**2)/2.30258
    #if (my_loss >= 0): my_loss = - my_loss

    return my_loss

def loss_l2(y_true, y_pred):
    my_loss = tf.nn.l2_loss(y_true - y_pred) / batch_size

    return my_loss


def loss_SI_SNR(y_true, y_pred):
    s_target = (y_true - y_pred)*y_true/tf.norm(y_true, ord=2)
    e_noise = y_pred - s_target

    z = 10*tf.math.log(tf.norm(s_target, ord=2)/tf.norm(e_noise, ord=2))

    return z

def loss_SI_SDR(y_true, y_pred):

    v_t = tf.math.multiply(y_true, y_pred)*y_true/tf.norm(y_true, ord=2)
    e_t = y_pred - v_t

    z = 10*tf.math.log(tf.norm(v_t, ord=2)/tf.norm(e_t, ord=2))

    return z

def my_loss_SI_SDR(y_true, y_pred):

##    v_t = y_true*((y_true - y_pred)**2)/tf.norm(y_true, ord=2)
    v_t = tf.nn.l2_loss(y_true - y_pred)/tf.norm(y_true, ord=2)
    e_t = y_pred - v_t

    z = 10*tf.math.log(tf.norm(v_t, ord=2)/tf.norm(e_t, ord=2))

    return -z

def my_loss_SI_SDR_2(y_true, y_pred):

##    v_t = y_true*((y_true - y_pred)**2)/tf.norm(y_true, ord=2)
    v_t = tf.nn.l2_loss(y_true - y_pred)/tf.norm(y_true, ord=2)
    e_t = y_pred - v_t

    z = 10*tf.math.log(tf.norm(v_t, ord=2)/tf.norm(e_t, ord=2))

    my_loss = tf.nn.l2_loss(y_true - y_pred)
    my_loss2 = tf.nn.l2_loss(y_true)
    my_loss= 10*tf.math.log(my_loss2/my_loss)/2.30258

    return - my_loss - z * 0.1

def my_loss_STFT(y_true, y_pred):
    e_ = 0.0001
    Lsc = tf.norm(tf.math.abs(y_true) - tf.math.abs(y_pred), ord=2)/tf.norm(tf.math.abs(y_true), ord=2)
    Lmag = tf.norm(tf.math.log(tf.math.abs(y_true)/(tf.math.abs(y_pred)+e_)), ord=1)

    return Lsc + Lmag/10000

def my_loss_STFT_2(y_true, y_pred):

    x_ = tf.summary(tf.norm(y_true, ord=2)**2)
    y_ = tf.norm(y_pred, ord=2)**2

    x_s = np.sum(x_n * y_n / N) / x_n.shape[0]

    Lmag = tf.norm(tf.math.log(tf.math.abs(y_true)/(tf.math.abs(y_pred)+e_)), ord=1)

    return Lsc + Lmag/10000

def my_loss_STOI(y_true, y_pred):
    N = 30
    # Take STFT
    x_spec = y_true
    y_spec = y_pred

    print("\n", x_spec.shape)

    # Ensure at least 30 frames for intermediate intelligibility
    if x_spec.shape[-1] < N:
        print('Not enough STFT frames to compute intermediate '
                      'intelligibility measure after removing silent '
                      'frames. Returning 1e-5. Please check you wav files')
        return 1e-5

    # Apply OB matrix to the spectrograms as in Eq. (1)
    x_tob = x_spec
    y_tob = y_spec

    # Take segments of x_tob, y_tob
    x_segments = np.array(
        [x_tob[:, m - N:m] for m in range(N, x_tob.shape[1] + 1)])
    y_segments = np.array(
        [y_tob[:, m - N:m] for m in range(N, x_tob.shape[1] + 1)])

    if extended:
        x_n = utils.row_col_normalize(x_segments)
        y_n = utils.row_col_normalize(y_segments)
        return np.sum(x_n * y_n / N) / x_n.shape[0]

    else:
        # Find normalization constants and normalize
        normalization_consts = (
            np.linalg.norm(x_segments, axis=2, keepdims=True) /
            (np.linalg.norm(y_segments, axis=2, keepdims=True) + utils.EPS))
        y_segments_normalized = y_segments * normalization_consts

        # Clip as described in [1]
        clip_value = 10 ** (-BETA / 20)
        y_primes = np.minimum(
            y_segments_normalized, x_segments * (1 + clip_value))

        # Subtract mean vectors
        y_primes = y_primes - np.mean(y_primes, axis=2, keepdims=True)
        x_segments = x_segments - np.mean(x_segments, axis=2, keepdims=True)

        # Divide by their norms
        y_primes /= (np.linalg.norm(y_primes, axis=2, keepdims=True) + utils.EPS)
        x_segments /= (np.linalg.norm(x_segments, axis=2, keepdims=True) + utils.EPS)
        # Find a matrix with entries summing to sum of correlations of vectors
        correlations_components = y_primes * x_segments

        # J, M as in [1], eq.6
        J = x_segments.shape[0]
        M = x_segments.shape[1]

        # Find the mean of all correlations
        d = np.sum(correlations_components) / (J * M)

        return d

def my_L_SI_SNR(y_true, y_pred):

        a_ = 1
        my_loss = tf.nn.l2_loss(y_true - y_pred)
        my_loss2 = tf.nn.l2_loss(y_true)
        l_si_snr= 10*tf.math.log(my_loss2/my_loss)/2.30258

        return l_si_snr

####### new
def my_snr_2(y_true, y_pred):

        my_loss = tf.nn.l2_loss(y_true - y_pred)
        my_loss2 = tf.nn.l2_loss(y_true)

        loss = 10*tf.math.log(my_loss/my_loss2)/2.30258

        return loss

def my_snr_(y_true, y_pred):

        my_loss = tf.nn.l2_loss(y_true - y_pred)
        my_loss2 = tf.nn.l2_loss(y_true**2)

        loss = 10*tf.math.log(my_loss2/my_loss)/2.30258

        return -loss

def my_Corr(y_true, y_pred):

        my_loss = tf.nn.l2_loss(y_true * y_pred)
        my_loss2 = tf.nn.l2_loss(y_true)
        my_loss3 = tf.nn.l2_loss(y_pred)

        loss = 10*tf.math.log(my_loss/(my_loss2 * my_loss3))/2.30258

        return loss

def my_get_SDR_MAE(y_true, y_pred):
    loss = tf.nn.l2_loss(y_true - y_pred)
    loss2 = tf.nn.l2_loss(y_true)
    loss_sdr = 10*tf.math.log(loss2/loss)/2.30258

    loss =  tf.nn.l2_loss(y_true - y_pred)
    loss_mae = 10*tf.math.log(loss)/2.30258

    return - 0.8 * loss_sdr - 0.2 * loss_mae

def my_get_PMSQ_loss(y_true, y_pred):

    mean = tf.math.reduce_mean(y_true)
    variance = tf.math.reduce_variance(y_true)

    loss_x = tf.nn.l2_loss(y_true)
    loss_x_ = tf.nn.l2_loss(y_pred)
    loss = (10*tf.math.log(loss_x/loss_x_)/2.30258)**2



    return loss / variance**2

def my_get_SDR_else(y_true, y_pred):

    mean = tf.math.reduce_mean(y_true, axis=1, keepdims=True)
    variance = tf.math.reduce_variance(y_true, axis=1, keepdims=True)

    mean_ = tf.math.reduce_mean(y_pred, axis=1, keepdims=True)
    variance_ = tf.math.reduce_variance(y_pred, axis=1, keepdims=True)


    loss_x = tf.nn.l2_loss(y_true)
    loss_x_ = tf.nn.l2_loss(y_true - y_pred)
    loss = 10*tf.math.log(loss_x/(loss_x_))/2.30258

    e = tf.math.abs(mean - mean_)
    e_ = tf.nn.l2_loss(variance)/tf.nn.l2_loss(variance - variance_)
    loss_e = 10*tf.math.log(e)/2.30258
    loss_e_ = 10*tf.math.log(e_)/2.30258


    return - loss - 0.01 * tf.math.log(loss_x_)

def my_MSE(y_true, y_pred):

    mean = tf.math.reduce_mean(y_true, axis=1, keepdims=True)
    variance = tf.math.reduce_variance(y_true, axis=1, keepdims=True)

    mean_ = tf.math.reduce_mean(y_pred, axis=1, keepdims=True)
    variance_ = tf.math.reduce_variance(y_pred, axis=1, keepdims=True)


    loss_x = tf.nn.l2_loss(y_true)
    loss_x_ = tf.nn.l2_loss(y_true - y_pred)
    loss = (loss_x - loss_x_)**2

    return loss/1000