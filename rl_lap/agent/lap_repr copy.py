class EmptyClass(object):
  pass

def l2_dist(x):
  x = tf.reshape(x, [x.shape.as_list()[0], -1])
  return tf.reduce_sum(tf.square(x), axis=-1)

def pos_loss(x1, x2):
  '''
  x1, x2: batch_size * repr_dim.
  Return: a loss that positively correlates with the distances between x1 and x2.
  '''
  return tf.reduce_mean(l2_dist(x1 - x2))


def neg_loss(x, c=1.0):
  '''
  Return: a loss for negative sampling.
  '''
  n = x.shape.as_list()[0]
  dot_prods = tf.matmul(x, tf.transpose(x))
  loss = tf.reduce_sum(tf.square(dot_prods / n - c * tf.eye(n))) # should be 1 or n^2
  return loss

def neg_loss_unbiased(x1, x2, c=1.0):
  '''
  x1, x2: n*d
  return E[(x'y)^2] - 2cE[x'x] + c^2 * d
  '''
  d = x1.shape.as_list()[1]
  part1 = tf.reduce_mean(tf.square(tf.matmul(x1, tf.transpose(x2))))
  part2 = - c * (tf.reduce_mean(tf.reduce_sum(tf.square(x1), axis=-1))
                 + tf.reduce_mean(tf.reduce_sum(tf.square(x2), axis=-1)))
  part3 = c * c * d
  loss = part1 + part2 + part3
  return loss



def neg_loss_sigmoid(x, c=1.0):
  '''
  c \in [0, 1]
  '''
  n = x.shape.as_list()[0]
  dot_prods = tf.matmul(x, tf.transpose(x))
  loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
      logits=dot_prods, labels=c*tf.eye(n))) / n # should be 1 or n^2
  return loss



class W2VLearner(object):

  def __init__(self, obs_shape, pos_batch_size, neg_batch_size,
               repr_module=None, neg_loss_c=1.0, w_neg=1.0,
               learning_rate=1e-4, optimizer=tf.train.AdamOptimizer,
               scope_name='learner', obs_to_repr=(lambda x: x),
               sample_discount=0.9, sample_max_range=1,
              ):
    self._obs_shape = obs_shape
    self._pos_batch_size = pos_batch_size
    self._neg_batch_size = neg_batch_size
    self._repr_module = repr_module
    self._neg_loss_c = neg_loss_c
    self._learning_rate = learning_rate
    self._scope_name = scope_name
    self._obs_to_repr = obs_to_repr
    self._global_step = 0
    self._optimizer=optimizer
    self._w_neg = w_neg
    self._sample_discount = sample_discount
    self._sample_max_range = sample_max_range

    #self._num_actions = self._action_spec.n
    self._build_learner()
    self._session = None

  def load_session(self, sess):
    self._session = sess

  def release_session(self):
    self._session = None

  def _build_learner(self):

    # set up tf graph for training
    self._train_holders = self._build_train_holders()
    # instantiate networks
    with tf.variable_scope(self._scope_name):
      with tf.variable_scope('repr_net'):
        self._repr_net = self._repr_module()

    repr_s1, repr_s2, repr_s_neg = map(
        self._repr_net,
        [self._train_holders.s1, self._train_holders.s2,
         self._train_holders.s_neg]
    )
    self._loss_pos = pos_loss(repr_s1, repr_s2)
    self._loss_neg = neg_loss(repr_s_neg, c=self._neg_loss_c)

    self._loss = self._loss_pos + self._w_neg * self._loss_neg

    # optimization
    opt = self._optimizer(learning_rate=self._learning_rate)
    self._train_op = opt.minimize(self._loss)
    # saver
    self._var_list = self._repr_net.get_variables()
    self._model_saver = tf.train.Saver(var_list=self._var_list)

  def _build_train_holders(self):
    holders = EmptyClass()
    holders.s1 = tf.placeholder(tf.float32, [self._pos_batch_size]+self._obs_shape)
    holders.s2 = tf.placeholder(tf.float32, [self._pos_batch_size]+self._obs_shape)
    holders.s_neg = tf.placeholder(tf.float32, [self._neg_batch_size]+self._obs_shape)
    #holders.a = tf.placeholder(tf.int32, [self._batch_size])
    #holders.r = tf.placeholder(tf.float32, [self._batch_size])
    #holders.dsc = tf.placeholder(tf.float32, [self._batch_size])  # discount
    return holders

  def _get_obs_repr(self, obs):
    """overload for specific use"""
    return self._obs_to_repr(obs)

  @property
  def obs_shape(self):
    return self._obs_shape

  @property
  def obs_to_repr(self):
    return self._obs_to_repr

  @property
  def repr_net(self):
    return self._repr_net

  @property
  def repr_module(self):
    return self._repr_module


  def _get_obs_batch(self, time_steps):
    # each time step is a pair of (action, time_step)
    obs_batch = [self._get_obs_repr(ts[1].observation) for ts in time_steps]
    return np.stack(obs_batch, axis=0)


  #def _get_action_batch(self, time_steps):
  #  action_batch = [ts[0] for ts in time_steps]
  #  return np.stack(action_batch, axis=0)


  def _get_train_batch(self, replay_memory):
    if self._sample_max_range > 1:
      ts1, ts2 = replay_memory.sample_positive(
          batch_size=self._pos_batch_size,
          discount=self._sample_discount,
          max_range=self._sample_max_range,
      )
    else:
      ts1, ts2 = replay_memory.sample_transitions(self._pos_batch_size)
    ts_neg = replay_memory.sample_steps(
        batch_size=self._neg_batch_size,
    )
    #a = self._get_action_batch(ts1)
    s1, s2, s_neg = map(self._get_obs_batch, [ts1, ts2, ts_neg])
    # compute reward and discount
    #r, dsc = self._get_r_dsc_batch(ts2)
    holders = self._train_holders
    feed_dict = {
        holders.s1: s1,
        holders.s2: s2,
        holders.s_neg: s_neg,
        }
    return feed_dict

  def _get_batch_dict(self, feed_dict):
    vals_dict = {}
    for kw, holder in vars(self._train_holders).items():
      vals_dict[kw] = feed_dict[holder]
    return vals_dict

  def train_step(self, sess, replay_memory, step, print_freq):
    #sess = self._session
    feed_dict = self._get_train_batch(replay_memory)
    loss, loss_pos, loss_neg, _ = sess.run(
        [self._loss, self._loss_pos, self._loss_neg, self._train_op],
        feed_dict=feed_dict)
    # print info
    if step == 0 or (step + 1) % print_freq == 0:
      logging.info(('Step {}:  loss {:.4g}, loss_pos {:.4g}, loss_neg {:.4g}.'
                   ).format(step+1, loss, loss_pos, loss_neg))
      #print(('Step {}:  loss {:.4g}, loss_pos {:.4g}, loss_neg {:.4g}.'
      #      ).format(step+1, loss, loss_pos, loss_neg))
    batch = self._get_batch_dict(feed_dict)
    self._global_step += 1
    return batch

  def save_model(self, sess, path):
      self._model_saver.save(sess, path)

  def load_model(self, sess, path):
      self._model_saver.restore(sess, path)


class W2VUnbiasedLearner(W2VLearner):

  def _build_learner(self):

    # set up tf graph for training
    self._train_holders = self._build_train_holders()
    # instantiate networks
    with tf.variable_scope(self._scope_name):
      with tf.variable_scope('repr_net'):
        self._repr_net = self._repr_module()

    repr_s1, repr_s2, repr_s1_neg, repr_s2_neg = map(
        self._repr_net,
        [self._train_holders.s1, self._train_holders.s2,
         self._train_holders.s1_neg, self._train_holders.s2_neg]
    )
    self._loss_pos = pos_loss(repr_s1, repr_s2)
    self._loss_neg = neg_loss_unbiased(repr_s1_neg, repr_s2_neg, c=self._neg_loss_c)

    self._loss = self._loss_pos + self._w_neg * self._loss_neg

    # optimization
    opt = self._optimizer(learning_rate=self._learning_rate)
    self._train_op = opt.minimize(self._loss)
    # saver
    self._var_list = self._repr_net.get_variables()
    self._model_saver = tf.train.Saver(var_list=self._var_list)

  def _build_train_holders(self):
    holders = EmptyClass()
    holders.s1 = tf.placeholder(tf.float32, [self._pos_batch_size]+self._obs_shape)
    holders.s2 = tf.placeholder(tf.float32, [self._pos_batch_size]+self._obs_shape)
    holders.s1_neg = tf.placeholder(tf.float32, [self._neg_batch_size]+self._obs_shape)
    holders.s2_neg = tf.placeholder(tf.float32, [self._neg_batch_size]+self._obs_shape)
    return holders

  def _get_train_batch(self, replay_memory):
    ts1, ts2 = replay_memory.sample_transitions(
        batch_size=self._pos_batch_size,
    )
    ts1_neg = replay_memory.sample_steps(
        batch_size=self._neg_batch_size,
    )
    ts2_neg = replay_memory.sample_steps(
        batch_size=self._neg_batch_size,
    )

    #a = self._get_action_batch(ts1)
    s1, s2, s1_neg, s2_neg = map(
        self._get_obs_batch, [ts1, ts2, ts1_neg, ts2_neg])
    # compute reward and discount
    #r, dsc = self._get_r_dsc_batch(ts2)
    holders = self._train_holders
    feed_dict = {
        holders.s1: s1,
        holders.s2: s2,
        holders.s1_neg: s1_neg,
        holders.s2_neg: s2_neg,
        }
    return feed_dict


class SpecNetLearner(W2VLearner):

  def __init__(self, obs_shape, batch_size, repr_module=None,
               learning_rate=1e-4, optimizer=tf.train.AdamOptimizer,
               scope_name='learner', obs_to_repr=(lambda x: x),
              ):
    self._obs_shape = obs_shape
    self._batch_size = batch_size
    self._repr_module = repr_module
    self._learning_rate = learning_rate
    self._scope_name = scope_name
    self._obs_to_repr = obs_to_repr
    self._global_step = 0
    self._optimizer=optimizer

    #self._num_actions = self._action_spec.n
    self._build_learner()
    self._session = None

  def _build_learner(self):

    # set up tf graph for training
    self._train_holders = self._build_train_holders()
    # instantiate networks
    with tf.variable_scope(self._scope_name):
      with tf.variable_scope('repr_net'):
        self._repr_net = self._repr_module()

    repr_s1, repr_s2 = map(
        self._repr_net,
        [self._train_holders.s1, self._train_holders.s2]
    )
    repr_s1 = self._final_layer(repr_s1)
    repr_s2 = self._final_layer(repr_s2)
    self._loss = pos_loss(repr_s1, repr_s2)

    # optimization
    opt = self._optimizer(learning_rate=self._learning_rate)
    self._train_op = opt.minimize(self._loss)
    # saver
    self._var_list = self._repr_net.get_variables()
    self._model_saver = tf.train.Saver(var_list=self._var_list)


  def _final_layer_old(self, x):
    n = x.shape.as_list()[0]
    d = x.shape.as_list()[1]
    eps = 0  # 1e-10
    h = tf.cholesky(tf.matmul(tf.transpose(x), x) + tf.eye(d) * eps)
    h = tf.transpose(tf.matrix_inverse(h)) * tf.sqrt(float(n))
    return tf.matmul(x, h)

  def _final_layer(self, x):
    n = x.shape.as_list()[0]
    #d = x.shape.as_list()[1]
    #eps = 0  # 1e-10
    #h = tf.cholesky(tf.matmul(tf.transpose(x), x) + tf.eye(d) * eps)
    #h = tf.transpose(tf.matrix_inverse(h)) * tf.sqrt(float(n))
    with tf.device('/cpu:0'):
      h = tf.svd(x)[1] * tf.sqrt(float(n))
    return h


  def _build_train_holders(self):
    holders = EmptyClass()
    holders.s1 = tf.placeholder(tf.float32, [self._batch_size]+self._obs_shape)
    holders.s2 = tf.placeholder(tf.float32, [self._batch_size]+self._obs_shape)
    return holders

  def _get_train_batch(self, replay_memory):
    ts1, ts2 = replay_memory.sample_transitions(
        batch_size=self._batch_size,
    )
    #a = self._get_action_batch(ts1)
    s1, s2 = map(self._get_obs_batch, [ts1, ts2])
    # compute reward and discount
    #r, dsc = self._get_r_dsc_batch(ts2)
    holders = self._train_holders
    feed_dict = {
        holders.s1: s1,
        holders.s2: s2,
        }
    return feed_dict


  def train_step(self, sess, replay_memory, step, print_freq):
    #sess = self._session
    feed_dict = self._get_train_batch(replay_memory)
    loss, _ = sess.run(
        [self._loss, self._train_op],
        feed_dict=feed_dict)
    # print info
    if step == 0 or (step + 1) % print_freq == 0:
      logging.info(('Step {}:  loss {:.4g}.'
                   ).format(step+1, loss))
      print(('Step {}:  loss {:.4g}.'
            ).format(step+1, loss))
    batch = self._get_batch_dict(feed_dict)
    self._global_step += 1
    return batch

