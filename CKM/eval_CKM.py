import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score,root_mean_squared_error,adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

model = tf.keras.models.load_model('uav_user_channel_model.h5')
X_test = np.load('X_test.npy')
env_test = np.load('env_test.npy')
y_test = np.load('y_test.npy')

test_loss = model.evaluate([X_test, env_test], y_test)
print(f'Test loss: {test_loss}')

y_pred = model.predict(X_test, env_test)
plt.plot(y_pred, 'r')
plt.plot(y_test, 'g')
plt.show()


plt.show()