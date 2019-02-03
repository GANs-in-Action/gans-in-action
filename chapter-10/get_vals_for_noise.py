import numpy as np
from keras.applications.resnet50 import ResNet50
from foolbox.criteria import Misclassification, ConfidentMisclassification
from keras.preprocessing import image as img
from keras.applications.resnet50 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import foolbox
import pprint as pp
Import keras
%matplotlib inline


mu_list = range(50, 200, 10)
sigma_list = range(10, 100, 2)
storage_df = pd.DataFrame()

for mu, sigma in zip(mu_list, sigma_list):
    rand_noise = np.random.normal(loc=mu, scale=sigma, size=(224,224, 3))
    rand_noise = np.clip(rand_noise, 0, 255.)
    noise_preds = kmodel.predict(np.expand_dims(rand_noise, axis=0))
    predictions = decode_predictions(noise_preds, top=20)[0]
    new_df = pd.DataFrame(predictions, columns=['id','class','prediction'])
    new_df['sigma'] = sigma
    new_df['mu'] = mu
    storage_df = pd.concat([new_df, storage_df])


max_vals.to_csv('initialization_vals_for_noise.csv')

sigma_list = list(max_vals.sigma)
mu_list = list(max_vals.mu)