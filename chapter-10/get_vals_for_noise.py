import numpy as np
from keras.preprocessing import image as img
from keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50
import foolbox
import pandas as pd
import keras

#TODO: add comments to this file

keras.backend.set_learning_phase(0)
kmodel = ResNet50(weights='imagenet')

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


storage_df.to_csv('initialization_vals_for_noise.csv')