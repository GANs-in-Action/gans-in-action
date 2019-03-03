'''
This file serves as a way to ensure that we start from a meaningful point for ResNet50 attack.

If you are not convinced and think this may be making the problem too trivial—though consider
why as an attacker you would not use that—checkout the InceptionV3 example.
There we make use of no pre-stored values for attacks.

In principle, the ResNet50 attack should replicate even without this initialization, but it
is then not guaranteed
'''
# We get the standard Imports
import numpy as np
from keras.preprocessing import image as img
from keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50
import foolbox
import pandas as pd
import keras

# Initialize our ResNet50 pre-trained model and our DataFrame to store the objects into 
keras.backend.set_learning_phase(0)
kmodel = ResNet50(weights='imagenet')

mu_list = range(50, 200, 10)
sigma_list = range(10, 100, 2)
storage_df = pd.DataFrame()

# In a for loop generate a bunch of mean and variances values
for mu, sigma in zip(mu_list, sigma_list):
    # sample for this particular mean and var from normal at that location
    rand_noise = np.random.normal(loc=mu, scale=sigma, size=(224,224, 3))
    # preprocess
    rand_noise = np.clip(preprocess_input(rand_noise), 0, 255.)
    # get raw predictions
    noise_preds = kmodel.predict(np.expand_dims(rand_noise, axis=0))
    # get human readable
    predictions = decode_predictions(noise_preds, top=20)[0]
    # store these predictions in a dataframe format 
    new_df = pd.DataFrame(predictions, columns=['id','class','prediction'])
    new_df['sigma'] = sigma
    new_df['mu'] = mu
    # add this data point to the data frame
    storage_df = pd.concat([new_df, storage_df])


storage_df.to_csv('initialization_vals_for_noise.csv')