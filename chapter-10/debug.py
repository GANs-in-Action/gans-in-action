import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
import foolbox
import pprint as pp
from PIL import Image
from keras.preprocessing import image as img
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from foolbox.criteria import Misclassification
import pandas as pd

def load_image(extension: str):
  img_path = f'{extension}.jpg'
  image = img.load_img(img_path, target_size=(224, 224))
  plt.imshow(image)
  x = img.img_to_array(image)
  return x


import keras
import numpy as np
from keras.applications.resnet50 import ResNet50

# instantiate model
keras.backend.set_learning_phase(0)
kmodel = ResNet50(weights='imagenet')
preprocessing = (np.array([104, 116, 123]), 1)
fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing)

max_vals = pd.read_csv('max_vals.csv')
# EOF 


fig = plt.figure(figsize=(20,20))
mu = 50
sigma = 10
sigma_list = list(max_vals.sigma)
mu_list = list(max_vals.mu)
sum_pred = []

def preprocess_resnet50(image: np.array):
    if image.shape==(224, 224, 3):
        # image = np.expand_dims(image, axis=0)
        pass
    elif image.shape!=(1,224,224,3):
        raise AssertionError("Unexpected Shape.")
    
    image = np.clip(image, 0, 255)
    image = preprocess_input(image)

    return image



def formatted_predictions(predictions: np.array):
    softmaxed = softmax(predictions)
    softmaxed = np.expand_dims(predictions, axis=0)
    return decode_predictions(softmaxed)


def make_subplot(x, y, z, new_row=False):
    rand_noise = np.random.normal(loc=mu, scale=sigma, size=(224,224, 3))
    to_predict = preprocess_resnet50(rand_noise)
    first_pred = fmodel.predictions(to_predict)
    import ipdb; ipdb.set_trace()
    label = np.argmax(first_pred)
    attack = foolbox.attacks.FGSM(fmodel, threshold=.5, criterion=Misclassification())
    adversarial = attack(rand_noise[:, :, ::-1], label)
    # adversarial = adversarial.image 
    adversarial = preprocess_resnet50(adversarial)
    noise_preds = kmodel.predict(adversarial)
    prediction, num = decode_predictions(noise_preds, top=20)[0][0][1:3]
    num = round(num * 100, 2)
    sum_pred.append(num)
    ax = fig.add_subplot(x,y,z)
    ax.annotate(prediction, xy=(0.1, 0.6), xycoords=ax.transAxes, fontsize=16, color='yellow')
    ax.annotate(f'{num}%' , xy=(0.1, 0.4), xycoords=ax.transAxes, fontsize=20, color='orange')
    if new_row:
        ax.annotate(f'$\mu$:{mu}, $\sigma$:{sigma}' , xy=(-.2, 0.8), xycoords=ax.transAxes,
                    rotation=90, fontsize=16, color='black')
    to_plot = (adversarial[0] + 127.5) / 127.5
    ax.imshow(to_plot) 
    ax.axis('off')

    
for i in range(1,101):
    if (i-1) % 10==0:
        mu = mu_list.pop(0)
        sigma = sigma_list.pop(0)
        make_subplot(10,10, i, new_row=True) 
    else:
        make_subplot(10,10, i)

plt.show()