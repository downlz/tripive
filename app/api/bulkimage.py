from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import io
import os
from flask_restplus import Resource
import requests
import json
# from io import BytesIO

from .security import require_auth
from . import api_rest

from keras.preprocessing.image import image
from keras.applications.inception_v3 import preprocess_input,decode_predictions

# global model = tf.keras.models.load_model('./inception.h5')

# model = tf.keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

class SecureResource(Resource):
    """ Calls require_auth decorator on all requests """
    method_decorators = [require_auth]

def pred_model(imgsrc):
  model = tf.keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
  img = image.load_img(imgsrc, target_size=(299, 299))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)

  preds = model.predict(x)
  # print(preds)
  print('Predicted:', decode_predictions(preds, top=5)[0])

# Reference https://firebase.google.com/docs/firestore/query-data/get-data
@api_rest.route('/tensor/<string:resource_id>')
class ImageAnalysisTensor(Resource):
    
    def get(self, resource_id):
        arr = ['taj.jpg','state.jpg','yos.jpg','waipio.jpg','owens.jpg','sonoma.jpg']
        # model = tf.keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
        # model = tf.keras.models.load_model('./inception.h5')
        for xi in arr:
            path = os.path.abspath('files/' + xi)
            img = image.load_img(path, target_size=(299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            preds = model.predict(x)
            # print(preds)
            print('Predicted:', decode_predictions(preds, top=5)[0])

        return {'images':'data'}
#Generate model
    def post(self, resource_id):
        # json_payload = request.json
        # doc_ref = db.collection('intialimageref').document(resource_id)         #Creating document for each labels
        # doc_ref.set(json_payload)
        model = tf.keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
        model.save('inception.h5')
        return {'message': 'success'}, 201        