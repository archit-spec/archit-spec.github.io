---
title: ml notes
draft: false
tags:
  - ML
---


$$

\frac{\partial f}{\partial t}

$$



$$\partial_t u + \nabla_x \cdot (b u) = 0\qquad\text{ s.t. } u(\cdot,t) = x_0$$
multi-headed attention is used everywhere (also the encoder). The idea behind multi-headed attention is essentially the same as how you use multiple channels in a CNN to increase fidelity.
I.e. MHSA simply executes `n_heads` self-attention steps in parallel (by splitting the input into `n_heads` subsets), and concatenates the results.


$$ \sin{2x}$$


Pseudocode is:
```python
def MHSA(query, key, value):
  queries = torch.chunk(query,n_heads, -1)
  keys = torch.chunk(key,n_heads, -1)
  values = torch.chunk(values,n_heads, -1)
  result = []
  for (k,q,v) in zip(keys,queries,values):
   result.append(single_headed_attention(q,k,v))
  return torch.cat(results,-1)
```
In practice this parallelization can be done by simply reshaping and contracting over the queries, keys and values.

We don't use padding because of the linear layers: the linear layers do not care about the sequence length since they are broadcasted across the entire sequence:
If you have an input of size `(B, L, D)` the linear layers always act on the last dimension, i.e.
`linear: (B, L, D) → (B, L, E)`
Even if you triple the sequence length that will not change the number of neurons, since you broadcast along the sequence.
The only way that two different tokens interact is via the attention mechanism which doesn't have any weights (it's just `softmax(Q@K.T / sqrt(d_k))@V`, so no weights. The linear layers that give you Q, K, V, are again broadcast along the entire sequence).


```python
class TransformerBlock(layers.Layer):
    def __init__(self, emb_dim, n_heads, mlp_dim, 
                 rate=0.1, initializer='glorot_uniform', eps=1e-6, activation='gelu'):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(emb_dim, n_heads, initializer=initializer)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation=activation, kernel_initializer=initializer), 
            layers.Dense(emb_dim, kernel_initializer=initializer),
            layers.Dropout(rate)
        ])
        self.ln1 = layers.LayerNormalization(epsilon=eps)
        self.ln2 = layers.LayerNormalization(epsilon=eps)

    def call(self, inputs, mask=None):
        x = self.ln1(inputs)
        x = inputs + self.attn(x, x, x, mask) 
        x = x + self.mlp(self.ln2(x))
        return x
```



%% Cell type:code id: tags:
``` python
pip install pystac_client odc-stac
```
%% Output
    Collecting pystac_client
``` python
from pystac_client import Client
from odc.stac import load
client = Client.open("https://earth-search.aws.element84.com/v1")
collection = "sentinel-2-l2a"
tas_bbox = [146.5, -43.6, 146.7, -43.4]
search = client.search(collections=[collection], bbox=tas_bbox, datetime="2020-12")
data = load(search.items(), bbox=tas_bbox, groupby="solar_day", chunks={})
data[["red", "green", "blue"]].isel(time=2).to_array().plot.imshow(robust=True)
```
%% Output
    <matplotlib.image.AxesImage at 0x7fdf85af2950>
%% Cell type:code id: tags:
``` python
import requests
# Replace these variables with your actual values
instance_id = '1bcfffc9-2232-46a0-807a-b450dc32e35d'
api_key = 'your_api_key'
layer_id = 'your_layer_id'
bounding_box = 'min_lon,min_lat,max_lon,max_lat'
width = 512
height = 512
time = '2023-01-01/2023-12-31'  # Specify the time range
# Construct the WMS request URL
wms_url = f'https://services.sentinel-hub.com/ogc/wms/{instance_id}?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&FORMAT=image/jpeg&TRANSPARENT=false&LAYERS={layer_id}&STYLES=&CRS=EPSG:4326&BBOX={bounding_box}&WIDTH={width}&HEIGHT={height}&TIME={time}&apikey={api_key}'
# Make the API request
response = requests.get(wms_url)
# Check if the request was successful (HTTP status code 200)
if response.status_code == 200:
    # Save the retrieved image
    with open('output_image.jpg', 'wb') as f:
        f.write(response.content)
    print('Image downloaded successfully.')
else:
    print(f'Error: {response.status_code}\n{response.text}')
```
%% Output
    Error: 400
    <?xml version='1.0' encoding="UTF-8"?>
    <ServiceExceptionReport version="1.3.0"
    	xmlns="http://www.opengis.net/ogc"
    	xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    	xsi:schemaLocation="http://www.opengis.net/ogc http://schemas.opengis.net/wms/1.3.0/exceptions_1_3_0.xsd">
    	<ServiceException>
    		<![CDATA[ Illegal BBOX format: min_lon,min_lat,max_lon,max_lat ]]>
    	</ServiceException>
    </ServiceExceptionReport>
%% Cell type:code id: tags:
``` python
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
```




