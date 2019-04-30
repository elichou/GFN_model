# GFN_model
## Gated Field Network Autoencoder model for visuomotor affordances

GFN_encoder : 
1. input_type =  ['image', 'image'] ,  ['image', 'posture'], ['posture', 'posture']
2. input_shape = (2, img_size, img_size,) 
3. latent_dim = 32
4. factor_type = ['dense']

The encoder extracts information from either :
1. A pair of successive images, and therefore represents the transformation from one frame to another. 
2. A pair of successive robot arm posture, idem.
3. One image and one posture, and therefore represent common static features in a latent representation space.

The decoder is used to train the encoder to rebuild either:
1. The next step input image knowing the previous.
2. The next step input posture knowing the previous.
3. The input image knowing the input posture.