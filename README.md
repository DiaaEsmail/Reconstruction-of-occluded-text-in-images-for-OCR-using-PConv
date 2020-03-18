# Reconstruction of Occluded Text in Images for Optical Character Recognition


Text occlusion is among the most intractable obstacles for the task of Optical
Character Recognition (OCR) in Computer Vision. A typical example
is when some characters of a text in an image are invisible due to occlusion
them by arbitrary geometrical shapes. This causes distorting the text and
missing some characters in the word because the holes arise in the image. It
requires finding a method to reconstruct the image by filling in holes with
plausible imagery before recognizing the whole text in an image. I introduce
an approach to reconstruct an image by using Partial Convolution,
which comprises a mask and renormalized convolution operation followed
by a mask-update step to automatically generate an updated mask for the
next layer as part of the forward pass. For the recognition task, I propose
a no-recurrence sequence-to-sequence model that depends only on attention
mechanism and dispenses completely with recurrences and convolutions. The
model consists of an encoder and decoder, and uses stacked self-attention
modules. The Experiments have shown that both proposed models can produce
reconstructed images and improve the effectiveness of text recognition
in these images.


#Dependencies

    Python 3.6
    Keras 2.2.4
    Tensorflow 1.12

