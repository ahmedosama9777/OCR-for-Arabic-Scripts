# OCR-for-Arabic-Scripts
Optical character recognition of arabic script images and output scripts as text.

# Project Pipeline
1. Take input image along with its text file to train the model
2. Segment image to lines
3. Segment lines to words
4. Segment words to characters
5. Compare segmented characters with its text files
6. Take the valid data as training dataset
7. Feature extraction of the training dataset
8. Building the neural network model
9. Classification of the dataset through the model
10. Cross validation of the data (80% training – 20% validation)
11. Input any test image to be segmented and write its characters into text file

# Pre-processing module
Input image is converted to a binary image then dilated to get lines and words
per image. It was also found that this corrects the skew of the input images.
Then by now, we have words to be segmented to characters and continue the
flow of the pipeline.

# Character Segmentation module
  - For every word image, a baseline index and maximum transition index are
calculated. Baseline is the line with the most pixel density along with the whole
line where most letters are on a horizontal segment with a constant width.
Maximum transition is the horizontal line with maximum number of vertical
transitions above the baseline. This is done by searching for the horizontal line
with the maximum number of pixel value change (transition from 0 to 1 or from
1 to 0).
  - All available cut points are then determined but there may exist invalid ones
among them.
  - Valid cut points are determined to be able to segment the word to characters.
  - Papers used: An efficient, font independent word and character segmentation algorithm for
printed Arabic text (KING SAUDI UNIVERSITY)

# Feature extraction/ selection module
Unsuccessful trials:
  - Image flattening
  - Vertical and horizontal projection of histogram
  - SIFT Descriptors
  
Successful (used):
  - Hue moments
  - Ratio between white and black pixels in the 4 quarters of the letter image
  - Center of mass
  - Connected components
Papers used:
  - Using SIFT Descriptors for OCR of Printed Arabic (TEL AVIV UNIVERSITY)
  - Printed Arabic Optical Character Recognition Using Support Vector Machine
    (International conference on Mathematics and Information Technology, Adrar,
    Algeria)

# Model selection and training modules
A fully connected neural network model is used with two hidden layers. The
input layer receives feature vector of size 20. First hidden layer is 40 neurons
with “Relu” activation function and the second one consists of 50 neurons along
with “Relu” function also. The output layer has 29 neurons for the 29 class then
determine the letter’s class using “Soft max” activation function.
And for the training modules, SGD (stochastical gradient descent) is used as a
training optimizer. Keras is used along with tensor flow as a backend.

# Enhancements and future work (methods planned to try)
  - Improve character segmentation accuracy
  - Add more features as hole detection and determining location of dots in letters
  - Tune the hyper parameters of the neural network as number of hidden layers
and number of neurons in each layer
  - Classify on handwritten Arabic OCR
