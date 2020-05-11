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

