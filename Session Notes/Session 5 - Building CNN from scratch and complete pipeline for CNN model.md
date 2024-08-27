# Session 5 - Building CNN from scratch and complete pipeline for CNN model

In today's session we shall cover all of these things in detail -
1. Visualisation of Convolutions demo in detail. - titos:/bolocut.eltute lo/to-explaineu
2. How to enable GPU resources on Google Colab.
3. Learn to create basic convolution architecutre with all the details.
4. Calculate the paramaters of the architecture.
5. Training and testing (inference)
6. Accuracy Metrics.
7. Advance topics like early stopping, batch normalisation, dropouts, checkpoints, callbacks.

- CNN visualiser - https://poloclub.github.io/cnn-explainer/
- Kaggle Dogs vs Cats - https://www.kaggle.com/datasets/salader/dogs-vs-cats/code
- Tensorflow documentation - https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
- Leaf disease prediction - https://www.kaggle.com/code/hamedetezadi/leaf-disease-prediction-using-cnn

That sounds like a comprehensive session on convolutional neural networks (CNNs) and related advanced topics. Let's break down each point and go through them in detail:

### 1. Visualization of Convolutions Demo

Understanding how convolutions work is crucial for building CNNs. Convolutions are operations applied to input data to extract features, and visualizing them can provide insights into how neural networks learn patterns.

- **Demo Setup**: For the visualization demo, we can use tools like `matplotlib` and `seaborn` to visualize convolutional operations on sample images. We'll look at how different filters (kernels) affect the image.
  
- **Visual Elements**:
  - **Original Image**: Display the original image before any convolution is applied.
  - **Filters**: Show different convolution filters such as edge detection, sharpening, and blurring.
  - **Feature Maps**: Visualize the feature maps generated after applying the convolution filters to the original image. This can help understand what features are being extracted at different layers.

- **Interactive Demos**: Using libraries like `ipywidgets` in Jupyter Notebooks or Google Colab, we can create interactive sliders to adjust filter values and observe changes in real-time.

### 2. Enabling GPU Resources on Google Colab

Using a GPU can significantly speed up the training of CNNs due to their parallel processing capabilities. 

- **Steps to Enable GPU**:
  1. Open Google Colab and create a new notebook.
  2. Go to `Runtime` > `Change runtime type`.
  3. Under `Hardware accelerator`, select `GPU`.
  4. Click `Save`.

- **Verify GPU Availability**: Run the following code in a cell to verify that the GPU is enabled:
  ```python
  import tensorflow as tf
  print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
  ```

### 3. Creating a Basic Convolutional Architecture

Designing a basic convolutional neural network involves stacking layers of convolutions, pooling, and fully connected layers.

- **Basic Architecture Example**:
  - **Input Layer**: Takes input images of a specific size.
  - **Convolutional Layer**: Applies multiple filters to extract features.
  - **Pooling Layer**: Reduces the spatial dimensions of the feature maps.
  - **Flatten Layer**: Converts the 2D matrix into a vector.
  - **Fully Connected Layer**: Connects every neuron from the previous layer to every neuron in the next layer.
  - **Output Layer**: Provides the final output, typically using a softmax activation function for classification tasks.

- **Example Code**:
  ```python
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

  model = Sequential([
      Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
      MaxPooling2D(pool_size=(2, 2)),
      Flatten(),
      Dense(128, activation='relu'),
      Dense(10, activation='softmax')
  ])
  ```

### 4. Calculating the Parameters of the Architecture

Understanding how to calculate the number of parameters in each layer is essential for model optimization.

- **Convolutional Layers**: The number of parameters is calculated as:
  \[
  \text{Parameters} = (K \times K \times C_{\text{in}} + 1) \times C_{\text{out}}
  \]
  where \(K\) is the kernel size, \(C_{\text{in}}\) is the number of input channels, and \(C_{\text{out}}\) is the number of output channels (filters).

- **Example Calculation**:
  For a `Conv2D` layer with 32 filters, a kernel size of (3, 3), and 3 input channels:
  \[
  (3 \times 3 \times 3 + 1) \times 32 = 896
  \]

### 5. Training and Testing (Inference)

Training the model involves feeding the input data through the network and updating the weights based on the loss function.

- **Training**: Use the `model.fit()` method to train the model on the dataset. Key parameters include the batch size, number of epochs, and optimizer.
  
- **Testing**: After training, use `model.evaluate()` to test the model's performance on unseen data.

- **Example Code**:
  ```python
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  model.fit(X_train, y_train, epochs=10, batch_size=32)
  model.evaluate(X_test, y_test)
  ```

### 6. Accuracy Metrics

Accuracy metrics help evaluate the model's performance.

- **Common Metrics**:
  - **Accuracy**: The ratio of correctly predicted instances to the total instances.
  - **Precision, Recall, F1-Score**: Useful for evaluating models on imbalanced datasets.

- **Confusion Matrix**: Provides insights into the number of true positives, true negatives, false positives, and false negatives.

### 7. Advanced Topics

These advanced techniques help improve model performance and prevent overfitting.

- **Early Stopping**: Stops training when the model's performance on a validation dataset stops improving.
  ```python
  from tensorflow.keras.callbacks import EarlyStopping
  early_stopping = EarlyStopping(monitor='val_loss', patience=3)
  ```

- **Batch Normalization**: Normalizes the output of a previous activation layer by subtracting the batch mean and dividing by the batch standard deviation.
  ```python
  from tensorflow.keras.layers import BatchNormalization
  model.add(BatchNormalization())
  ```

- **Dropouts**: Randomly drops neurons during training to prevent overfitting.
  ```python
  from tensorflow.keras.layers import Dropout
  model.add(Dropout(0.5))
  ```

- **Checkpoints**: Saves the model during training for future use.
  ```python
  from tensorflow.keras.callbacks import ModelCheckpoint
  checkpoint = ModelCheckpoint('model.h5', save_best_only=True)
  ```

- **Callbacks**: Functions that are executed at certain points during training, such as after each epoch.
  ```python
  callbacks = [early_stopping, checkpoint]
  model.fit(X_train, y_train, epochs=10, callbacks=callbacks)
  ```

By covering these points, you'll have a comprehensive understanding of CNNs, how to implement them, and advanced techniques for improving their performance.



