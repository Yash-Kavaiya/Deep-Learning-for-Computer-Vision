# Session 3 - Padding and Strides during Convolution of Images

### Padding in Convolutional Neural Networks (CNNs)

Padding is a technique used in Convolutional Neural Networks (CNNs) to control the spatial dimensions (height and width) of the output feature maps after applying convolution operations. Padding is crucial because it helps to preserve the spatial resolution of the input and ensures that the information at the edges of the image is not lost during the convolution process.

### Types of Padding

1. **Valid Padding (No Padding):**
   - In valid padding, no extra padding is added to the input image. This results in a reduction in the spatial dimensions of the output after each convolution operation.
   - If the kernel size is \(k \times k\) and the stride is 1, the output dimensions are reduced by \(k-1\) in both height and width.

   **Example:**
   - Input size: \(6 \times 6\)
   - Kernel size: \(3 \times 3\)
   - Stride: 1
   - Padding: 0 (valid padding)

   **Mathematics:**
   \[
   \text{Output height} = \left(\frac{\text{Input height} - \text{Kernel height}}{\text{Stride}} + 1\right)
   \]
   \[
   \text{Output width} = \left(\frac{\text{Input width} - \text{Kernel width}}{\text{Stride}} + 1\right)
   \]

   Substituting the values:
   \[
   \text{Output height} = \frac{6 - 3}{1} + 1 = 4
   \]
   \[
   \text{Output width} = \frac{6 - 3}{1} + 1 = 4
   \]
   - Output size: \(4 \times 4\)

2. **Same Padding:**
   - In same padding, padding is added to the input image so that the output size remains the same as the input size after the convolution operation.
   - The padding size is calculated based on the kernel size to ensure the output size matches the input size.

   **Example:**
   - Input size: \(6 \times 6\)
   - Kernel size: \(3 \times 3\)
   - Stride: 1
   - Padding: 1 (same padding)

   **Mathematics:**
   \[
   \text{Output height} = \left(\frac{\text{Input height} - \text{Kernel height} + 2 \times \text{Padding}}{\text{Stride}} + 1\right)
   \]
   \[
   \text{Output width} = \left(\frac{\text{Input width} - \text{Kernel width} + 2 \times \text{Padding}}{\text{Stride}} + 1\right)
   \]

   Substituting the values:
   \[
   \text{Output height} = \frac{6 - 3 + 2 \times 1}{1} + 1 = 6
   \]
   \[
   \text{Output width} = \frac{6 - 3 + 2 \times 1}{1} + 1 = 6
   \]
   - Output size: \(6 \times 6\)

3. **Full Padding:**
   - Full padding adds enough padding such that the output size is larger than the input size. It effectively pads the input to ensure that every pixel in the input image is covered by the kernel.

   **Example:**
   - Input size: \(6 \times 6\)
   - Kernel size: \(3 \times 3\)
   - Stride: 1
   - Padding: 2 (full padding)

   **Mathematics:**
   \[
   \text{Output height} = \frac{6 - 3 + 2 \times 2}{1} + 1 = 8
   \]
   \[
   \text{Output width} = \frac{6 - 3 + 2 \times 2}{1} + 1 = 8
   \]
   - Output size: \(8 \times 8\)

### Why Padding is Important

1. **Preserving Spatial Dimensions:** Padding ensures that the convolution operation does not shrink the spatial dimensions of the feature maps, which is particularly important in deep networks where multiple layers of convolutions could drastically reduce the feature map size.

2. **Handling Border Information:** Without padding, the convolution operation can miss information at the edges of the image because the kernel cannot fully overlap with the pixels at the borders. Padding allows the kernel to cover every part of the image.

3. **Controlling the Receptive Field:** Padding can influence the receptive field of the convolutional layer. By adjusting the padding, one can control how much of the input the kernel covers, affecting the layer's ability to capture global or local information.

### Example with Python Code

Here's a simple example using Python and TensorFlow/Keras to demonstrate the effect of padding on an image.

```python
import tensorflow as tf
import numpy as np

# Create a random input image of size 6x6
input_image = np.random.rand(1, 6, 6, 1)  # Batch size, height, width, channels

# Define a 3x3 convolutional layer with valid padding
conv_valid = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding='valid')
output_valid = conv_valid(input_image)

# Define a 3x3 convolutional layer with same padding
conv_same = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding='same')
output_same = conv_same(input_image)

print("Input shape:", input_image.shape)
print("Output shape with valid padding:", output_valid.shape)
print("Output shape with same padding:", output_same.shape)
```

### Output:

```
Input shape: (1, 6, 6, 1)
Output shape with valid padding: (1, 4, 4, 1)
Output shape with same padding: (1, 6, 6, 1)
```

This example shows how the output shape changes based on the padding strategy.

### Conclusion

Padding is a crucial component in CNNs that helps control the size of the output feature maps and ensures that important information, especially at the borders of an image, is not lost. By carefully choosing the padding type, one can balance the trade-offs between preserving spatial dimensions and computational efficiency in CNN models.

### Strides in Convolutional Neural Networks (CNNs)

**Strides** in Convolutional Neural Networks (CNNs) refer to the step size with which the convolutional filter (or kernel) moves across the input image or feature map. The concept of strides is crucial in determining the size of the output feature map after the convolution operation. By adjusting the stride, we can control the spatial dimensions of the output and influence the computational efficiency and capacity of the network.

### How Strides Work

In a convolutional operation, a filter (or kernel) slides over the input matrix (e.g., an image), and at each position, it performs element-wise multiplication between the filter values and the input values covered by the filter. The results are then summed to produce a single value in the output feature map.

- **Stride of 1:** The filter moves one pixel at a time across the input. This is the default stride and is often used when the goal is to extract detailed features without reducing the spatial dimensions of the input too much.

- **Stride of 2 or More:** The filter skips pixels as it moves across the input. This results in a smaller output feature map and effectively downsamples the input.

### Mathematical Formulation

The output dimensions (height and width) of a convolutional layer are determined by the following formula:

\[
\text{Output height} = \left\lfloor \frac{\text{Input height} - \text{Kernel height} + 2 \times \text{Padding}}{\text{Stride}} \right\rfloor + 1
\]

\[
\text{Output width} = \left\lfloor \frac{\text{Input width} - \text{Kernel width} + 2 \times \text{Padding}}{\text{Stride}} \right\rfloor + 1
\]

Where:
- **Input height** and **Input width** are the dimensions of the input image.
- **Kernel height** and **Kernel width** are the dimensions of the filter.
- **Padding** is the number of pixels added around the input (usually 0 for valid padding).
- **Stride** is the step size at which the filter moves across the input.

### Examples of Strides

#### Example 1: Stride of 1
- **Input size:** \(6 \times 6\)
- **Kernel size:** \(3 \times 3\)
- **Padding:** 0 (valid padding)
- **Stride:** 1

**Output Calculation:**

\[
\text{Output height} = \left\lfloor \frac{6 - 3 + 0}{1} \right\rfloor + 1 = 4
\]
\[
\text{Output width} = \left\lfloor \frac{6 - 3 + 0}{1} \right\rfloor + 1 = 4
\]

- **Output size:** \(4 \times 4\)

The filter moves one pixel at a time, covering most of the input and producing a detailed feature map.

#### Example 2: Stride of 2
- **Input size:** \(6 \times 6\)
- **Kernel size:** \(3 \times 3\)
- **Padding:** 0 (valid padding)
- **Stride:** 2

**Output Calculation:**

\[
\text{Output height} = \left\lfloor \frac{6 - 3 + 0}{2} \right\rfloor + 1 = 2
\]
\[
\text{Output width} = \left\lfloor \frac{6 - 3 + 0}{2} \right\rfloor + 1 = 2
\]

- **Output size:** \(2 \times 2\)

The filter moves two pixels at a time, resulting in a smaller feature map with less detail.

### Visual Example

Consider a \(4 \times 4\) input and a \(2 \times 2\) filter:

1. **Stride = 1:**
   - The filter moves over every possible position.
   - The output size is \(3 \times 3\).

2. **Stride = 2:**
   - The filter moves by two pixels, skipping some positions.
   - The output size is \(2 \times 2\).

### Python Code Example

Here's a simple example using TensorFlow/Keras to demonstrate the effect of different strides on the output size:

```python
import tensorflow as tf
import numpy as np

# Create a random input image of size 6x6
input_image = np.random.rand(1, 6, 6, 1)  # Batch size, height, width, channels

# Define a 3x3 convolutional layer with stride 1
conv_stride_1 = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding='valid')
output_stride_1 = conv_stride_1(input_image)

# Define a 3x3 convolutional layer with stride 2
conv_stride_2 = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=2, padding='valid')
output_stride_2 = conv_stride_2(input_image)

print("Input shape:", input_image.shape)
print("Output shape with stride 1:", output_stride_1.shape)
print("Output shape with stride 2:", output_stride_2.shape)
```

### Output:

```
Input shape: (1, 6, 6, 1)
Output shape with stride 1: (1, 4, 4, 1)
Output shape with stride 2: (1, 2, 2, 1)
```

### Implications of Using Strides

1. **Downsampling:** Increasing the stride reduces the size of the output feature map, which effectively downsamples the input. This can be useful in reducing the computational load, but it may also lead to a loss of detail.

2. **Computational Efficiency:** Larger strides reduce the number of operations required because the filter performs fewer convolutions, making the model more computationally efficient.

3. **Information Loss:** While larger strides can speed up computation, they also reduce the resolution of the output, potentially leading to a loss of important spatial information.

4. **Receptive Field:** The receptive field (the area of the input that influences a particular output unit) increases with larger strides, which can be beneficial for capturing more global patterns in the input.

### Conclusion

Strides are a powerful tool in CNNs for controlling the spatial dimensions of the output feature maps and the computational efficiency of the model. By adjusting the stride, one can trade off between preserving detailed information and reducing the computational complexity of the network. Understanding the role of strides is essential for designing effective CNN architectures tailored to specific tasks.

**Pooling** is a key operation in Convolutional Neural Networks (CNNs) that is used to reduce the spatial dimensions (height and width) of the input feature maps, while retaining the most important information. Pooling helps to make the representation more manageable and reduces the computational complexity of the network, as well as to some extent controls overfitting.

### Types of Pooling

There are several types of pooling, each with a different method of summarizing the features in the input regions:

1. **Max Pooling**
2. **Average Pooling**
3. **Global Pooling**

#### 1. Max Pooling
Max pooling is the most common form of pooling. It works by sliding a window (of a specified size) over the input feature map and taking the maximum value from the windowed area.

- **Operation:** For each position of the window, the maximum value within that window is chosen and stored in the corresponding position in the output feature map.

- **Effect:** Max pooling reduces the dimensionality while preserving the most significant features (those with the highest values), which often correspond to edges or other prominent patterns.

**Example:**

Given a \(4 \times 4\) input matrix and a \(2 \times 2\) max pooling operation with a stride of 2:

| **Input**      | **Max-Pooled Output**  |
|----------------|------------------------|
| 1, 3, 2, 4     | **3**, **4**           |
| 5, 6, 8, 7     | **8**, **7**           |
| 1, 2, 3, 1     |                        |
| 0, 1, 0, 1     |                        |

The pooling window takes the maximum value within each \(2 \times 2\) subregion, resulting in a \(2 \times 2\) output matrix.

#### 2. Average Pooling
Average pooling works similarly to max pooling, but instead of taking the maximum value, it takes the average of all the values in the window.

- **Operation:** For each position of the window, the average of the values within that window is calculated and stored in the output feature map.

- **Effect:** Average pooling tends to retain more spatial information compared to max pooling, but the resulting features may not be as sharp.

**Example:**

Given the same \(4 \times 4\) input matrix and a \(2 \times 2\) average pooling operation with a stride of 2:

| **Input**      | **Average-Pooled Output** |
|----------------|---------------------------|
| 1, 3, 2, 4     | **3.75**, **5.25**         |
| 5, 6, 8, 7     | **1.25**, **1.25**         |
| 1, 2, 3, 1     |                           |
| 0, 1, 0, 1     |                           |

The pooling window takes the average value within each \(2 \times 2\) subregion, resulting in a \(2 \times 2\) output matrix.

#### 3. Global Pooling
Global pooling is a special case where the pooling window size is equal to the size of the input feature map, resulting in a single value for each feature map. This is often used in the final layers of a CNN before the output layer.

- **Operation:** The entire feature map is reduced to a single value, either by taking the maximum value (global max pooling) or the average value (global average pooling).

- **Effect:** Global pooling drastically reduces the dimensionality and is often used in classification tasks to convert the feature map into a single vector that can be fed into a fully connected layer.

### Mathematical Formulation

Given an input feature map of size \( H \times W \) and a pooling window of size \( F \times F \), with a stride \( S \), the output feature map size can be calculated as:

\[
\text{Output Height} = \left\lfloor \frac{H - F}{S} \right\rfloor + 1
\]

\[
\text{Output Width} = \left\lfloor \frac{W - F}{S} \right\rfloor + 1
\]

Where:
- **H, W** are the height and width of the input feature map.
- **F** is the size of the pooling window (e.g., 2 for \(2 \times 2\) pooling).
- **S** is the stride.

### Example Calculation with Max Pooling

Consider an input feature map of size \(6 \times 6\) with a \(2 \times 2\) max pooling operation and stride 2:

- **Input size:** \(6 \times 6\)
- **Pooling window size:** \(2 \times 2\)
- **Stride:** 2

**Output Calculation:**

\[
\text{Output Height} = \left\lfloor \frac{6 - 2}{2} \right\rfloor + 1 = 3
\]

\[
\text{Output Width} = \left\lfloor \frac{6 - 2}{2} \right\rfloor + 1 = 3
\]

- **Output size:** \(3 \times 3\)

### Python Code Example

Here’s a simple example using TensorFlow/Keras to perform max pooling on a feature map:

```python
import tensorflow as tf
import numpy as np

# Create a random input feature map of size 6x6
input_feature_map = np.random.rand(1, 6, 6, 1)  # Batch size, height, width, channels

# Define a max pooling layer with a 2x2 window and stride of 2
max_pooling_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
output_feature_map = max_pooling_layer(input_feature_map)

print("Input shape:", input_feature_map.shape)
print("Output shape after max pooling:", output_feature_map.shape)
```

### Output:

```
Input shape: (1, 6, 6, 1)
Output shape after max pooling: (1, 3, 3, 1)
```

### Benefits of Pooling

1. **Dimensionality Reduction:** Pooling reduces the spatial dimensions of the input, which helps decrease the number of parameters and computation in the network, making the model more efficient.

2. **Translation Invariance:** Pooling helps the model become more invariant to small translations in the input. This means that slight movements of the input will not significantly affect the output.

3. **Control Overfitting:** By reducing the spatial dimensions and abstracting the input, pooling layers can help in controlling overfitting by generalizing the extracted features.

### Considerations

- **Choice of Pooling Type:** Max pooling is often preferred for tasks where the presence of a feature is more important than its precise location, while average pooling is used when spatial information is more critical.
- **Window Size and Stride:** These parameters must be chosen carefully to balance the trade-off between dimensionality reduction and retaining useful information.

### Conclusion

Pooling is a fundamental operation in CNNs that plays a crucial role in reducing the size of feature maps while preserving important information. It helps improve computational efficiency, controls overfitting, and provides translation invariance to the network. By understanding and carefully choosing the type of pooling, window size, and stride, one can effectively design CNN architectures tailored to specific tasks.

Here’s a comparison table summarizing the convolution operations with different configurations involving kernels, padding, and strides:

| **Operation**              | **Convolution Formula**                                                                                                        | **Features**                                                                                                                                                      | **Benefits**                                                                                                   | **Advantages**                                                                                                                                              | **Disadvantages**                                                                                                                                              |
|----------------------------|--------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **1. Kernel Only**          | \(\text{Output Size} = \left\lfloor \frac{(H - K)}{1} + 1 \right\rfloor\) \) \(\times \left\lfloor \frac{(W - K)}{1} + 1 \right\rfloor\)                     | - No padding applied.<br/> - Stride is 1 (default).<br/> - Output size decreases by \(K-1\) in each dimension.<br/> - Kernel size determines the receptive field. | - Simple to implement.<br/> - Captures local spatial features effectively.                                | - Reduces dimensionality without any additional operations.<br/> - Helps to quickly reduce feature map size.                                                   | - May lose border information.<br/> - Output size can decrease significantly, which may lead to excessive reduction in information as the network deepens.      |
| **2. Padding + Kernel**     | \(\text{Output Size} = \left\lfloor \frac{(H - K + 2P)}{1} + 1 \right\rfloor\) \) \(\times \left\lfloor \frac{(W - K + 2P)}{1} + 1 \right\rfloor\)            | - Padding \(P\) is applied.<br/> - Stride is 1 (default).<br/> - Retains the original input size (for \(P = \frac{K-1}{2}\)).                                   | - Preserves spatial resolution.<br/> - Helps in retaining border information.                               | - Ensures that the output size remains close to the input size.<br/> - Prevents loss of important edge features.                                                 | - Increases computational complexity due to padding.<br/> - May require careful choice of padding size to maintain desired output dimensions.                   |
| **3. Kernel + Strides**     | \(\text{Output Size} = \left\lfloor \frac{(H - K)}{S} + 1 \right\rfloor\) \) \(\times \left\lfloor \frac{(W - K)}{S} + 1 \right\rfloor\)                     | - No padding applied.<br/> - Stride \(S\) > 1.<br/> - Output size decreases based on the stride.                                                                  | - Reduces dimensionality more aggressively.<br/> - Reduces computational cost.                              | - Effective in downsampling feature maps.<br/> - Reduces the number of parameters and computation in deeper layers.                                              | - Higher stride may lead to loss of finer details.<br/> - May not capture all features, especially those located in between strides.                             |
| **4. Padding + Kernel + Strides** | \(\text{Output Size} = \left\lfloor \frac{(H - K + 2P)}{S} + 1 \right\rfloor\) \) \(\times \left\lfloor \frac{(W - K + 2P)}{S} + 1 \right\rfloor\)    | - Padding \(P\) applied.<br/> - Stride \(S\) > 1.<br/> - Output size is a function of padding and stride.                                                       | - Combines benefits of both padding and strides.<br/> - Allows flexible control over output dimensions.      | - Provides balance between preserving information (padding) and reducing dimensionality (strides).<br/> - Retains important spatial features while downsampling. | - Complex to optimize padding and stride together.<br/> - May still lead to loss of detail if strides are too large relative to the kernel size and padding.     |

### Explanation of Terms:

- **Kernel (K):** The size of the filter applied during convolution. It determines the receptive field, i.e., the area of the input covered by the kernel at any one time.

- **Padding (P):** The number of pixels added to the input image around the borders to control the spatial dimensions of the output. Padding helps to retain more border information and prevents the shrinking of the input dimensions.

- **Strides (S):** The step size with which the kernel moves across the input feature map. Strides greater than 1 reduce the spatial dimensions of the output, leading to downsampling.

### Key Takeaways:

- **Kernel Only:** Useful when you want to simply convolve the input without any padding or stride adjustment. This operation will reduce the dimensions of the input feature map, which might be desirable for early layers in the network.

- **Padding + Kernel:** Helps retain the original size of the input, ensuring that no information is lost due to the reduction in dimensions. This is often used in cases where spatial resolution needs to be preserved.

- **Kernel + Strides:** Useful for aggressively reducing the spatial dimensions and thus the computational cost. However, it should be used cautiously as it can skip over important features.

- **Padding + Kernel + Strides:** This combination provides the most flexibility, allowing for a balanced approach between retaining information and reducing dimensionality. It's commonly used in modern CNN architectures to manage both spatial resolution and computational efficiency.