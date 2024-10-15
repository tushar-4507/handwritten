
from sklearn.datasets import fetch_openml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for displaying plots

# Load the MNIST dataset
mnist = fetch_openml('mnist_784')
x, y = mnist['data'], mnist['target']

# Convert x to a DataFrame
x = pd.DataFrame(x)

# Reset the index of x and store it in x_train_reset
x_train_reset = x.reset_index(drop=True)

# Shuffle the DataFrame using the sample method
shuffled_x_train = x_train_reset.sample(frac=1, random_state=42)

# Similarly, shuffle the corresponding y values
y_train = pd.Series(y).reset_index(drop=True)
shuffled_y_train = y_train.sample(frac=1, random_state=42)

# Split the data into training and testing sets
x_train, x_test = shuffled_x_train[:60000], shuffled_x_train[60000:]
y_train, y_test = shuffled_y_train[:60000], shuffled_y_train[60000:]

# Visualize a digit from the dataset
def visualize_digit(data, index=0):
    some_digit = data.iloc[index]
    some_digit_image = some_digit.values.reshape(28, 28)
    plt.imshow(some_digit_image, cmap='binary')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    visualize_digit(x_train)
