import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageDraw, ImageOps

# Load the trained model
model = tf.keras.models.load_model("Model.keras")  # Update with your model file name

# Preprocess the drawn image
def preprocess_image(img):
    img = img.resize((28, 28))
    img = img.convert('L')  # Convert to grayscale
    img = ImageOps.invert(img)  # Invert the image

    img = np.array(img)

    if img.sum() == 0:
        return None  # Return None if no drawing is detected

    non_empty_columns = np.where(img.min(axis=0) < 255)[0]
    non_empty_rows = np.where(img.min(axis=1) < 255)[0]
    if len(non_empty_rows) == 0 or len(non_empty_columns) == 0:
        return None
    
    crop_box = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
    img = img[crop_box[0]:crop_box[1]+1, crop_box[2]:crop_box[3]+1]
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    return img

# Predict the digit
def predict_digit(img):
    img = preprocess_image(img)
    if img is None:
        return "No digit detected", 0  

    cv2.imshow("Preprocessed Image", img[0].reshape(28, 28) * 255)  
    cv2.waitKey(1)  

    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)
    confidence = max(prediction[0])
    
    return predicted_digit, confidence

# Create the Tkinter GUI
class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        
        # Set up the window to be full screen
        self.attributes("-fullscreen", True)
        self.configure(bg="#2C3E50")  # Dark background
        
        # Title Label
        self.title_label = tk.Label(self, text="Handwritten Digit Recognition", 
                                     font=("Helvetica", 36, "bold"), bg="#2C3E50", fg="#E74C3C")  # Changed to red
        self.title_label.pack(pady=20)

        # Stylish Frame for the Canvas
        self.frame = tk.Frame(self, bg="#34495E", bd=5, relief=GROOVE)
        self.frame.pack(pady=20)

        # Canvas for drawing
        self.canvas = tk.Canvas(self.frame, width=400, height=400, bg='white', cursor='cross', highlightthickness=2)
        self.canvas.pack(padx=10, pady=10)
        
        # Clear button
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_canvas, bg="#E67E22", font=("Helvetica", 14, "bold"), relief=RAISED)
        self.button_clear.pack(side=LEFT, padx=50, pady=10)

        # Prediction button
        self.button_predict = tk.Button(self, text="Predict", command=self.predict_digit, bg="#3498DB", font=("Helvetica", 14, "bold"), relief=RAISED)
        self.button_predict.pack(side=RIGHT, padx=50, pady=10)

        # Prediction label
        self.label = tk.Label(self, text="Draw a digit and click 'Predict'", font=("Helvetica", 18), bg="#2C3E50", fg="#ECF0F1")
        self.label.pack(pady=10)
        
        # Creator credits
        self.footer_label = tk.Label(self, text="Created by : Vaibhav Jawkar , Tushar Sakhare , Vedant Bodkhe", 
                                      font=("Helvetica", 12, "italic"), bg="#2C3E50", fg="#ECF0F1")
        self.footer_label.pack(side=BOTTOM, pady=10)

        # Add an interactive footer
        self.info_label = tk.Label(self, text="Interact with the canvas!", 
                                    font=("Helvetica", 14, "bold"), bg="#2C3E50", fg="#F39C12")
        self.info_label.pack(side=BOTTOM, pady=10)

        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        
        # PIL image for drawing
        self.image1 = Image.new("RGB", (400, 400), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image1)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle((0, 0, 400, 400), fill=(255, 255, 255))
        self.label.configure(text="Draw a digit and click 'Predict'")
        
    def draw_lines(self, event):
        x, y = event.x, event.y
        r = 10  # Thicker brush size
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black')
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill='black')

    def predict_digit(self):
        digit, acc = predict_digit(self.image1)
        if digit == "No digit detected":
            self.label.configure(text="No digit detected, please try again!")
        else:
            self.label.configure(text=f"Predicted: {digit} (Confidence: {acc*100:.2f}%)")

# Start the application
app = App()
app.mainloop()

# Clean up OpenCV windows
cv2.destroyAllWindows()
