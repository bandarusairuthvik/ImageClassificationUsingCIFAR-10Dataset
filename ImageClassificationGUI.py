import tkinter as tk
from tkinter import filedialog, Button, Label
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model

# Load the trained model to classify the images
model = load_model('model1_cifar_10epoch.h5')

# Dictionary to label all the CIFAR-10 dataset classes
class_mapping = { 
    0: 'aeroplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck' 
}

# Initialise GUI
window = tk.Tk()
window.geometry('800x600')
window.title('Image Classification CIFAR10')
window.configure(background='#CDCDCD')
result_label = Label(window, background='#CDCDCD', font=('arial', 15, 'bold'))
image_display = Label(window)

def classify_image(file_path):
    image = Image.open(file_path)
    image = image.resize((32, 32))
    image_array = np.expand_dims(np.array(image), axis=0)
    prediction = model.predict_classes(image_array)[0]
    predicted_class = class_mapping[prediction]
    result_label.configure(foreground='#011638', text=predicted_class)

def display_classify_button(file_path):
    classify_button = Button(window, text="Classify Image", command=lambda: classify_image(file_path), padx=10, pady=5)
    classify_button.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_button.place(relx=0.79, rely=0.46)

def open_image():
    try:
        file_path = filedialog.askopenfilename()
        image = Image.open(file_path)
        image.thumbnail(((window.winfo_width() / 2.25), (window.winfo_height() / 2.25)))
        image_tk = ImageTk.PhotoImage(image)
        image_display.configure(image=image_tk)
        image_display.image = image_tk
        result_label.configure(text='')
        display_classify_button(file_path)
    except:
        pass

upload_button = Button(window, text="Upload an image", command=open_image, padx=10, pady=5)
upload_button.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
upload_button.pack(side=BOTTOM, pady=50)
image_display.pack(side=BOTTOM, expand=True)
result_label.pack(side=BOTTOM, expand=True)
title_label = Label(window, text="Image Classification CIFAR10", pady=20, font=('arial', 20, 'bold'))
title_label.configure(background='#CDCDCD', foreground='#364156')
title_label.pack()
window.mainloop()