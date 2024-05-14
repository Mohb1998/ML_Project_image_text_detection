import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from predict import extract_and_predict


class WelcomePage(tk.Frame):
    def __init__(self, parent, next_callback):
        super().__init__(parent)

        self.message_label = tk.Label(
            self,
            text="""Welcome to our handwritten to text converter.
            Press next to select an image""",
            font=("Arial", 14),
        )
        self.message_label.pack(pady=20)

        self.next_button = tk.Button(
            self, text="Next", command=next_callback, font=("Arial", 12)
        )
        self.next_button.pack(pady=10)


class ImageToTextConverter(tk.Tk):
    def __init__(self, cnn_model, encoder):
        super().__init__()
        self.title("Image to Text Converter")

        self.welcome_page = WelcomePage(self, self.show_interface)
        self.welcome_page.pack()

        self.cnn_model = cnn_model
        self.encoder = encoder

        self.interface_page = None

    def show_interface(self):
        self.welcome_page.pack_forget()
        self.interface_page = InterfacePage(self, self.cnn_model, self.encoder)
        self.interface_page.pack()


class InterfacePage(tk.Frame):
    def __init__(self, parent, cnn_model, encoder):
        super().__init__(parent)

        self.cnn_model = cnn_model
        self.encoder = encoder
        self.message_label = tk.Label(
            self,
            text="Press the button below to select an image",
            font=("Arial", 14),
        )
        self.message_label.pack(pady=20)
        self.select_image_button = tk.Button(
            self, text="Select Image", command=self.open_image, font=("Arial", 12)
        )
        self.select_image_button.pack(pady=10)

        self.result_frame = tk.Frame(self)
        self.result_frame.pack()

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            predicted_text, image, prediction_duration = extract_and_predict(
                file_path, self.cnn_model, self.encoder
            )
            self.display_result(predicted_text, image, prediction_duration)

    def display_result(self, predicted_text, image, prediction_duration):
        self.result_frame.destroy()
        self.result_frame = tk.Frame(self)
        self.result_frame.pack()

        result_label = tk.Label(
            self.result_frame,
            text=f"Predicted Text: {predicted_text}\n"
            f"Prediction Time: {prediction_duration:.2f} seconds",
            font=("Arial", 12),
        )
        result_label.pack(pady=20)

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (200, 200))
        img = Image.fromarray(img)
        self.photo_image = ImageTk.PhotoImage(img)

        image_label = tk.Label(self.result_frame, image=self.photo_image)
        image_label.pack(pady=10)

        try_again_button = tk.Button(
            self.result_frame,
            text="Try Again",
            font=("Arial", 12),
            padx=10,
            pady=5,
            command=self.open_image,
        )
        try_again_button.pack(side="left", padx=10)

        exit_button = tk.Button(
            self.result_frame,
            text="Exit",
            font=("Arial", 12),
            padx=10,
            pady=5,
            command=self.quit,
        )
        exit_button.pack(side="right", padx=10)


def initialize_interface(cnn_model, encoder):
    app = ImageToTextConverter(cnn_model, encoder)
    app.mainloop()
