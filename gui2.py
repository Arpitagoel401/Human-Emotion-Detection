import tkinter as tk
from tkinter import filedialog
from tkinter import *
from tensorflow.keras.models import model_from_json # type: ignore
from PIL import Image, ImageTk
import numpy as np
import cv2

class EmotionDetectorApp:
    def __init__(self, master):
        self.master = master
        self.master.title('Emotion Detector')
        self.master.geometry('800x600')
        self.master.configure(background='#CDCDCD')

        # Labels for displaying images and text
        self.label1 = Label(self.master, background='#CDCDCD', font=('arial', 15, 'bold'))
        self.sign_image = Label(self.master)

        # Load the face detection model and emotion recognition model
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.model = self.load_model("model_a.json", "model_weights.weights.h5")
        self.EMOTION_LIST = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

        # GUI Heading
        self.heading = Label(self.master, text='Emotion Detector', pady=20, font=('arial', 25, 'bold'))
        self.heading.configure(background="#CDCDCD", foreground="#364156")
        self.heading.pack()

        # Upload, Real-Time Detection, and Stop buttons
        self.upload_button = Button(self.master, text="Upload Image", command=self.upload_image, padx=10, pady=5)
        self.upload_button.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
        self.upload_button.pack(side='bottom', pady=20)

        self.real_time_button = Button(self.master, text="Start Real-Time Detection", command=self.start_real_time, padx=10, pady=5)
        self.real_time_button.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
        self.real_time_button.pack(side='bottom', pady=10)

        self.stop_button = Button(self.master, text="Stop Real-Time Detection", command=self.stop_real_time, padx=10, pady=5)
        self.stop_button.configure(background="#FF6347", foreground='white', font=('arial', 20, 'bold'))
        self.stop_button.pack(side='bottom', pady=10)

        # Initially, hide the Stop button
        self.stop_button.pack_forget()

        # Pack the image label and text label
        self.sign_image.pack(side='bottom', expand=True)
        self.label1.pack(side='bottom', expand=True)

        # Initialize webcam capture
        self.cap = None

    def load_model(self, json_file, weights_file):
        """Load the trained emotion detection model"""
        with open(json_file, "r") as file:
            loaded_model_json = file.read()
            model = model_from_json(loaded_model_json)
        model.load_weights(weights_file)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def detect_emotion(self, frame):
        """Detect emotion from a single image/frame"""
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_image, 1.3, 5)
        if len(faces) == 0:
            self.label1.configure(foreground="#011638", text="Unable to detect any faces")
            return

        for (x, y, w, h) in faces:
            fc = gray_image[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            pred = self.EMOTION_LIST[np.argmax(self.model.predict(roi[np.newaxis, :, :, np.newaxis]))]
            self.label1.configure(foreground="#011638", text=pred)

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    def show_frame(self):
        """Display the current frame from the webcam"""
        ret, frame = self.cap.read()
        if ret:
            self.detect_emotion(frame)

            # Convert frame to PIL Image and then to Tkinter PhotoImage
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)

            # Update the image in the label
            self.sign_image.configure(image=img_tk)
            self.sign_image.image = img_tk

        # Continuously display the frame
        self.master.after(10, self.show_frame)

    def start_real_time(self):
        """Start real-time webcam capture for emotion detection"""
        # Open the webcam
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            self.label1.configure(foreground="#011638", text="Error: Unable to access webcam")
            return

        # Show the stop button when real-time detection starts
        self.stop_button.pack(side='bottom', pady=10)

        # Hide other buttons
        self.upload_button.pack_forget()
        self.real_time_button.pack_forget()

        self.show_frame()

    def stop_real_time(self):
        """Stop real-time webcam capture"""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.label1.configure(foreground="#011638", text="Real-time detection stopped")

        # Hide the stop button and show the upload and real-time buttons again
        self.stop_button.pack_forget()

        self.upload_button.pack(side='bottom', pady=20)
        self.real_time_button.pack(side='bottom', pady=10)

        self.label1.configure(text="")

    def upload_image(self):
        """Handle image upload for emotion detection"""
        try:
            file_path = filedialog.askopenfilename()
            if not file_path:
                return

            uploaded = Image.open(file_path)
            uploaded.thumbnail(((self.master.winfo_width() // 2.3), (self.master.winfo_height() // 2.3)))
            im = ImageTk.PhotoImage(uploaded)

            self.sign_image.configure(image=im)
            self.sign_image.image = im
            self.label1.configure(text='')
            self.detect_emotion(cv2.imread(file_path))
        except Exception as e:
            self.label1.configure(foreground="#011638", text="Error loading image")
            print(f"Error: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectorApp(root)
    root.mainloop()
