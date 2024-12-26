import cv2
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection App")
        
        # Load the pre-trained face detection classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize variables
        self.current_image = None
        self.photo = None
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Create buttons
        self.load_button = tk.Button(
            self.root, 
            text="Load Image", 
            command=self.load_image,
            padx=20,
            pady=10
        )
        self.load_button.pack(pady=10)
        
        self.detect_button = tk.Button(
            self.root, 
            text="Detect Faces", 
            command=self.detect_faces,
            padx=20,
            pady=10,
            state=tk.DISABLED
        )
        self.detect_button.pack(pady=10)
        
        # Create canvas for image display
        self.canvas = tk.Canvas(self.root, width=800, height=600)
        self.canvas.pack(pady=10)
        
    def load_image(self):
        # Open file dialog
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")
            ]
        )
        
        if file_path:
            # Load the image
            self.current_image = cv2.imread(file_path)
            if self.current_image is None:
                messagebox.showerror("Error", "Could not load image!")
                return
                
            # Convert BGR to RGB for display
            rgb_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            
            # Resize image to fit canvas while maintaining aspect ratio
            height, width = rgb_image.shape[:2]
            max_size = (800, 600)
            scale = min(max_size[0]/width, max_size[1]/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            rgb_image = cv2.resize(rgb_image, (new_width, new_height))
            
            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(Image.fromarray(rgb_image))
            
            # Update canvas
            self.canvas.config(width=new_width, height=new_height)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            
            # Enable detect button
            self.detect_button.config(state=tk.NORMAL)
            
    def detect_faces(self):
        if self.current_image is None:
            return
            
        # Convert to grayscale
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Create a copy of the image
        image_with_faces = self.current_image.copy()
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(
                image_with_faces,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )
        
        # Convert BGR to RGB for display
        rgb_image = cv2.cvtColor(image_with_faces, cv2.COLOR_BGR2RGB)
        
        # Resize image
        height, width = rgb_image.shape[:2]
        max_size = (800, 600)
        scale = min(max_size[0]/width, max_size[1]/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        rgb_image = cv2.resize(rgb_image, (new_width, new_height))
        
        # Update display
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb_image))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Show number of faces detected
        messagebox.showinfo(
            "Detection Complete",
            f"Found {len(faces)} faces in the image!"
        )

def main():
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()