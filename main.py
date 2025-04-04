import os  # Used for handling file and directory operations
import cv2  # OpenCV library for computer vision tasks like image processing and camera handling
import numpy as np  # Library for numerical operations, especially with arrays (like images)
import tkinter as tk  # GUI toolkit for creating the application interface
from tkinter import filedialog, messagebox, Radiobutton, StringVar  # Specific tkinter modules for file dialogs and messages
from PIL import Image, ImageTk  # Pillow library for image manipulation and displaying in tkinter
from deepface import DeepFace  # DeepFace library for facial analysis (age, gender, emotion)
from mtcnn import MTCNN  # MTCNN library for accurate face detection in slow mode
from tkinter import ttk  # Themed widgets for tkinter (e.g., sliders)
import ssl  # Used to disable SSL verification for model downloads
from datetime import datetime  # For timestamping detections
import time  # For timing operations like FPS calculation

# Disable SSL verification to allow DeepFace to download models without SSL issues
ssl._create_default_https_context = ssl._create_unverified_context

# Default settings for connecting to a mobile camera via IP Webcam app
DEFAULT_IP = "192.168.192.76"  # Default IP address (change based on your mobile device's IP)
DEFAULT_PORT = "8080"  # Default port number for the IP Webcam server

# Initialize the MTCNN face detector for slow mode (used globally for accurate detection)
detector = MTCNN()

# Function to preprocess image frames before detection
def preprocess_frame(frame):
    """Enhance image quality for better face detection.
    
    Args:
        frame: The input image frame from the camera or file.
    Returns:
        Processed frame ready for detection.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR (OpenCV default) to RGB
    frame_rgb = cv2.GaussianBlur(frame_rgb, (5, 5), 0)  # Apply blur to reduce noise
    frame_rgb = cv2.normalize(frame_rgb, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)  # Adjust brightness/contrast
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV compatibility

# Function to draw a fancy label box for slow mode detections
def draw_creative_label(image, startX, startY, endX, endY, label, timestamp):
    """Draw a box with text below a detected face, including a timestamp.
    
    Args:
        image: The image to draw on.
        startX, startY, endX, endY: Coordinates of the face bounding box.
        label: Text to display (e.g., 'Male, 25, Happy').
        timestamp: Current time for the detection.
    """
    # Calculate sizes of the text to be drawn
    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
    timestamp_text = f"Time: {timestamp.strftime('%H:%M:%S')}"  # Format time as HH:MM:SS
    (ts_width, ts_height), _ = cv2.getTextSize(timestamp_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    
    # Define the box position below the face
    margin = 10
    padding = 5
    text_top = endY + margin
    text_bottom = text_top + label_height + ts_height + padding * 3
    text_left = startX
    text_right = text_left + max(label_width, ts_width) + margin * 2

    # Draw a black rectangle as the background
    cv2.rectangle(image, (text_left, text_top), (text_right, text_bottom), (0, 0, 0), -1)
    # Add the label text (e.g., gender, age, emotion)
    cv2.putText(image, label, (text_left + margin, text_top + label_height + padding),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
    # Add the timestamp below the label
    cv2.putText(image, timestamp_text, (text_left + margin, text_top + label_height + ts_height + padding * 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

# Function for slow, accurate face detection using MTCNN and DeepFace
def process_detection_slow(frame, conf_threshold=0.8):
    """Detect and analyze faces in an image using slow but accurate methods.
    
    Args:
        frame: The input image frame.
        conf_threshold: Minimum confidence level for face detection (default 0.8).
    Returns:
        Frame with detection results drawn on it.
    """
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for MTCNN
        detections = detector.detect_faces(rgb_frame)  # Use MTCNN to find faces
        timestamp = datetime.now()  # Record the time of detection

        for detection in detections:
            confidence = detection['confidence']
            if confidence > conf_threshold:  # Only process confident detections
                x, y, w, h = detection['box']  # Get face coordinates
                # Ensure coordinates stay within image bounds
                startX, startY = max(0, x), max(0, y)
                endX = min(frame.shape[1] - 1, startX + w)
                endY = min(frame.shape[0] - 1, startY + h)

                face_roi = frame[startY:endY, startX:endX]  # Extract the face area
                if face_roi.size == 0:  # Skip if the face area is empty
                    continue

                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)  # Convert face to RGB for DeepFace

                try:
                    # Analyze the face for age, gender, and emotion
                    result = DeepFace.analyze(face_rgb, actions=['age', 'gender', 'emotion'], enforce_detection=False)
                    age = result[0]['age']
                    if not (0 <= age <= 100):  # Check if age is realistic
                        age = "Uncertain"
                    # gender = result[0]['dominant_gender']
                    emotion = result[0]['dominant_emotion']
                    label = f"{age}, {emotion}"  # Create the label text

                    # Draw a rectangle around the face
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 150, 255), 3)
                    # Draw the fancy label with timestamp
                    draw_creative_label(frame, startX, startY, endX, endY, label, timestamp)
                except Exception as e:
                    print(f"Error in DeepFace analysis: {e}")
                    continue
    except Exception as e:
        print(f"Error in slow detection: {e}")
    return frame

# Class for fast, real-time face detection using OpenCV DNN models
class FastDetector:
    def __init__(self):
        """Set up the fast detector with pre-trained models for face, gender, and age."""
        # Load the face detection model (SSD with Caffe)
        self.face_net = cv2.dnn.readNetFromCaffe('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
        # Load the gender detection model
        self.gender_net = cv2.dnn.readNetFromCaffe('models/gender_deploy.prototxt', 'models/gender_net.caffemodel')
        # Load the age detection model
        self.age_net = cv2.dnn.readNetFromCaffe('models/age_deploy.prototxt', 'models/age_net.caffemodel')
        
        # Define possible outputs for gender and age
        self.gender_list = ['Male', 'Female']
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    
    def process_detection_fast(self, frame):
        """Detect and analyze faces in real-time using fast models, with optional DeepFace emotion detection.
        
        Args:
            frame: The input image frame.
        Returns:
            Frame with detection results drawn on it.
        """
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (300, 300))
        blob = cv2.dnn.blobFromImage(small_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")
                face_roi = frame[startY:endY, startX:endX]
                if face_roi.size == 0:
                    continue
                
                # Prepare the face for gender and age models (227x227, BGR)
                face_blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.43, 87.77, 114.90), swapRB=False)
                
                # Predict gender
                # self.gender_net.setInput(face_blob)
                # gender_pred = self.gender_net.forward()
                # gender = self.gender_list[gender_pred[0].argmax()]
                
                # Predict age
                self.age_net.setInput(face_blob)
                age_pred = self.age_net.forward()
                age = self.age_list[age_pred[0].argmax()]
                
                # Attempt to detect emotion using DeepFace
                label = f"{age}"
                try:
                    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    result = DeepFace.analyze(face_rgb, actions=['emotion'], enforce_detection=False)
                    emotion = result[0]['dominant_emotion']
                    label = f"{age}, {emotion}"
                except Exception as e:
                    print(f"Emotion detection with DeepFace failed: {e}")
                    # Skip emotion if DeepFace fails (keeps fast mode performant)
                
                # Draw the results on the frame
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame

# Class to handle camera input (local or mobile) with manual processing toggle
class CameraHandler:
    def __init__(self, mode):
        """Initialize the camera handler with the chosen mode (slow or fast).
        
        Args:
            mode: 'slow' for accurate detection or 'fast' for real-time detection.
        """
        self.cap = None  # Camera object (to be set later)
        self.running = False  # Flag to control the camera loop
        self.mode = mode  # Detection mode
        self.fast_detector = FastDetector()  # Used for fast mode processing
        self.state = 'capture'  # Start in capture mode (live feed only)

    def start_mobile_camera(self, ip, port, zoom=1.0):
        """Connect to a mobile camera using IP and port from the IP Webcam app.
        
        Args:
            ip: IP address of the mobile device.
            port: Port number for the camera server.
            zoom: Zoom level for the camera (default 1.0).
        Returns:
            True if successful, False otherwise.
        """
        url = f"http://{ip}:{port}/video?zoom={int(zoom*100)}"  # Build the camera URL
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            messagebox.showerror("Connection Failed", f"Failed to connect to {url}")
            return False
        self.running = True
        self.process_stream()  # Start processing the video stream
        return True

    def start_local_camera(self):
        """Open the local webcam with a higher resolution.
        
        Returns:
            True if successful, False otherwise.
        """
        self.cap = cv2.VideoCapture(0)  # Open default camera (index 0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width to 1280 pixels
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height to 720 pixels
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open local camera")
            return False
        self.running = True
        self.process_stream()  # Start processing the video stream
        return True

    def process_stream(self):
        """Handle the camera stream with manual toggle for processing."""
        # Create and size the display window
        cv2.namedWindow("Camera Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera Detection", 800, 600)
        prev_time = time.time()  # For FPS calculation

        while self.running:
            ret, frame = self.cap.read()
            if not ret:  # If reading fails, stop the loop
                self.running = False
                break
            frame = preprocess_frame(frame)  # Preprocess the frame
            height, width = frame.shape[:2]  # Get frame dimensions

            # Define a human-shaped boundary for face positioning guidance
            head_center = (width // 2, height // 4)  # Center of the head ellipse
            head_size = (width // 4, height // 4)  # Size of the head ellipse
            body_top_left = (width // 3, height // 2)  # Top-left of body trapezoid
            body_top_right = (2 * width // 3, height // 2)  # Top-right
            body_bottom_left = (width // 4, height)  # Bottom-left
            body_bottom_right = (3 * width // 4, height)  # Bottom-right

            # Draw the boundary on the frame for guidance
            cv2.ellipse(frame, head_center, head_size, 0, 0, 360, (0, 255, 0), 2)  # Head ellipse
            cv2.line(frame, body_top_left, body_top_right, (0, 255, 0), 2)  # Top of body
            cv2.line(frame, body_top_right, body_bottom_right, (0, 255, 0), 2)  # Right side
            cv2.line(frame, body_bottom_right, body_bottom_left, (0, 255, 0), 2)  # Bottom
            cv2.line(frame, body_bottom_left, body_top_left, (0, 255, 0), 2)  # Left side

            if self.state == 'capture':
                status_text = "Processing: OFF | Press P to start processing"
            elif self.state == 'processing':
                if self.mode == 'slow':
                    frame = process_detection_slow(frame)
                else:
                    frame = self.fast_detector.process_detection_fast(frame)
                status_text = "Processing: ON | Press P to stop processing"

            # Calculate and display frames per second (FPS)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display status text
            cv2.putText(frame, status_text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display general controls
            cv2.putText(frame, "ESC: Exit | P: Toggle Processing", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show the frame in the window
            cv2.imshow("Camera Detection", frame)

            # Handle keyboard input
            key = cv2.waitKey(10) & 0xFF  # Wait 10ms for a key press
            if key == 27:  # ESC key to stop
                self.running = False
            elif key in (ord('p'), ord('P')):  # 'P' key to toggle processing
                if self.state == 'capture':
                    self.state = 'processing'
                else:
                    self.state = 'capture'

        # Clean up when done
        self.cap.release()
        cv2.destroyAllWindows()

# Window for setting up a mobile camera connection
class IPConfigWindow(tk.Toplevel):
    def __init__(self, parent, callback):
        """Create a window to input mobile camera settings.
        
        Args:
            parent: The main application window.
            callback: Function to call with the entered settings.
        """
        super().__init__(parent)
        self.title("IP Camera Configuration")
        self.geometry("400x300")
        self.configure(bg="#333333")  # Dark background
        self.callback = callback
        self.zoom_level = tk.DoubleVar(value=1.0)  # Default zoom
        self.create_widgets()

    def create_widgets(self):
        """Set up the input fields and buttons for IP configuration."""
        tk.Label(self, text="IP Camera Settings", font=("Arial", 14, "bold"), bg="#333333", fg="#DDDDDD").pack(pady=10)
        # IP address input
        ip_frame = tk.Frame(self, bg="#333333")
        tk.Label(ip_frame, text="IP Address:", font=("Arial", 12), bg="#333333", fg="#DDDDDD").pack(side=tk.LEFT)
        self.ip_entry = tk.Entry(ip_frame, width=15, font=("Arial", 12))
        self.ip_entry.insert(0, DEFAULT_IP)
        self.ip_entry.pack(side=tk.LEFT, padx=5)
        ip_frame.pack(pady=5)
        # Port input
        port_frame = tk.Frame(self, bg="#333333")
        tk.Label(port_frame, text="Port:", font=("Arial", 12), bg="#333333", fg="#DDDDDD").pack(side=tk.LEFT)
        self.port_entry = tk.Entry(port_frame, width=6, font=("Arial", 12))
        self.port_entry.insert(0, DEFAULT_PORT)
        self.port_entry.pack(side=tk.LEFT, padx=5)
        port_frame.pack(pady=5)
        # Zoom slider
        zoom_frame = tk.Frame(self, bg="#333333")
        tk.Label(zoom_frame, text="Zoom Level:", font=("Arial", 12), bg="#333333", fg="#DDDDDD").pack(side=tk.LEFT)
        ttk.Scale(zoom_frame, from_=0.5, to=4.0, variable=self.zoom_level, orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT)
        zoom_frame.pack(pady=10)
        # Connect button
        tk.Button(self, text="Connect", command=self.connect, width=15, font=("Arial", 12, "bold"), bg="#4A90E2", fg="white").pack(pady=10)
        # Help text
        help_text = "IP Webcam Setup:\n1. Install IP Webcam app\n2. Start server\n3. Enter IP/Port above"
        tk.Label(self, text=help_text, justify=tk.LEFT, font=("Arial", 10, "italic"), bg="#333333", fg="#DDDDDD").pack(pady=10)

    def connect(self):
        """Pass the entered settings to the callback function and close the window."""
        ip = self.ip_entry.get()
        port = self.port_entry.get()
        zoom = self.zoom_level.get()
        self.destroy()
        self.callback(ip, port, zoom)

# Class to display processed images from a folder
class ImageSwiper:
    def __init__(self, master, images, fixed_height=None):
        """Set up a window to swipe through processed images.
        
        Args:
            master: The parent window.
            images: List of processed PIL images.
            fixed_height: Optional height to resize images to (default None).
        """
        self.master = master
        self.master.configure(bg="#333333")
        self.fixed_height = fixed_height
        self.images = self.resize_images(images) if fixed_height else images
        self.index = 0  # Current image index
        self.create_widgets()

    def resize_images(self, images):
        """Resize all images to a fixed height while keeping the aspect ratio."""
        resized = []
        for img in images:
            width, height = img.size
            new_width = int((self.fixed_height / height) * width)
            resized.append(img.resize((new_width, self.fixed_height), Image.LANCZOS))  # High-quality resizing
        return resized

    def create_widgets(self):
        """Create the image display and navigation buttons."""
        self.image_label = tk.Label(self.master, bg="#333333", borderwidth=2, relief="groove")
        self.image_label.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        # Navigation buttons
        nav_frame = tk.Frame(self.master, bg="#333333")
        tk.Button(nav_frame, text="Previous", command=self.prev, font=("Arial", 12), bg="#4A90E2", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Next", command=self.next, font=("Arial", 12), bg="#4A90E2", fg="white").pack(side=tk.RIGHT, padx=5)
        nav_frame.pack(pady=10)
        # Status label (e.g., "1/5")
        self.status = tk.Label(self.master, text=f"1/{len(self.images)}", font=("Arial", 12), bg="#333333", fg="#DDDDDD")
        self.status.pack()
        self.show_image()

    def show_image(self):
        """Display the current image in the swiper."""
        img = ImageTk.PhotoImage(self.images[self.index])  # Convert to tkinter-compatible format
        self.image_label.config(image=img)
        self.image_label.image = img  # Keep a reference to avoid garbage collection
        self.status.config(text=f"{self.index+1}/{len(self.images)}")

    def prev(self):
        """Show the previous image if available."""
        if self.index > 0:
            self.index -= 1
            self.show_image()

    def next(self):
        """Show the next image if available."""
        if self.index < len(self.images) - 1:
            self.index += 1
            self.show_image()

# Main application class that ties everything together
class MainApplication:
    def __init__(self):
        """Set up the main GUI window."""
        self.root = tk.Tk()
        self.root.title("Advanced Face Analysis System")
        self.root.geometry("600x400")
        self.root.configure(bg="#333333")  # Dark theme
        self.camera_handler = None  # Will hold the camera handler object
        self.mode = StringVar(value="slow")  # Default to slow mode
        self.create_widgets()

    def create_widgets(self):
        """Create the buttons and options in the main window."""
        main_frame = tk.Frame(self.root, bg="#333333", padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        tk.Label(main_frame, text="Face Analysis System", font=("Arial", 24, "bold"), bg="#333333", fg="#DDDDDD").pack(pady=10)

        # Mode selection radio buttons
        mode_frame = tk.Frame(main_frame, bg="#333333")
        tk.Label(mode_frame, text="Detection Mode:", font=("Arial", 12), bg="#333333", fg="#DDDDDD").pack(side=tk.LEFT)
        Radiobutton(mode_frame, text="Slow (Accurate)", variable=self.mode, value="slow", bg="#333333", fg="#DDDDDD", selectcolor="#4A90E2").pack(side=tk.LEFT)
        Radiobutton(mode_frame, text="Fast (Real-Time)", variable=self.mode, value="fast", bg="#333333", fg="#DDDDDD", selectcolor="#4A90E2").pack(side=tk.LEFT)
        mode_frame.pack(pady=10)

        # Action buttons
        buttons = [
            ("Process Folder", self.process_folder, "#4A90E2"),
            ("Local Camera", self.start_local_camera, "#4A90E2"),
            ("Mobile Camera", self.start_mobile_camera, "#4A90E2"),
            ("Instructions", self.show_instructions, "#4A90E2"),
            ("Exit", self.root.quit, "#4A90E2"),
        ]
        for text, cmd, color in buttons:
            tk.Button(main_frame, text=text, command=cmd, font=("Arial", 14), bg=color, fg="white", width=20, height=2).pack(pady=5, fill=tk.X)

    def process_folder(self):
        """Process all images in a selected folder and show results."""
        path = filedialog.askdirectory()  # Open a dialog to choose a folder
        if not path:
            return
        messagebox.showinfo("Processing", "Processing images, please wait. Initial model loading may take a moment...")
        processed = []
        for file in sorted(os.listdir(path)):
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")):  # Check for image files
                try:
                    image = cv2.imread(os.path.join(path, file))  # Load the image
                    if self.mode.get() == 'slow':
                        result = process_detection_slow(image)
                    else:
                        fast_detector = FastDetector()
                        result = fast_detector.process_detection_fast(image)
                    # Convert to PIL format for display
                    processed.append(Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)))
                except Exception as e:
                    print(f"Error processing {file}: {e}")
        if processed:
            self.show_image_swiper(processed)  # Show the processed images
        else:
            messagebox.showinfo("No Results", "No faces detected in selected folder")

    def show_image_swiper(self, images):
        """Open a window to view the processed images."""
        swiper = tk.Toplevel()
        swiper.title("Processed Images")
        swiper.geometry("800x600")
        swiper.configure(bg="#333333")
        ImageSwiper(swiper, images, fixed_height=500)

    def start_local_camera(self):
        """Start the local camera with the selected mode."""
        self.camera_handler = CameraHandler(self.mode.get())
        if not self.camera_handler.start_local_camera():
            # Offer mobile camera option if local fails
            if messagebox.askyesno("Camera Not Found", "Use mobile camera instead?"):
                self.start_mobile_camera()

    def start_mobile_camera(self):
        """Open the mobile camera setup window."""
        IPConfigWindow(self.root, self.connect_mobile_camera)

    def connect_mobile_camera(self, ip, port, zoom):
        """Connect to the mobile camera with the given settings."""
        self.camera_handler = CameraHandler(self.mode.get())
        if not self.camera_handler.start_mobile_camera(ip, port, zoom):
            messagebox.showerror("Connection Error", "Check IP/Port and try again")

    def show_instructions(self):
        """Show a help window with instructions for using the app."""
        help_window = tk.Toplevel(self.root)
        help_window.title("Instructions")
        help_window.geometry("500x400")
        help_window.configure(bg="#333333")
        instructions = """
        Face Analysis System - Instructions
        
        1. Detection Mode:
        - Slow (Accurate): Uses MTCNN and DeepFace for age, gender, and emotion detection with high accuracy, slower performance.
        - Fast (Real-Time): Uses OpenCV DNN for age and gender detection with real-time performance, slightly lower accuracy. Emotion detection via DeepFace is attempted when possible.
        
        2. Process Folder:
        - Select a folder containing images
        - View detected faces with analysis based on selected mode
        
        3. Local Camera:
        - Position your face within the outline for better detection
        - Press 'P' to start real-time analysis
        - Press 'P' again to stop analysis and return to live view
        - Press 'ESC' to exit camera view
        
        4. Mobile Camera:
        - Install IP Webcam app on your mobile device
        - Start the server in the app
        - Enter the displayed IP and port
        - Adjust zoom level as needed
        - Follow the same steps as local camera
        
        Note: For best results, ensure good lighting conditions. In fast mode, emotion detection may not always succeed due to performance constraints.
        """
        label = tk.Label(help_window, text=instructions, justify=tk.LEFT, font=("Arial", 10), bg="#333333", fg="#DDDDDD", padx=20, pady=20)
        label.pack(fill=tk.BOTH, expand=True)
        tk.Button(help_window, text="Close", command=help_window.destroy, font=("Arial", 12), bg="#4A90E2", fg="white").pack(pady=10)

# Start the application
if __name__ == "__main__":
    app = MainApplication()
    app.root.mainloop()  # Run the GUI event loop