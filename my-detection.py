import jetson.inference
import jetson.utils

# Initialize the SSD-MobileNet v2 model with a detection threshold
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# Load the image from the specified path
image_path = "/home/nvidia/jetson-inference/python/1.jpg"
img = jetson.utils.loadImage(image_path)  # This function loads the image into GPU memory

# Perform object detection on the loaded image
detections = net.Detect(img)

# Set up the display output to show the result on a monitor
display = jetson.utils.videoOutput("display://0")

# Check if we have a valid image and detections to display
if img is not None and detections is not None:
    # Render the image with detections
    display.Render(img)

    # Display status information: object detection and network FPS
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

    # Print detections to the console
    for detection in detections:
        print(f"ClassID: {detection.ClassID}")
        print(f"Confidence: {detection.Confidence}")
        print(f"Left: {detection.Left}")
        print(f"Top: {detection.Top}")
        print(f"Right: {detection.Right}")
        print(f"Bottom: {detection.Bottom}")
        print(f"Width: {detection.Width}")
        print(f"Height: {detection.Height}")
        print(f"Area: {detection.Area}")
        print(f"Center: ({detection.Center[0]}, {detection.Center[1]})")

# Close the display after rendering
display.Close()





