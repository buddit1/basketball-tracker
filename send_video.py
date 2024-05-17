import cv2 as cv
import numpy as np
import socket
import struct
import argparse
import os



def calculate_frame_resized_shape(frame):
    """
    Calculate shape to resize frame to in order to pass to YOLO model.
    YOLO model expects larger dimension to have size 512 and smaller 
    dimension to be a multiple of 32 that approximately preserves the 
    aspect ratio of the original image.
    """
    max_dim_idx = np.argmax(frame.shape)
    min_dim_idx = 0 if max_dim_idx == 1 else 1
    rescale_ratio = 512 / frame.shape[max_dim_idx]
    new_min_dim = int(frame.shape[min_dim_idx] * rescale_ratio)
    if new_min_dim % 32 <= 16:
        new_min_dim = new_min_dim - new_min_dim % 32
    else:
        new_min_dim = new_min_dim + 32 - (new_min_dim % 32)
    new_shape = (512, new_min_dim) if max_dim_idx == 1 else (new_min_dim, 512)
    return new_shape


def main(args):
    # Set up socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_ip = args.backend_ip
    port = args.port
    client_socket.connect((host_ip, port))

    #check that input file exists if passed
    if args.file != None:
        if not os.path.isfile(args.file):
            raise FileNotFoundError(f"The file {args.file} does not exist.")
    
    #setup input video stream
    video_source = args.file if args.file is not None else 0
    cap = cv.VideoCapture(video_source)
    if not cap.isOpened():
        raise OSError(f"Could not open video source: {video_source}")
    
    new_shape = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if new_shape == None:
            new_shape = calculate_frame_resized_shape(frame)
        frame = cv.resize(frame, new_shape)

        # Serialize the frame
        success, image_bytes = cv.imencode('.jpg', frame)
        if not success:
            raise RuntimeError("Failed to encode frame to jpg format")
        image_bytes = image_bytes.tobytes()
        message_size = struct.pack("Q", len(image_bytes))
        # Send the packed size and serialized frame
        client_socket.sendall(message_size + image_bytes)

        cv.imshow('Sending...', frame)
        if cv.waitKey(24) & 0xFF == ord('q'):
            break

    cap.release()
    client_socket.close()
    cv.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="client side application to stream video to backend.")
    parser.add_argument("-backend-ip", type=str, help='IP adress for backend service.')
    parser.add_argument("-port", type=int, help='backend port to send data to.')
    parser.add_argument("--file", type=str, help='optional argument to read video from a file instead of webcam.')
    args = parser.parse_args()
    main(args)
