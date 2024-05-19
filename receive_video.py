import cv2 as cv
import socket
import struct
import numpy as np
import argparse
import scipy
import scipy.signal
from skimage import restoration
import skimage


import models
import time



def richardson_lucy_denoise(frame, psf, iters):
    frame = skimage.util.img_as_float(frame)
    frame = restoration.richardson_lucy(frame, psf, num_iter=iters)
    frame = np.uint8(np.clip(frame*255, 0, 255))
    return frame


def draw_trajectory(frame, points):
    interiors = np.zeros_like(frame, np.uint8)
    perimeters = np.zeros_like(frame, np.uint8)
    for point in points:
        if point[-1] == 0:
            cv.circle(perimeters, point[0:2], radius=point[2], color=(248, 90, 252), thickness=2)
        cv.circle(interiors, point[0:2], radius=point[2], color=(248, 90, 252), thickness=-3)
    interiors_alpha = 0.7
    perimeters_alpha = 0.2
    interiors_mask = interiors.astype(bool)
    perimeters_mask = perimeters.astype(bool)
    frame[interiors_mask] = cv.addWeighted(frame, interiors_alpha, interiors, 1 - interiors_alpha, 0)[interiors_mask]
    frame[perimeters_mask] = cv.addWeighted(frame, perimeters_alpha, perimeters, 1 - perimeters_alpha, 0)[perimeters_mask]
    return frame



def main(args):
    # Set up socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_ip = args.ip
    port = args.port
    socket_address = (host_ip, port)
    server_socket.bind(socket_address)
    server_socket.listen(5)

    model = models.YOLOBasketballBB(args.model, device='cuda:0')

    if args.denoise_method == 'richardson-lucy':
        psf = np.ones((args.denoise_kernel_size, args.denoise_kernel_size, 1)) / args.denoise_kernel_size**2


    print("Listening at", socket_address)
    points = np.empty((0, 4), dtype=np.int32) #store detection locations as (x, y, radius, age)

    frames_w_ball = 0
    # Accept a client connection
    while True:
        client_socket, addr = server_socket.accept()
        print('Connection from:', addr)
        data = b""
        payload_size = struct.calcsize("Q")

        start = time.time()
        n_frames = 0
        model_warm = False
        while True:
            #warm up model for input shape before receiving real data
            if not model_warm:
                packet = client_socket.recv(1024)
                if not packet: 
                    break
                #get shape from client program to warm up model
                shape_size = struct.unpack("Q", packet[:payload_size])[0]
                shape_data = packet[payload_size:payload_size + shape_size]
                frame_shape = np.frombuffer(shape_data, dtype=np.int32)
                fake_data = np.random.rand(*frame_shape, 3)
                model.predict(fake_data, imgsz=frame_shape.tolist())
                #send data back to client to say that model is warmed up
                client_socket.sendall(b'1')
                model_warm = True
                if args.output is not None:
                    writer = cv.VideoWriter(args.output, cv.VideoWriter_fourcc(*'MP4V'), 30, frame_shape)
                start = time.time()

            #receive real data
            while len(data) < payload_size:
                packet = client_socket.recv(4*1024)  # 4K
                if not packet: break
                data += packet
            if not data: break
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]
            while len(data) < msg_size:
                data += client_socket.recv(4*1024)
            frame_data = data[:msg_size]
            
            data = data[msg_size:]

            # Deserialize the frame
            jpg_as_np = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv.imdecode(jpg_as_np, flags=cv.IMREAD_COLOR)

            if args.denoise_method == "richardson-lucy":
                richardson_lucy_denoise(frame, psf, args.richardson_lucy_iters)
            if args.denoise_method == "weiner":
                frame = scipy.signal.wiener(frame, (args.denoise_kernel_size, args.denoise_kernel_size, 1), args.weiner_noise)
                frame = np.uint8(np.clip(frame, 0, 255))

            ball_centers = model.find_balls(frame, imgsz=frame.shape[0:2])

            if ball_centers is not None:
                frames_w_ball += 1
                ball_centers = np.concatenate([ball_centers, np.zeros((ball_centers.shape[0], 1), dtype=np.int32)], axis=-1)
                points = np.concatenate([points, ball_centers])
            if len(points) > 0:
                draw_trajectory(frame, points)
            points = points[points[:,-1] < args.trajectory_length]
            points[:, -1] += 1

            if args.output is not None:
                writer.write(frame)
            if args.no_display != True:
                cv.imshow('Receiving...', frame)
                if cv.waitKey(10) & 0xFF == ord('q'):
                    break

            n_frames += 1

        end = time.time()
        print(f"processed {n_frames} frames in {end - start:.2f} seconds")
        client_socket.close()
        break

    server_socket.close()
    cv.destroyAllWindows()
    print(f"detected ball in {frames_w_ball} out of {n_frames} frames")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="backend to track basketballs in videos.")
    parser.add_argument('-ip', type=str, help='ip address to host service on.')
    parser.add_argument('-port', type=int, help='port to listen for incoming connections on.')
    parser.add_argument('-model', type=str, help='yolo variant to use for tracking.',
                        choices=['yolov9c', 'yolov9e', 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
                        )
    parser.add_argument('--trajectory-length', type=int, help="""How long to keep displaying points displayed after initial
                        detection. Default 10""", default=10)
    parser.add_argument('--denoise-method', type=str, help='method used to denoise image. Default None',
                        choices=['None', 'weiner', 'richardson-lucy'])
    parser.add_argument('--richardson-lucy-iters', type=int, help='number of iterations used for Richardson-Lucy deconvolution. Default 2.',
                        default=2
                        )
    parser.add_argument('--denoise-kernel-size', type=int, help='kernel size for denoising filter. Default 5.', default=5)
    parser.add_argument('--weiner-noise', type=float, help="Noise parameter for Weiner filtering. Should be between 0 and 1. Default 0.1",
                        default=0.1)
    parser.add_argument("--output", type=str, help="output file destination. Currently only tested with .mp4")
    parser.add_argument("--no-display", action='store_true', help="Pass this flag to prevent displaying video.")
    args = parser.parse_args()
    main(args)