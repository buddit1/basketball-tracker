import cv2 as cv
import socket
import struct
import numpy as np
import argparse

import models


def draw_trajectory(frame, points):
    for point in points[:,0:2]:
        cv.circle(frame, point, radius=5, color=(248, 90, 252), thickness=-3)
    return



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
    points = np.empty((0, 3), dtype=np.int32) #store detection locations as (x, y, age)

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

            ball_centers = model.find_ball_centers(frame, imgsz=frame.shape[0:2])

            if ball_centers is not None:
                frames_w_ball += 1
                ball_centers = np.concatenate([ball_centers, np.zeros((ball_centers.shape[0], 1), dtype=np.int32)], axis=-1)
                points = np.concatenate([points, ball_centers])
            points = points[points[:,-1] < args.trajectory_length]
            points[:, -1] += 1
            if len(points) > 0:
                draw_trajectory(frame, points)

            cv.imshow('Receiving...', frame)
            if cv.waitKey(24) & 0xFF == ord('q'):
                break
        client_socket.close()
        break

    server_socket.close()
    cv.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="backend to track basketballs in videos.")
    parser.add_argument('-ip', type=str, help='ip address to host service on.')
    parser.add_argument('-port', type=int, help='port to listen for incoming connections on.')
    parser.add_argument('-model', type=str, help='yolo variant to use for tracking.',
                        choices=['yolov9c', 'yolov9e', 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
                        )
    parser.add_argument('--trajectory-length', type=int, help="""How long to keep displaying points displayed after initial
                        detection. Default 10""", default=10)
    args = parser.parse_args()
    main(args)