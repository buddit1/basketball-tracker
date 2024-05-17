import cv2 as cv
import socket
import struct
import numpy as np
import argparse






def main(args):
    # Set up socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_ip = args.ip
    port = args.port
    socket_address = (host_ip, port)
    server_socket.bind(socket_address)
    server_socket.listen(5)

    print("Listening at", socket_address)

    # Accept a client connection
    while True:
        client_socket, addr = server_socket.accept()
        print('Connection from:', addr)
        data = b""
        payload_size = struct.calcsize("Q")
        while True:
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
            img = cv.imdecode(jpg_as_np, flags=1)
            cv.imshow('Receiving...', img)
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
    args = parser.parse_args()
    main(args)