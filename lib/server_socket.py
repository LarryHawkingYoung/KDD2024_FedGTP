import pickle, struct
import socket
import argparse
import time

class ServerSocket():
    def __init__(self, n_clients, port, ip='0.0.0.0'):
        self.ip = ip
        self.port = port
        self.n_clients = n_clients
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.server_socket.settimeout(9999)
        self.server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.server_socket.bind((ip, port))
        self.server_socket.listen()
        self.log_info("server listening")
        self.connections = {}
        for _ in range(self.n_clients):
            conn, addr = self.server_socket.accept()
            self.log_info(f"accept connection from {addr}")
            cid = self.recv(conn)
            self.log_info(f"received msg from client {cid}")
            self.connections[cid] = conn
            self.send("ACK", conn)

    def send(self, msg, conn=None):
        msg = pickle.dumps(msg)
        data_len = len(msg)
        header = struct.pack('i', data_len)
        if conn == None:
            for i in range(1, self.n_clients+1):
                conn = self.connections[i]
                conn.send(header)
                conn.send(msg)
            return data_len
        else:
            if type(conn) == int: conn = self.connections[conn]
            conn.send(header)
            conn.send(msg)
            return data_len
    
    def recv(self, conn=None):
        if conn == None:
            rcvd_msgs = []
            for i in range(1, self.n_clients+1):
                conn = self.connections[i]
                while True:
                    data_len = conn.recv(4, socket.MSG_WAITALL)
                    # data_len = conn.recv(4)
                    if data_len != None and len(data_len)==4:
                        self.log_info(f"data_len: {data_len}")
                        break
                    else: time.sleep(0.01)
                data_len = struct.unpack('i', data_len)[0]
                self.log_info(f"data_len: {data_len}")
                recv_data = conn.recv(data_len, socket.MSG_WAITALL)
                # recv_data = conn.recv(data_len)
                rcvd_msgs.append(pickle.loads(recv_data))
            assert len(rcvd_msgs) == self.n_clients
            return rcvd_msgs
        else:
            if type(conn) == int: conn = self.connections[conn]
            data_len = conn.recv(4, socket.MSG_WAITALL)
            # data_len = conn.recv(4)
            if not data_len:
                self.log_info(f"msg from client {i} not received!")
                return None
            data_len = struct.unpack('i', data_len)[0]
            recv_data = conn.recv(data_len, socket.MSG_WAITALL)
            # recv_data = conn.recv(data_len)
            recv_data = pickle.loads(recv_data)
            return recv_data

    def log_info(self, info, on=False):
        if on: print(f"[SERVER] {info}")
    
    def close(self):
        self.server_socket.close()
        for i in range(1, self.n_clients+1):
            self.connections[i].close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', dest='n')
    parser.add_argument('-p', dest='port')
    parser.add_argument('-i', dest='ip')
    args = parser.parse_args()

    server = ServerSocket(n_clients=int(args.n), port=int(args.port), ip=args.ip)