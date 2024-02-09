import pickle, struct, socket
import argparse, time

class ClientSocket():
    def __init__(self, client_id, server_port, self_port, server_ip='127.0.0.1', self_ip='127.0.0.1'):
        self.client_id = client_id
        self.server_port = server_port
        self.self_port = self_port
        self.server_ip = server_ip
        self.self_ip = self_ip

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.socket.settimeout(9999)
        # self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # self.socket.bind((self.self_ip, self.self_port))
        self.log_info(f"start connect")
        self.socket.connect((self.server_ip, self.server_port))
        self.log_info(f"send msg")
        self.send(client_id)
        self.log_info(self.recv())
    
    def send(self, msg):
        msg = pickle.dumps(msg)
        data_len = len(msg)
        header = struct.pack('i', data_len)
        self.socket.send(header)
        self.socket.send(msg)
        return data_len

    def recv(self):
        while True:
            data_len = self.socket.recv(4, socket.MSG_WAITALL)
            # data_len = self.socket.recv(4)
            if data_len != None and len(data_len)==4:
                self.log_info(f"data_len: {data_len}")
                break
            else: time.sleep(0.01)
        data_len = struct.unpack('i', data_len)[0]
        self.log_info(f"data_len: {data_len}")
        recv_data = self.socket.recv(data_len, socket.MSG_WAITALL)
        # recv_data = self.socket.recv(data_len)
        recv_data = pickle.loads(recv_data)
        return recv_data
    
    def log_info(self, info, on=False):
        if on: print(f"[CLIENT_{self.client_id}] {info}")
    
    def close(self):
        self.socket.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', dest='cid')
    parser.add_argument('-sp', dest='server_port')
    parser.add_argument('-cp', dest='self_port')
    args = parser.parse_args()

    client = ClientSocket(int(args.cid), int(args.server_port), int(args.self_port))