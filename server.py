from lib.server_socket import ServerSocket
import argparse
import time
import copy
import torch
import collections

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

class Server():
    def __init__(self, n_clients, port, ip):
        self.socket = ServerSocket(n_clients, port, ip)
        while True:
            rcvd_msgs = self.socket.recv()
            if rcvd_msgs:
                if type(rcvd_msgs[0])==collections.OrderedDict or type(rcvd_msgs[0])==dict:
                    self.socket.send(FedAvg(rcvd_msgs))
                else:
                    self.socket.send(sum(rcvd_msgs))
            else:
                print("[SERVER RECVED NONE]")
                self.socket.close()
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', dest='n')
    parser.add_argument('-p', dest='port')
    parser.add_argument('-i', dest='ip')
    args = parser.parse_args()

    server = Server(n_clients=int(args.n), port=int(args.port), ip=args.ip)