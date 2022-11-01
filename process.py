import os
import random
import networkx as nx
import pickle
import numpy as np
import tarfile

import torch

class DataProcess(object):
    def __init__(self, path, data_name, test_rate):
        self.dir_path = f"{path}/{data_name}"
        self.graph_file = f"{self.dir_path}/graph_attr.txt"
        self.pt_path = f"{self.dir_path}/process"
        self.split_file = f"{self.dir_path}/split.pt"
        self.test_rate = test_rate

        if not os.path.exists(self.pt_path):
            os.mkdir(self.pt_path)

        self.pt_train = f"{self.pt_path}/train"
        self.pt_eval = f"{self.pt_path}/eval"
        if not os.path.exists(self.pt_train):
            os.mkdir(self.pt_train)
        if not os.path.exists(self.pt_eval):
            os.mkdir(self.pt_eval)

        # 节点数量
        self.n_node = 0
        self.offset = 0
        # 最大路由长度
        self.max_route = 0

    def _get_result(self, line):
        traffic = []
        delay = []
        jitter = []
        for src in range(self.n_node):
            for dst in range(self.n_node):
                if not src == dst:
                    traffic.append(float(line[(src * self.n_node + dst) * 3]))
                    delay.append(float(line[self.offset + (src * self.n_node + dst) * 7]))
                    jitter.append(float(line[self.offset + (src * self.n_node + dst) * 7 + 6]))
        return traffic, delay, jitter
    
    def _make_pt(self, file_list, out_path, edges, connections, link_cap):
        for file in file_list:
            pt_file = file.split('.')[0] + '.pt'
            tar = tarfile.open(os.path.join(self.dir_path, file), "r:gz")
            dir_info = tar.next()
            routing_file = tar.extractfile(dir_info.name + "/Routing.txt")
            results_file = tar.extractfile(dir_info.name + "/simulationResults.txt")
            R = np.loadtxt(routing_file, delimiter=',', dtype=str)
            R = R[:, :-1]
            R = R.astype(int)
            # MatrixPath = np.zeros((self.n_node, self.n_node, self.max_route), dtype=int)
            n_path = self.n_node * (self.n_node - 1)
            # MatrixPath = torch.zeros((n_path, self.max_route), dtype=torch.float32)
            p_lidx = torch.full((n_path*self.max_route,), fill_value=-1, dtype=torch.long)
            mask = torch.zeros((n_path, self.max_route+1), dtype=torch.float32)
            idx = 0
            # row = []
            # col = []
            # link_idx = []
            for src in range(self.n_node):
                for dst in range(self.n_node):
                    if not src == dst:
                        node = src
                        route = 0
                        mask[idx, 0] = 1.
                        while not node == dst:
                            next_node = connections[node][R[node][dst]]
                            link_idx = edges.index((node, next_node))  # 获取链路序号link_index即lidx
                            p_lidx[idx*self.max_route+route] = link_idx
                            # MatrixPath[idx, route] = edge_idx  # 获取当前链路的带宽
                            mask[idx, route+1] = 1.  # 当前mask置1
                            node = next_node
                            route += 1
                        idx += 1
            sample_list = []
            for line in results_file:
                line = line.decode().split(",")
                traffic, delay, jitter = self._get_result(line)
                bw = torch.FloatTensor(link_cap)
                tr = torch.FloatTensor(traffic)
                p_lidx = torch.LongTensor(p_lidx)
                # MatrixPath = torch.FloatTensor(MatrixPath)
                # mask = torch.FloatTensor(mask)
                delay = torch.FloatTensor(delay)
                jitter = torch.FloatTensor(jitter)
                sample = (bw, tr, p_lidx, mask, delay, jitter)
                # sample = (tr, MatrixPath, mask, delay, jitter)
                sample_list.append(sample)
            with open(os.path.join(out_path, pt_file), 'wb') as f:
                pickle.dump(sample_list, f)
            tar.close()

    def process(self):
        # 加载图信息
        G = nx.read_gml(self.graph_file, destringizer=int)
        # 节点数量
        self.n_node = G.number_of_nodes()
        self.offset = self.n_node * self.n_node * 3
        # 更新
        edges = list(map(lambda x: (x[0], x[1]), G.edges))
        n_links = len(edges)
        # # 构建拓扑链路的邻接矩阵adj
        # out_edge = {}
        # for idx, edge in enumerate(edges):
        #     src, dst = int(edge[0]), int(edge[1])
        #     if not src in out_edge.keys():
        #         out_edge[src] = [idx]
        #     else:
        #         out_edge[src].append(idx)
        # adj = torch.zeros((n_links, n_links), dtype=torch.float32)
        # for idx, edge in enumerate(edges):
        #     src, dst = int(edge[0]), int(edge[1])
        #     # 添加自环
        #     adj[idx, idx] = 1.
        #     for edge_o in out_edge[dst]:
        #         adj[idx, edge_o] = 1.
        # 链路带宽
        link_cap = []
        for e in edges:
            bandwidth = G[e[0]][e[1]][0]['bandwidth'].replace("kbps", "")
            link_cap.append(float(bandwidth))
        # 端口连接信息
        connections = {}
        for src in G:
            connections[src] = {}
            for dst in G[src].keys():
                port = G[src][dst][0]['port']
                connections[src][port] = dst
        # n_paths = n_node * (n_node -1)
        # n_links = len(edges) + 1
        # 获取数据文件名列表
        _, _, filename = next(os.walk(self.dir_path))
        tar_files = [f for f in filename if f.endswith(".tar.gz")]
        # 按照比例随机获取
        # random.seed(3407) # play a joke
        # evaling = int(len(tar_files) * self.test_rate)
        # random.shuffle(tar_files)
        # train_file, eval_file = tar_files[evaling:], tar_files[:evaling]
        # with open(self.split_file, 'wb') as f:
        #     pickle.dump((train_file, eval_file), f)
        # 加载分好的文件名列表
        with open(self.split_file, 'rb') as f:
            train_file, eval_file = pickle.load(f)

        # 获取最大路由长度
        print('# get max route path...')
        for file in tar_files:
            tar = tarfile.open(os.path.join(self.dir_path, file), "r:gz")
            dir_info = tar.next()
            routing_file = tar.extractfile(dir_info.name + "/Routing.txt")
            R = np.loadtxt(routing_file, delimiter=',', dtype=str)
            R = R[:,:-1].astype(int)
            for src in range(self.n_node):
                for dst in range(self.n_node):
                    node = src
                    route = 0
                    while not node == dst:
                        node = connections[node][R[node][dst]]
                        route += 1
                    if route > self.max_route:
                        self.max_route = route

        print('# max route path: max_len = ' + str(self.max_route))
        print('# number of links: n_link = ' + str(n_links))
        print('# process training data...')
        self._make_pt(train_file, self.pt_train, edges, connections, link_cap)
        print('# process evaling data...')
        self._make_pt(eval_file, self.pt_eval, edges, connections, link_cap)
        return (self.max_route, n_links)


if __name__ == '__main__':
    pes = DataProcess('./dataset/', 'nsfnetbw', 0.2)
    pes.process()