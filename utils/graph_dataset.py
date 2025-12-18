import torch
import json
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from collections import deque, defaultdict # 引入 deque 做 BFS


class ExpDataset(Dataset):
    def __init__(self, hf_dataset, relation_vocab_path, tokenizer, max_node_len=32, max_tgt_len=64):
        """
        参数:
            hf_dataset: HuggingFace Dataset 对象 (Arrow格式，已经 load_from_disk 好的)
            relation_vocab_path: 关系字典(relations.json)的路径
            tokenizer: BART Tokenizer
        """
        # 1. 持有数据引用 (Arrow 格式，高效内存映射)
        # 程序只认这个，不看那些 debug 用的 json
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_node_len = max_node_len
        self.max_tgt_len = max_tgt_len

        # 2. 加载关系映射表 (这是唯一读取的 JSON)
        # 作用: 把 "spouse" 变成 0
        with open(relation_vocab_path, 'r', encoding='utf-8') as f:
            self.rel2id = json.load(f)

    def __len__(self):
        """返回数据集样本总数"""
        return len(self.data)

    def _calculate_structure_info(self, nodes_text, edge_index_list):
        """
        [新增] 动态计算节点类型和跳数距离
        输入:
            nodes_text: list of str, e.g., ["[TOPIC] Obama", "Hawaii", ...]
            edge_index_list: list of list, e.g., [[0, 1], [1, 2]] (注意这里输入原始列表，非Tensor)
        输出:
            node_types: list of int
            hops: list of int
        """
        num_nodes = len(nodes_text)

        # --- 1. 确定 Node Types ---
        # 0: Topic (Start), 1: Ans (End), 2: Normal/Noise
        node_types = [2] * num_nodes
        start_node_idx = -1

        for i, text in enumerate(nodes_text):
            if "[TOPIC]" in text:
                node_types[i] = 0
                start_node_idx = i
            elif "[ANS]" in text:
                node_types[i] = 1

        # --- 2. 确定 Hop Distance (BFS) ---
        # 初始化所有距离为 9 (代表不可达或很远，embedding时会截断)
        hops = [9] * num_nodes

        if start_node_idx != -1:
            # 构建邻接表 (Adjacency List) 用于 BFS
            adj = defaultdict(list)
            # edge_index_list[0] 是 source, edge_index_list[1] 是 target
            src_list = edge_index_list[0]
            tgt_list = edge_index_list[1]

            for u, v in zip(src_list, tgt_list):
                adj[u].append(v)

            # 开始 BFS
            queue = deque([(start_node_idx, 0)])  # (current_node, current_dist)
            hops[start_node_idx] = 0
            visited = {start_node_idx}

            while queue:
                curr, dist = queue.popleft()

                # 遍历邻居
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        hops[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

        return node_types, hops

    def __getitem__(self, idx):
        """
        核心方法：只在需要时（DataLoader取数据时）才被调用。
        返回: 一个 PyG Data 对象
        """
        # A. 从 Arrow 中取出一条原始数据
        # 格式: {'nodes': [...], 'edge_index': [[...],[...]], 'edge_attr': [...], 'question': "..."}
        item = self.data[idx]

        # B. 处理节点 (Nodes) -> Token IDs
        # 输入是字符串列表: ["[TOPIC] Obama", "Hawaii", ...]
        node_encodings = self.tokenizer(
            item['nodes'],
            padding='max_length',  # 强制填充到 32
            truncation=True,  # 超长截断
            max_length=self.max_node_len,
            return_tensors='pt',  # 返回 PyTorch Tensor
            add_special_tokens=True
        )
        x_ids = node_encodings['input_ids']  # Shape: [num_nodes, 32]
        x_mask = node_encodings['attention_mask']  # Shape: [num_nodes, 32]

        # C. 处理边 (Edges) -> Relation IDs
        # 边索引直接转 Tensor
        edge_index = torch.tensor(item['edge_index'], dtype=torch.long)  # Shape: [2, num_edges]

        # 边属性查表: "spouse" -> 0
        # 如果遇到字典里没有的关系(极少见)，默认给 0，防止报错
        rel_ids = [self.rel2id.get(r, 0) for r in item['edge_attr']]
        edge_attr = torch.tensor(rel_ids, dtype=torch.long)  # Shape: [num_edges]

        # D. 处理目标问题 (Target Question) -> Label IDs
        target_encoding = self.tokenizer(
            item['question'],
            padding='max_length',
            truncation=True,
            max_length=self.max_tgt_len,
            return_tensors='pt'
        )
        labels = target_encoding['input_ids'].view(1, -1)

        # 将 Padding 部分的 ID (通常是1) 设为 -100
        # 这样计算 Loss 时会自动忽略这些位置
        labels[labels == self.tokenizer.pad_token_id] = -100

        # E. [新增] 结构信息计算 ---
        # 注意：这里我们传入 item['edge_index'] (原始列表)，而不是转成 Tensor 后的
        node_types_list, hops_list = self._calculate_structure_info(item['nodes'], item['edge_index'])

        node_types = torch.tensor(node_types_list, dtype=torch.long)
        hops = torch.tensor(hops_list, dtype=torch.long)

        # F. 组装成 PyG Data
        data = Data(
            x=x_ids,  # 节点 Token IDs
            node_mask=x_mask,  # 节点 Mask
            edge_index=edge_index,  # 拓扑结构
            edge_attr=edge_attr,  # 关系 IDs
            y=labels,   # 目标问题 IDs
            # [新增字段]
            node_type=node_types,   # [num_nodes]
            hop_id=hops             # [num_nodes]
        )

        return data


class ExpCollator:
    """
    DataLoader 的胶水函数。
    作用: 把 DataLoader 取出的 [Data1, Data2, Data3, Data4] 拼成一个 Batch 对象。
    """

    def __call__(self, batch_list):
        # PyG 的 Batch.from_data_list 会自动处理图的拼接
        # 例如: 自动修正 edge_index 的偏移量
        return Batch.from_data_list(batch_list)