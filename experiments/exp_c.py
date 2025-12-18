import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_batch
from transformers.modeling_outputs import BaseModelOutput

# 复用基础组件
from models.bart import get_bart_with_lora


class ExpCModel(nn.Module):
    def __init__(self, num_relations, gnn_layers=3, dropout=0.1):
        """
        ExpC: 融合增强模型 (Fusion Strategy)
        核心思想: Fused = α * GNN_Feature + (1-α) * BART_Feature
        """
        super().__init__()

        # 1. 加载 BART + LoRA (Shared Base)
        # 这部分是 ExpA 和 ExpB 共有的基础
        print(">>> Initializing BART with LoRA (Fusion Mode)...")
        self.tokenizer, self.bart = get_bart_with_lora(lora_rank=8)
        self.d_model = self.bart.config.d_model

        # --- 组件 A: 图逻辑流 (来自 ExpA) ---
        self.relation_embedding = nn.Embedding(num_relations, self.d_model)
        self.gnn_layers = nn.ModuleList()
        for _ in range(gnn_layers):
            self.gnn_layers.append(
                GATv2Conv(
                    self.d_model,
                    self.d_model,
                    heads=4,
                    concat=False,
                    dropout=dropout,
                    edge_dim=self.d_model
                )
            )

        # --- 组件 B: 纯语义流 (来自 ExpB) ---
        # ExpB 不需要额外的参数，它直接用 BART Encoder 的输出

        # --- 组件 C: 门控融合层 (The Innovation) ---
        # 输入是 [Text; Graph]，维度是 2 * d_model
        # 输出是 1 个标量 (alpha)
        self.gate_net = nn.Sequential(
            nn.Linear(self.d_model * 2, 1),
            nn.Sigmoid()  # 保证输出在 0~1 之间
        )

        # --- 组件 D: 最终适配层 (Projection) ---
        # 融合后的特征再次整理，准备喂给 Decoder
        self.projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU()
        )

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # 【新增】专门用于 WandB 记录的缓存字典
        self.alpha_stats = {}

    def encode_nodes(self, input_ids, attention_mask):
        """
        获取节点初始特征 (ExpA 和 ExpB 通用)
        """
        encoder = self.bart.get_encoder()
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

    def forward(self, data):
        # 1. 获取纯语义特征 (这是 ExpB 的全部)
        # x_text: [total_nodes, 768]
        x_text = self.encode_nodes(data.x, data.node_mask)

        # 2. 获取图逻辑特征 (这是 ExpA 的核心)
        # 我们 clone 一份 x_text 作为 GNN 的输入，以免梯度混乱
        x_graph = x_text.clone()
        edge_attr_emb = self.relation_embedding(data.edge_attr)

        for gnn in self.gnn_layers:
            residual = x_graph
            x_graph = gnn(x_graph, data.edge_index, edge_attr=edge_attr_emb)
            x_graph = self.activation(x_graph)
            x_graph = self.dropout(x_graph) + residual

        # 3. 【核心创新】自适应门控融合
        # 拼接: [N, 768] + [N, 768] -> [N, 1536]
        combined = torch.cat([x_text, x_graph], dim=-1)

        # 计算 Alpha: [N, 1]
        # alpha 大，说明模型觉得“看图”更重要
        # alpha 小，说明模型觉得“看字”更重要
        alpha = self.gate_net(combined)

        # 【核心修改】计算统计量并暴露给外部
        # 我们只在训练时记录，或者你也可以去掉 if self.training 全程记录
        if self.training:
            with torch.no_grad():
                self.alpha_stats = {
                    "alpha/mean": alpha.mean().item(),
                    "alpha/std": alpha.std().item(),  # 标准差 (看是不是所有节点都一样)
                    "alpha/min": alpha.min().item(),
                    "alpha/max": alpha.max().item()
                }

        # 加权求和
        x_fused = alpha * x_graph + (1 - alpha) * x_text

        # 4. 最终适配与重构
        x_final = self.projection(x_fused)

        encoder_hidden_states, encoder_attention_mask = to_dense_batch(x_final, data.batch)

        # 5. 解码
        encoder_outputs_obj = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        outputs = self.bart(
            encoder_outputs=encoder_outputs_obj,
            attention_mask=encoder_attention_mask,
            labels=data.y
        )

        return outputs.loss

    def generate(self, data, num_beams=4):
        # 1. 语义流
        x_text = self.encode_nodes(data.x, data.node_mask)

        # 2. 逻辑流
        x_graph = x_text.clone()
        edge_attr_emb = self.relation_embedding(data.edge_attr)
        for gnn in self.gnn_layers:
            x_graph = x_graph + self.dropout(self.activation(gnn(x_graph, data.edge_index, edge_attr=edge_attr_emb)))

        # 3. 融合
        combined = torch.cat([x_text, x_graph], dim=-1)
        alpha = self.gate_net(combined)
        x_fused = alpha * x_graph + (1 - alpha) * x_text

        # 4. 适配
        x_final = self.projection(x_fused)

        # 5. 生成
        encoder_hidden_states, encoder_attention_mask = to_dense_batch(x_final, data.batch)
        encoder_outputs_obj = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        generated_ids = self.bart.generate(
            encoder_outputs=encoder_outputs_obj,
            attention_mask=encoder_attention_mask,
            max_length=64,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

        return generated_ids
