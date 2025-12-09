import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from transformers.modeling_outputs import BaseModelOutput

from models.bart import get_bart_with_lora


class ExpBModel(nn.Module):
    def __init__(self, num_relations=None, gnn_layers=None, dropout=0.1):
        """
        ExpB 模型 (Pure BART Baseline)

        参数说明:
            num_relations: 为了兼容 train.py 的接口保留此参数，但在 ExpB 中【不使用】。
            gnn_layers:    同上，【不使用】。
            dropout:       保留使用。
        """
        super().__init__()

        # 1. 加载 BART + LoRA
        # 这一步和 ExpA 完全一致，复用预训练参数
        print(">>> Initializing BART with LoRA (Node-only Mode)...")
        self.tokenizer, self.bart = get_bart_with_lora(lora_rank=8)

        # 获取隐藏层维度
        self.d_model = self.bart.config.d_model

        # --- [关键区别] ---
        # 这里没有 relation_embedding
        # 这里没有 gnn_layers
        # 这里没有 projection (为了保持最纯粹的 BART 语义透传)

        self.dropout = nn.Dropout(dropout)

    def encode_nodes(self, input_ids, attention_mask):
        """
        Step 1: 文本 -> BART Encoder -> 节点初始向量
        (代码逻辑与 ExpA 完全一致)
        """
        encoder = self.bart.get_encoder()

        outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 取 [CLS] 作为节点特征
        node_embeddings = outputs.last_hidden_state[:, 0, :]
        return node_embeddings

    def forward(self, data):
        """
        训练时的前向传播
        注意：这里完全忽略 data.edge_index 和 data.edge_attr
        """
        # --- A. 节点向量化 ---
        x = self.encode_nodes(data.x, data.node_mask)

        # ExpB 特有逻辑：直接 Dropout，跳过 GNN
        x = self.dropout(x)

        # --- B. 重构为 Batch ---
        # 此时 x 是纯文本语义特征
        encoder_hidden_states, encoder_attention_mask = to_dense_batch(x, data.batch)

        # --- C. 解码与 Loss 计算 ---
        # 包装为 HF 对象 (防止报错)
        encoder_outputs_obj = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        outputs = self.bart(
            encoder_outputs=encoder_outputs_obj,
            attention_mask=encoder_attention_mask,
            labels=data.y
        )

        return outputs.loss

    def generate(self, data, num_beams=4):
        """
        推理时的生成方法
        """
        # A. Encode
        x = self.encode_nodes(data.x, data.node_mask)

        # ExpB: Skip GNN

        # B. Reshape
        encoder_hidden_states, encoder_attention_mask = to_dense_batch(x, data.batch)

        # C. Generate
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
