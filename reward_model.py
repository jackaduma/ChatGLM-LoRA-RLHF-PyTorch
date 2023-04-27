#!python
# -*- coding: utf-8 -*-
# @author: Kun


import torch
import torch.nn as nn

from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.tokenization_utils import PreTrainedTokenizer

"""
#! implement Reward Model for SeqClassification because GLM not support SeqCLS task in transformers
"""

##################################################################################
class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """

    def forward(self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(chosen_reward - reject_reward)
        log_probs = torch.log(probs)
        loss = -log_probs.mean()
        return loss

##################################################################################

class RewardModel(PreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, config, model, tokenizer):
        super().__init__(config)
        self.model_type = config.model_type
        print("model_type: ", self.model_type)
        self.pad_id = tokenizer.pad_token_id
        self.transformer = model
        self.v_head = nn.Linear(config.hidden_size, 1, bias=False, dtype=torch.float16)
        self.loss_fn = PairWiseLoss()

    def gradient_checkpointing_enable(self):
        self.transformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.transformer.gradient_checkpointing_disable()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, PreTrainedModel):
            module.gradient_checkpointing = value

    def reward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None
    ):
        batch_size = input_ids.shape[0]
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        if self.model_type == "glm":
            hidden_states = transformer_outputs.mems[-1]

        elif self.model_type == "chatglm":
            hidden_states = transformer_outputs[0]
            seq_len, batch_size, hidden_size = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, seq_len, hidden_size)

        elif self.model_type == "pangu":
            hidden_states = transformer_outputs[0]
            hidden_states = hidden_states.squeeze(1)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        assert len(hidden_states.shape) == 3

        # print("hidden_states: ", type(hidden_states), hidden_states.dtype)
        rewards = self.v_head(hidden_states).squeeze(-1)

        rewards = rewards.mean(dim=-1)
        if len(rewards.shape) == 2:
            rewards = rewards.squeeze(1)    # ensure shape is (B)

        assert len(rewards.shape) == 1 and rewards.shape[0] == batch_size

        return rewards

    def forward(
            self,
            chosen_input_ids=None,
            chosen_attention_mask=None,
            chosen_position_ids=None,
            rejected_input_ids=None,
            rejected_attention_mask=None,
            rejected_position_ids=None,
            past_key_values=None,
            token_type_ids=None,
            head_mask=None,
            inputs_embeds=None,
            mc_token_ids=None,
            labels=None,
            return_dict=False,
            output_attentions=False,
            output_hidden_states=False,
    ):
        if chosen_input_ids is not None:
            chosen_reward = self.reward(chosen_input_ids, attention_mask=chosen_attention_mask, position_ids=chosen_position_ids)
            # print("chosen_reward: ", chosen_reward.shape)
        else:
            chosen_reward = None

        if rejected_input_ids is not None:
            reject_reward = self.reward(rejected_input_ids, attention_mask=rejected_attention_mask, position_ids=rejected_position_ids)
            # print("reject_reward: ", reject_reward.shape)
        else:
            reject_reward = None

        if chosen_reward is not None and reject_reward is not None:
            loss = self.loss_fn(chosen_reward, reject_reward)   
        else:
            loss = None

        return {
            "loss": loss,
            "chosen_reward": torch.sigmoid(chosen_reward) if chosen_reward is not None else chosen_reward,
            "reject_reward": torch.sigmoid(reject_reward) if reject_reward is not None else reject_reward,
        }
