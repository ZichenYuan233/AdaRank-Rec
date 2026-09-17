"""Frozen Qwen/Hugging Face decoder bridge for four-token SID prediction."""
from __future__ import annotations
import torch
from torch import Tensor, nn


class FrozenHFDecoder(nn.Module):
    """Use fixed context embeddings with any Hugging Face causal LM.

    The base model remains frozen. SID input embeddings and the four SID output
    heads are new trainable parameters, matching the paper's added SID input /
    output parameters without altering Qwen's ordinary vocabulary.
    """
    def __init__(self, model: nn.Module, prompt_ids: Tensor, codes_per_level: int = 256):
        super().__init__()
        self.model = model
        self.register_buffer("prompt_ids", prompt_ids.long())
        self.codes_per_level = codes_per_level
        hidden = model.get_input_embeddings().embedding_dim
        self.sid_input = nn.Embedding(4 * codes_per_level, hidden)
        self.sid_output = nn.Parameter(torch.empty(4, codes_per_level, hidden))
        self.sid_bias = nn.Parameter(torch.zeros(4, codes_per_level))
        nn.init.normal_(self.sid_input.weight, std=.02)
        nn.init.normal_(self.sid_output, std=.02)
        for parameter in self.model.parameters(): parameter.requires_grad_(False)

    def forward(self, context: Tensor, sid_targets: Tensor | None = None) -> Tensor:
        if sid_targets is None:
            raise ValueError("teacher-forced SID targets are required for causal decoder scoring")
        bsz = context.shape[0]
        embeddings = self.model.get_input_embeddings()
        prompt = embeddings(self.prompt_ids)[None].expand(bsz, -1, -1)
        parts = [context, prompt]
        # Feed the first 3 SID tokens; the preceding position predicts each target.
        parts.append(self.sid_input(sid_targets[:, :3]))
        input_embeds = torch.cat(parts, 1)
        output = self.model(inputs_embeds=input_embeds, use_cache=False, output_hidden_states=True)
        hidden = output.hidden_states[-1][:, -4:, :]
        local_logits = torch.einsum("bph,pvh->bpv", hidden, self.sid_output) + self.sid_bias
        # Preserve the global 1,024-token SID indexing used elsewhere.
        result = hidden.new_full((bsz, 4, 4 * self.codes_per_level), torch.finfo(hidden.dtype).min)
        for position in range(4):
            start = position * self.codes_per_level
            result[:, position, start:start+self.codes_per_level] = local_logits[:, position]
        return result
