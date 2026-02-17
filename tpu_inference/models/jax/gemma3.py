from typing import List, Tuple

import jax 
from flax import nnx 

from vllm.config import VllmConfig
from tpu_inference.layers.common.attention_metadata import AttentionMetadata


class Gemma3ForCausalLM(nnx.Module):
    def __init__(
        self, 
        vllm_config: VllmConfig, 
        rng: nnx.Rngs, mesh: 
        jax.sharding.Mesh
    ):
        pass 
    
    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        *args,   
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array]]:
        pass 
    
    def load_weights(self, rng_key: jax.Array): 
        pass 
    
    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        pass 
    