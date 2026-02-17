from typing import List, Tuple

import jax 
import jax.numpy as jnp 
from flax import nnx 

from vllm.config import VllmConfig
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.models.jax.jax_intermediate_tensor import JaxIntermediateTensors

class RMSNorm(nnx.Module): 
    def __init__(self, dim: int, eps: float = 1e-6): 
        self.eps = eps 
        self.weight = nnx.Param(jnp.ones(dim))
    
    def __call__(self, x: jax.Array) -> jax.Array: 
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps) 
        return self.weight * (x / rms)

class Gemma3Model(nnx.Module): 
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
        intermediate_tensors: JaxIntermediateTensors | None,
    ): 
        pass 

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


# Playground :D
if __name__ == '__main__': 
    norm = RmsNorm(hidden_dim=4)
    arr = jnp.array([10, 1, -5, 2])
    print(arr)
    arr_norm = norm(arr)
    print(arr_norm)