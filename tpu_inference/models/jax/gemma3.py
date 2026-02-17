from typing import List, Tuple

import jax 
import jax.numpy as jnp 
from flax import nnx 

from vllm.config import VllmConfig
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.models.jax.jax_intermediate_tensor import JaxIntermediateTensors
from tpu_inference.logger import init_logger
from tpu_inference.layers.common.sharding import ShardingAxisName

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()

def apply_rope(x: jax.Array, base_freq: int, sacle_factor: float = 1.0) -> jax.Array: 
    pass 

class RMSNorm(nnx.Module): 
    def __init__(
        self, 
        dim: int, 
        dtype: jnp.dtype,
        eps: float = 1e-6
    ): 
        self.eps = eps 
        self.weight = nnx.Param(jnp.ones(dim, param_dtype=dtype))
    
    def __call__(self, x: jax.Array) -> jax.Array: 
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps) 
        return self.weight * (x / rms)

class Gemma3MLP(nnx.Module): 
    def __init__(
        self,
        dim: int, 
        hid_dim: int,
        act_fn: str,
        dtype: jnp.dtype, 
        rng: nnx.Rngs
    ): 
        self.gate_proj = nnx.Linear(
            dim, 
            hid_dim, 
            use_bias=False, 
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.MLP_TENSOR), 
            ),
            rngs=rng 
        )
        self.up_proj = nnx.Linear(
            dim, 
            hid_dim, 
            use_bias=False, 
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.MLP_TENSOR), 
            ),
            rngs=rng 
        )
        self.down_proj = nnx.Linear(
            hid_dim, 
            dim, 
            use_bias=False, 
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (ShardingAxisName.MLP_TENSOR, None), 
            ),
            rngs=rng 
        )
        self.act_fn = {'gelu': lambda x: jax.nn.gelu(x, approximate=True), 'silu': lambda x: jax.nn.silu(x)}[act_fn]
    
    def __call__(
        self, 
        x: jax.Array  
    ) -> jax.Array: 
        gate = self.act_fn(self.gate_proj(x)) 
        up = self.up_proj(x)
        fuse = gate * up 
        return self.down_proj(fuse)


class Gemma3Model(nnx.Module): 
    def __init__(
        self, 
        vllm_config: VllmConfig, 
        rng: nnx.Rngs, 
        mesh: jax.sharding.Mesh
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
        rng: nnx.Rngs, 
        mesh: jax.sharding.Mesh
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
    key = jax.random.key(42)
    rng = nnx.Rngs(key)
    mlp = Gemma3MLP(
        dim=1024,
        hid_dim=3072,
        act_fn='silu', 
        dtype=jnp.bfloat16, 
        rng=rng
    )
    res = mlp(jax.random.normal(jax.random.key(0), (2, 4, 1024)))
    print(res)