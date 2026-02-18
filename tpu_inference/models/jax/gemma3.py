from typing import List, Tuple

import jax 
import jax.numpy as jnp 
from flax import nnx 
from vllm.config import VllmConfig
from transformers import GemmaConfig 

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.models.jax.jax_intermediate_tensor import JaxIntermediateTensors
from tpu_inference.logger import init_logger
from tpu_inference.layers.common.sharding import ShardingAxisName

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()

class RMSNorm(nnx.Module): 
    def __init__(
        self, 
        dim: int, 
        config: GemmaConfig,
        dtype: jnp.dtype,
    ): 
        self.rms_norm_eps = config.rms_norm_eps 
        self.weight = nnx.Param(jnp.ones(dim, param_dtype=dtype))
    
    def __call__(self, x: jax.Array) -> jax.Array: 
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps) 
        return self.weight * (x / rms)

class Gemma3MLP(nnx.Module): 
    def __init__(
        self,
        config: GemmaConfig,
        dtype: jnp.dtype, 
        rng: nnx.Rngs
    ): 
        self.hidden_size = config.hidden_size 
        self.intermediate_size = config.intermediate_size  
        self.hidden_act = config.hidden_act
        self.gate_proj = nnx.Linear(
            in_features=self.hidden_size, 
            out_features=self.intermediate_size, 
            use_bias=False, 
            param_dtype=dtype,
            kernel_init=init_fn,
            rngs=rng 
        )
        self.up_proj = nnx.Linear(
            in_features=self.hidden_size, 
            out_features=self.intermediate_size, 
            use_bias=False, 
            param_dtype=dtype,
            kernel_init=init_fn, 
            rngs=rng 
        )
        self.down_proj = nnx.Linear(
            in_features=self.intermediate_size, 
            out_features=self.hidden_size, 
            use_bias=False, 
            param_dtype=dtype,
            kernel_init=init_fn,            
            rngs=rng 
        )
        self.act_fn = {'gelu_pytorch_tanh': lambda x: jax.nn.gelu(x, approximate=True)}[self.hidden_act]
    
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
    cfg = GemmaConfig(
        hidden_size=1152, 
        intermidate_size=6912,
    )
    mlp = Gemma3MLP(
        config=cfg,
        dtype=jnp.bfloat16, 
        rng=rng
    )
    res = mlp(jax.random.normal(jax.random.key(0), (2, 4, cfg.hidden_size)))
    print(res)