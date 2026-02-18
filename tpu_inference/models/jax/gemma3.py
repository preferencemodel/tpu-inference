from typing import List, Tuple, Optional

import jax 
import jax.numpy as jnp
from jax.sharding import Mesh 
from flax import nnx 
from vllm.config import VllmConfig
from transformers import GemmaConfig 

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.models.jax.jax_intermediate_tensor import JaxIntermediateTensors
from tpu_inference.layers.common.attention_interface import attention
from tpu_inference.logger import init_logger
from tpu_inference.layers.jax.rope_interface import apply_rope

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

        
class Gemma3Attention(nnx.Module): 
    """
    T - seq. length; 
    D - hidden size; 
    H - head dim.;
    N - num. query heads;
    K - num. kv heads;
    """
    def __init__(
        self, 
        config: GemmaConfig, 
        dtype: jnp.dtype,
        rng: nnx.Rngs,
        mesh: Mesh, 
        kv_cache_dtype: str,
        is_local: bool = False
    ):
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim 
        self.num_heads = config.num_attention_heads 
        self.num_kv_heads = config.num_key_value_heads
        self.rope_theta = config.rope_theta if not is_local else config.rope_local_base_freq
        self.sliding_window = None if not is_local else config.sliding_window 
        self.query_pre_attn_scalar = config.query_pre_attn_scalar
        self.mesh = mesh 

        self.q_norm = RMSNorm(dim=self.head_dim)
        self.k_norm = RMSNorm(dim=self.head_dim)

        self.q_proj = nnx.Einsum(
            "TD,DNH->TNH", 
            (self.hidden_size, self.num_heads, self.head_dim),
            param_dtype=dtype, 
            kernel_init=init_fn,
            rngs=rng 
        )
        self.k_proj = nnx.Einsum(
            "TD,DKH->TKH", 
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype, 
            kernel_init=init_fn,
            rngs=rng 
        ) 
        self.v_proj = nnx.Einsum(
            "TD,DKH->TKH", 
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype, 
            kernel_init=init_fn,
            rngs=rng 
        )
        self.o_proj = nnx.Einsum(
            "TNH,NHD->TD", 
            (self.num_heads, self.head_dim, self.hidden_size),
            param_dtype=dtype, 
            kernel_init=init_fn,
            rngs=rng 
        )

    def __call__(
        self,
        kv_cache: Optional[jax.Array],
        x: jax.Array,
        attention_metadata: AttentionMetadata 
    ) -> Tuple[jax.Array, jax.Array]:
        md = attention_metadata

        # q: (T, N, H)
        q = self.q_proj(x)
        q = self.q_norm(q)
        q = q * self.query_pre_attn_scalar ** -0.5
        q = apply_rope(q, positions=md.intput_positions, head_dim=self.head_dim, rope_theta=self.rope_theta)

        # k: (T, K, H)
        k = self.k_proj(x)
        k = self.k_norm(k)
        k = apply_rope(k, positions=md.intput_positions, head_dim=self.head_dim, rope_theta=self.rope_theta)

        # v: (T, K, H)
        v = self.v_proj(x)

        new_kv_cache, outputs = attention(
            kv_cache, 
            q, 
            k, 
            v, 
            attention_metadata, 
            self.mesh,
            self.head_dim, 
            attention_chunk_size=self.sliding_window
        )

        # o: (T, D)
        o = self.o_proj(outputs)

        return new_kv_cache, o


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