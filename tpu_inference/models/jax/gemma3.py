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
from tpu_inference.models.jax.utils.weight_utils import StandardWeightLoader

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
        self.weight = nnx.Param(jnp.ones(dim, dtype=dtype))
    
    def __call__(self, x: jax.Array) -> jax.Array: 
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.rms_norm_eps) 
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

        self.q_norm = RMSNorm(dim=self.head_dim, config=config, dtype=dtype)
        self.k_norm = RMSNorm(dim=self.head_dim, config=config, dtype=dtype)

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
        q = apply_rope(q, positions=md.input_positions, head_dim=self.head_dim, rope_theta=self.rope_theta)

        # k: (T, K, H)
        k = self.k_proj(x)
        k = self.k_norm(k)
        k = apply_rope(k, positions=md.input_positions, head_dim=self.head_dim, rope_theta=self.rope_theta)

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



class Gemma3DecoderLayer(nnx.Module): 
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
    
        self.input_layer_norm = RMSNorm(
            dim=self.hidden_size, 
            config=config,
            dtype=dtype
        )
        self.self_attn = Gemma3Attention(
            config,
            dtype, 
            rng, 
            mesh, 
            kv_cache_dtype, 
            is_local
        )
        self.post_attn_layer_norm = RMSNorm(
            dim=self.hidden_size, 
            config=config,
            dtype=dtype
        )
        self.pre_feedforward_layernorm = RMSNorm(
            dim=self.hidden_size, 
            config=config,
            dtype=dtype
        )
        self.mlp = Gemma3MLP(
            config, 
            dtype,
            rng
        )
        self.post_feedforward_layernorm = RMSNorm(
            dim=self.hidden_size, 
            config=config,
            dtype=dtype
        )

    def __call__(
        self,
        kv_cache: Optional[jax.Array],
        x: jax.Array,
        attention_metadata: AttentionMetadata 
    ) -> Tuple[jax.Array, jax.Array]: 
        x_norm = self.input_layer_norm(x)
        kv_cache, attn_output = self.self_attn(
            kv_cache,
            x_norm, 
            attention_metadata 
        )
        attn_output = self.post_attn_layer_norm(attn_output)
        attn_output += x 

        outputs = self.pre_feedforward_layernorm(attn_output)
        outputs = self.mlp(outputs)
        outputs = self.post_feedforward_layernorm(outputs)
        outputs += attn_output 

        return kv_cache, outputs 
        


class Gemma3Model(nnx.Module): 
    def __init__(
        self, 
        vllm_config: VllmConfig, 
        rng: nnx.Rngs, 
        mesh: jax.sharding.Mesh
    ):
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config 

        self.vocab_size = model_config.get_vocab_size() 
        self.dtype = model_config.dtype 
        self.hidden_size = hf_config.hidden_size 
        self.sliding_window_pattern = hf_config.sliding_window_pattern

        self.embed = nnx.Embed(
            num_embeddings=self.vocab_size, 
            features=self.hidden_size, 
            param_dype=self.dtype,
            embedding_init=init_fn,
            rngs=rng
        ) 
        self.layers = [
            Gemma3DecoderLayer(
                config=hf_config,
                dtype=self.dtype,
                rng=rng,
                mesh=mesh,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype, 
                is_local=(i + 1) % self.sliding_window_pattern != 0,
            ) for i in range(hf_config.num_hidden_layers)
        ]
        self.norm = RMSNorm(
            self.hidden_size,
            hf_config,
            self.dtype
        )
    
    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: Optional[jax.Array],
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        x = self.embed(input_ids)
        
        for i, layer in enumerate(self.layers): 
            kv_cache = kv_caches[i]
            kv_cache, x = layer(
                kv_cache,
                x, 
                attention_metadata
            ) 
            kv_caches[i] = kv_cache

        x = self.norm(x)

        return kv_caches, x 


class Gemma3ForCausalLM(nnx.Module):
    WeightLoader = StandardWeightLoader

    def __init__(
        self, 
        vllm_config: VllmConfig, 
        rng: nnx.Rngs, 
        mesh: jax.sharding.Mesh
    ):
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config 

        self.vocab_size = model_config.get_vocab_size() 
        self.dtype = model_config.dtype 
        self.hidden_size = hf_config.hidden_size 
    
        self.model = Gemma3Model(
            vllm_config,
            rng,
            mesh
        )
        self.lm_head = nnx.Linear(
            in_features=self.hidden_size,
            out_features=self.vocab_size, 
            use_bias=False, 
            param_dtype=self.dtype,
            kernel_init=init_fn, 
            rngs=rng 
        )
            
    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        *args,   
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array]]:
        kv_caches, x = self.model(
            kv_caches,
            input_ids,
            attention_metadata
        )
        return kv_caches, x, []
    
    def load_weights(self, rng_key: jax.Array): 
        pass 
    
    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        return self.lm_head(hidden_states)


# Playground :D
if __name__ == '__main__': 
    pass