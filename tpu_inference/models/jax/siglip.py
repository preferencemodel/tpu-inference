import jax 
import math 
from flax import nnx 
import jax.numpy as jnp
from jax.sharding import Mesh
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig 

from tpu_inference.layers.common.attention_interface import sharded_flash_attention

init_fn = nnx.initializers.uniform()    

class SiglipMLP(nnx.Module): 
    def __init__(
        self, 
        config: SiglipVisionConfig,
        dtype: jnp.dtype, 
        rng: nnx.Rngs
    ): 
        self.hidden_size = config.hidden_size 
        self.intermediate_size = config.intermediate_size 
        self.fc1 = nnx.Linear(
            in_features=self.hidden_size, 
            out_features=self.intermediate_size,
            use_bias=True,
            kernel_init=init_fn, 
            param_dtype=dtype,
            rngs=rng 
        )
        self.fc2 = nnx.Linear(
            in_features=self.intermediate_size, 
            out_features=self.hidden_size,
            use_bias=True,
            kernel_init=init_fn, 
            param_dtype=dtype,
            rngs=rng 
        )
        self.act_fn = lambda x: jax.nn.gelu(x, approximate=True)

    def __call__(
        self,
        x: jax.Array     
    ) -> jax.Array:  
        x = self.act_fn(self.fc1(x))
        x = self.fc2(x)         
        return x 


class SiglipSdpaAttention(nnx.Module): 
    """
    B - batch size; 
    T - seq length; 
    D - hidden size; 
    H - head dim;
    N - num heads;
    """
    def __init__(
        self, 
        config: SiglipVisionConfig,
        dtype: jnp.dtype, 
        rng: nnx.Rngs, 
        mesh: Mesh
    ):
        self.hidden_size = config.hidden_size 
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads

        self.q_proj = nnx.Linear(
            in_features=self.hidden_size, 
            out_features=self.hidden_size, 
            use_bias=True, 
            kernel_init=init_fn, 
            param_dtype=dtype,
            rngs=rng 
        ) 
        self.k_proj = nnx.Linear(
            in_features=self.hidden_size, 
            out_features=self.hidden_size, 
            use_bias=True, 
            kernel_init=init_fn, 
            param_dtype=dtype,
            rngs=rng 
        ) 
        self.v_proj = nnx.Linear(
            in_features=self.hidden_size, 
            out_features=self.hidden_size, 
            use_bias=True, 
            kernel_init=init_fn, 
            param_dtype=dtype,
            rngs=rng 
        ) 
        self.out_proj = nnx.Linear(
            in_features=self.hidden_size, 
            out_features=self.hidden_size, 
            use_bias=True, 
            kernel_init=init_fn, 
            param_dtype=dtype,
            rngs=rng 
        ) 

        self.flash_attention = sharded_flash_attention(
            mesh=mesh,
            causal=False,
            sm_scale=1.0 / math.sqrt(self.head_dim),
        )
    
    def __call__(
        self,
        x: jax.Array     
    ) -> jax.Array:  
        B, T, D = x.shape 

        q = self.q_proj(x)
        # (B, T, D) -> (B, T, N, H)
        q = q.reshape(B, T, self.num_attention_heads, self.head_dim) 
        k = self.k_proj(x)
        k = k.reshape(B, T, self.num_attention_heads, self.head_dim) 
        v = self.v_proj(x)
        v = v.reshape(B, T, self.num_attention_heads, self.head_dim) 

        # (B, T, N, H) -> (B, N, T, H)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # 3 * (B, N, T, H) -> (B, N, T, H)
        o = self.flash_attention(q, k, v, None)
        # (B, N, T, H) -> (B, T, N, H)
        o = jnp.transpose(o, (0, 2, 1, 3))
        # (B, T, N, H) -> (B, T, D)
        o = o.reshape(B, T, D)
        o = self.out_proj(o)

        return o 


class SiglipBlock(nnx.Module): 
    def __init__(
        self, 
        config: SiglipVisionConfig,
        dtype: jnp.dtype, 
        rng: nnx.Rngs, 
        mesh: Mesh
    ):
        self.hidden_size = config.hidden_size

        self.self_attn = SiglipSdpaAttention(
            config, 
            dtype,
            rng, 
            mesh 
        )
        self.mlp = SiglipMLP(
            config, 
            dtype,
            rng 
        )
        self.layer_norm1 = nnx.LayerNorm(
            num_features=self.hidden_size, 
            epsilon=1e-06, 
            param_dtype=dtype, 
            rngs=rng,
            use_bias=True,
        )
        self.layer_norm2 = nnx.LayerNorm(
            num_features=self.hidden_size, 
            epsilon=1e-06, 
            param_dtype=dtype, 
            rngs=rng,
            use_bias=True,
        )
    
    def __call__(
        self,
        x: jax.Array     
    ) -> jax.Array:  
        residual = x 
        x = self.layer_norm1(x)
        x = self.self_attn(x)
        x += residual 

        residual = x 
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x += residual 

        return x 


class SiglipEncoder(nnx.Module): 
    def __init__(
        self, 
        config: SiglipVisionConfig,
        dtype: jnp.dtype, 
        rng: nnx.Rngs, 
        mesh: Mesh
    ):
        self.num_hidden_layers = config.num_hidden_layers
        
        self.layers = [
            SiglipBlock(
                config,
                dtype,
                rng,
                mesh 
            ) for _ in range(self.num_hidden_layers)
        ]

    def __call__(
        self,
        x: jax.Array     
    ) -> jax.Array:  
        for layer in self.layers:
            x = layer(x) 

        return x 


class SiglipVisionEmbeddings(nnx.Module): 
    """
    B - batch size; 
    H - img height (896); 
    W - img width (896); 
    C - channels (3);
    Gh - grid height (H // patch_size = 64); 
    Gw - grid width (W // patch_size = 64); 
    N - num patches (Gh * Gw = 4096);
    D - hidden size (1152); 
    """
    def __init__(
        self, 
        config: SiglipVisionConfig,
        dtype: jnp.dtype, 
        rng: nnx.Rngs
    ):  
        self.image_size = config.image_size 
        self.patch_size = config.patch_size 
        self.num_patches = (self.image_size // self.patch_size) ** 2 
        self.num_channels = config.num_channels 
        self.hidden_size = config.hidden_size 
        
        self.patch_embedding = nnx.Conv(
            in_features=self.num_channels, 
            out_features=self.hidden_size,
            kernel_size=(self.patch_size, self.patch_size), 
            strides=(self.patch_size, self.patch_size), 
            padding='VALID', 
            param_dtype=dtype,
            kernel_init=init_fn,
            rngs=rng,
        )
        self.position_embedding = nnx.Embed(
            num_embeddings=self.num_patches,
            features=self.hidden_size, 
            param_dtype=dtype, 
            rngs=rng,
        )

    def __call__(
        self,
        x: jax.Array     
    ) -> jax.Array:  
        # (B, H, W, C) -> (B, Gh, Gw, D)
        x = self.patch_embedding(x)
        B, Gh, Gw, D = x.shape 
        # (B, Gh, Gw, D) -> (B, N, D)
        x = jnp.reshape(x, (B, Gh * Gw, D))
        # (N,) position indices
        pos_ids = jnp.arange(Gh * Gw)
        # (B, N, D) + (N, D) -> (B, N, D)
        x = x + self.position_embedding(pos_ids)
        return x

    
class SiglipVisionTransformer(nnx.Module): 
    def __init__(
        self, 
        config: SiglipVisionConfig,
        dtype: jnp.dtype, 
        rng: nnx.Rngs, 
        mesh: Mesh
    ):
        self.hidden_size = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(
            config,
            dtype, 
            rng 
        )
        self.encoder = SiglipEncoder(
            config,
            dtype, 
            rng,
            mesh
        )
        self.post_layernorm = nnx.LayerNorm(
            num_features=self.hidden_size, 
            epsilon=1e-06, 
            param_dtype=dtype, 
            rngs=rng,
            use_bias=True,
        )

    @nnx.jit
    def __call__(
        self,
        x: jax.Array     
    ) -> jax.Array:  
        # (B, H, W, C) -> (B, N, D)
        x = self.embeddings(x)
        # (B, N, D) -> (B, N, D)
        x = self.encoder(x)
        # (B, N, D) -> (B, N, D)
        x = self.post_layernorm(x)
        return x
    