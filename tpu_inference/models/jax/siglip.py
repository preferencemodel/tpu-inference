import jax 
from flax import nnx 
import jax.numpy as jnp
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig 

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
        fc1 = self.act_fn(self.fc1(x))
        fc2 = self.fc2(fc1)         
        return fc2 


