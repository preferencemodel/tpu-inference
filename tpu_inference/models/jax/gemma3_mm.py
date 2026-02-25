from copy import deepcopy
from typing import TypedDict, List, Tuple, Optional, Callable

import jax 
import jax.numpy as jnp 
from flax import nnx 
from vllm.config import VllmConfig
from transformers import GemmaConfig

from tpu_inference.logger import init_logger
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.models.jax.utils.weight_utils import StandardWeightLoader
from tpu_inference.models.jax.gemma3 import Gemma3Model, RMSNorm
from tpu_inference.models.jax.utils.multi_modal_utils import MultiModalEmbeddings, merge_multimodal_embeddings

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()

class Gemma3ImagePixelInputs(TypedDict): 
    pass

class Gemma3MultiModalProjector(nnx.Module): 
    def __init__(
        self,
        config: VllmConfig,
        rng: nnx.Rngs, 
    ): 
        hf_config = config.model_config.hf_config
        self.vision_hidden_size = hf_config.vision_config.hidden_size 
        self.text_hidden_size = hf_config.text_config.hidden_size 
        self.dtype = config.model_config.dtype  

        self.mm_soft_emb_norm = RMSNorm(
            dim=self.vision_hidden_size,
            config=config.model_config.hf_config.text_config, 
            dtype=self.dtype
        )
        self.mm_input_projection = nnx.Linear(
            in_features=self.vision_hidden_size, 
            out_features=self.text_hidden_size,  
            param_dtype=self.dtype,
            kernel_init=init_fn, 
            rngs=rng 
        )
        

    def __call__(
        self, 
        x: jax.Array
    ) -> jax.Array: 
        x  = self.mm_soft_emb_norm(x)
        x = self.mm_input_projection(x)
        return x  

class Gemma3ForConditionalGeneration(nnx.Module): 
    WeightLoader = StandardWeightLoader

    def __init__(
        self, 
        vllm_config: VllmConfig, 
        rng: nnx.Rngs, 
        mesh: jax.sharding.Mesh
    ):
        self.vllm_config = vllm_config
        self.model_config = self.vllm_config.model_config
        self.rng = nnx.Rngs(rng)
        self.mesh = mesh

        self.vocab_size = self.model_config.get_vocab_size() 
        self.dtype = self.model_config.dtype 
        self.hidden_size = self.model_config.hf_text_config.hidden_size 

        self.multi_modal_projector = Gemma3MultiModalProjector(
            self.vllm_config,
            self.rng
        )
        self.model = Gemma3Model(
            self._preproc_hf_config_for_text_model(self.vllm_config),
            self.rng,
            self.mesh
        )

    def _preproc_hf_config_for_text_model(self, vllm_config: VllmConfig) -> VllmConfig: 
        # to keep the same interface with Gemma3Model
        vllm_config = deepcopy(vllm_config)
        vllm_config.model_config.hf_config = vllm_config.model_config.hf_text_config 
        vllm_config.model_config.hf_config.sliding_window_pattern = vllm_config.model_config.hf_text_config._sliding_window_pattern 
        return  vllm_config

    def _preproc_hf_config_for_weight_loading(self, vllm_config: VllmConfig) -> VllmConfig: 
        # to keep interface for weight loading  # num_key_value_heads/num_attention_heads
        vllm_config = deepcopy(vllm_config)
        vllm_config.model_config.hf_config.num_attention_heads = vllm_config.model_config.hf_text_config.num_attention_heads 
        vllm_config.model_config.hf_config.num_key_value_heads = vllm_config.model_config.hf_text_config.num_key_value_heads
        return  vllm_config

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
    
    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        return hidden_states @ self.model.embed.embedding.T

    def _parse_and_validate_image_input(
        self,
        **kwargs: object, 
    ) -> Optional[jax.Array]:  
        """supports only pixel values""" 
        pixel_values = kwargs.pop('pixel_values', None)
        if pixel_values is not None: # (num_patches, num_channels * patch_size * patch_size)
            logger.info(pixel_values.shape)
            breakpoint()

    def _process_image_input(self, image_input: ...) -> ...:
        image_features = self.vision_encoder(image_input)
        return self.multi_modal_projector(image_features)

    def embed_multimodal(
        self,
        **kwargs: object,
    ) -> MultiModalEmbeddings | None:
        # Validate the multimodal input keyword arguments
        logger.info(str(kwargs))
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None

        # Run multimodal inputs through encoder and projector
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings


    def load_weights(self, rng_key: jax.Array): 
        # NOTE: Since we are using nnx.eval_shape to init the model,
        # we have to pass dynamic arrays here for __call__'s usage.
        self.rng = nnx.Rngs(rng_key)

        # Key: path to a HF layer weight
        # Value: path to a nnx layer weight
        mappings = {
            "multi_modal_projector.mm_input_projection_weight": "multi_modal_projector.mm_input_projection.kernel", 
            "multi_modal_projector.mm_soft_emb_norm.weight": "multi_modal_projector.mm_soft_emb_norm.weight", 
            "language_model.model.embed_tokens.weight": "model.embed.embedding",
            "language_model.model.layers.*.mlp.down_proj.weight": "model.layers.*.mlp.down_proj.kernel",
            "language_model.model.layers.*.mlp.gate_proj.weight": "model.layers.*.mlp.gate_proj.kernel",
            "language_model.model.layers.*.mlp.up_proj.weight": "model.layers.*.mlp.up_proj.kernel",
            "language_model.model.layers.*.self_attn.k_proj.weight": "model.layers.*.self_attn.k_proj.kernel",
            "language_model.model.layers.*.self_attn.o_proj.weight": "model.layers.*.self_attn.o_proj.kernel",
            "language_model.model.layers.*.self_attn.q_proj.weight": "model.layers.*.self_attn.q_proj.kernel",
            "language_model.model.layers.*.self_attn.v_proj.weight": "model.layers.*.self_attn.v_proj.kernel",
            "language_model.model.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            "language_model.model.layers.*.post_attention_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "language_model.model.layers.*.pre_feedforward_layernorm.weight": "model.layers.*.pre_feedforward_layernorm.weight",
            "language_model.model.layers.*.post_feedforward_layernorm.weight": "model.layers.*.post_feedforward_layernorm.weight",
            "language_model.model.layers.*.self_attn.q_norm.weight": "model.layers.*.self_attn.q_norm.weight",
            "language_model.model.layers.*.self_attn.k_norm.weight": "model.layers.*.self_attn.k_norm.weight",
            "language_model.model.norm.weight": "model.norm.weight",
        }
        
        loader = self.WeightLoader(
            self._preproc_hf_config_for_weight_loading(self.vllm_config), 
            self.mesh
        )
        loader.load_weights(
            self, 
            mappings, 
            keep_hf_weight_suffix_when_match=['language_model', 'multi_modal_projector']
        ) 

    def precompile_vision_encoder(
        self,
        run_compilation_fn: Callable,
    ) -> None: 
        pass 
    