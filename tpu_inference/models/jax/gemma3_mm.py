from copy import deepcopy
from typing import TypedDict, List, Tuple, Optional, Callable

import jax 
import jax.numpy as jnp 
from flax import nnx 
from vllm.config import VllmConfig
from transformers import GemmaConfig

from tpu_inference.logger import init_logger
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.models.jax.utils.weight_utils import StandardWeightLoader, MetadataMap, get_default_maps, load_hf_weights
from tpu_inference.models.jax.gemma3 import Gemma3Model, RMSNorm
from tpu_inference.models.jax.siglip import SiglipVisionTransformer
from tpu_inference.models.jax.utils.multi_modal_utils import MultiModalEmbeddings

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
            rngs=rng, 
            use_bias=False
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
        self.num_heads = self.model_config.hf_text_config.num_attention_heads
        self.num_kv_heads = self.model_config.hf_text_config.num_key_value_heads
        self.head_dim = self.model_config.hf_text_config.head_dim 

        self.multi_modal_projector = Gemma3MultiModalProjector(
            self.vllm_config,
            self.rng
        )
        self.model = Gemma3Model(
            self._preproc_hf_config_for_text_model(self.vllm_config),
            self.rng,
            self.mesh
        )
        self.vision_model = SiglipVisionTransformer(
            self.vllm_config.model_config.hf_config.vision_config, 
            self.dtype,
            self.rng,
            self.mesh, 
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
        inputs_embeds: Optional[jax.Array] = None,
        *args,   
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array]]:
        kv_caches, x = self.model(
            kv_caches,
            input_ids,
            attention_metadata, 
            inputs_embeds
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

    def embed_input_ids(
        self, 
        input_ids: Optional[jax.Array],
        multimodal_embeddings: Optional[jax.Array]
    ) -> jax.Array:
        input_embeds = self.model.embed(input_ids)
        return input_embeds  

    def load_weights(self, rng_key: jax.Array):
        self.rng = nnx.Rngs(rng_key)

        # 1. Load language model + projector weights
        lang_mappings = {
            # multi-modal projector
            "multi_modal_projector.mm_input_projection_weight": "multi_modal_projector.mm_input_projection.kernel",
            "multi_modal_projector.mm_soft_emb_norm.weight": "multi_modal_projector.mm_soft_emb_norm.weight",
            # language model
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
        lang_metadata = get_default_maps(
            self._preproc_hf_config_for_weight_loading(self.vllm_config).model_config,
            self.mesh,
            lang_mappings
        )
        load_hf_weights(
            vllm_config=self._preproc_hf_config_for_weight_loading(self.vllm_config),
            model=self,
            metadata_map=lang_metadata,
            mesh=self.mesh,
            filter_regex=r"^(language_model|multi_modal_projector)\.",
            keep_hf_weight_suffix_when_match=['language_model', 'multi_modal_projector'],
        )

        # 2. Load vision encoder weights
        vision_mappings = {
            # embeddings
            "vision_tower.vision_model.embeddings.patch_embedding.weight": "vision_model.embeddings.patch_embedding.kernel",
            "vision_tower.vision_model.embeddings.patch_embedding.bias": "vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight": "vision_model.embeddings.position_embedding.embedding",
            # transformer layers
            "vision_tower.vision_model.encoder.layers.*.layer_norm1.weight": "vision_model.encoder.layers.*.layer_norm1.scale",
            "vision_tower.vision_model.encoder.layers.*.layer_norm1.bias": "vision_model.encoder.layers.*.layer_norm1.bias",
            "vision_tower.vision_model.encoder.layers.*.layer_norm2.weight": "vision_model.encoder.layers.*.layer_norm2.scale",
            "vision_tower.vision_model.encoder.layers.*.layer_norm2.bias": "vision_model.encoder.layers.*.layer_norm2.bias",
            "vision_tower.vision_model.encoder.layers.*.self_attn.q_proj.weight": "vision_model.encoder.layers.*.self_attn.q_proj.kernel",
            "vision_tower.vision_model.encoder.layers.*.self_attn.q_proj.bias": "vision_model.encoder.layers.*.self_attn.q_proj.bias",
            "vision_tower.vision_model.encoder.layers.*.self_attn.k_proj.weight": "vision_model.encoder.layers.*.self_attn.k_proj.kernel",
            "vision_tower.vision_model.encoder.layers.*.self_attn.k_proj.bias": "vision_model.encoder.layers.*.self_attn.k_proj.bias",
            "vision_tower.vision_model.encoder.layers.*.self_attn.v_proj.weight": "vision_model.encoder.layers.*.self_attn.v_proj.kernel",
            "vision_tower.vision_model.encoder.layers.*.self_attn.v_proj.bias": "vision_model.encoder.layers.*.self_attn.v_proj.bias",
            "vision_tower.vision_model.encoder.layers.*.self_attn.out_proj.weight": "vision_model.encoder.layers.*.self_attn.out_proj.kernel",
            "vision_tower.vision_model.encoder.layers.*.self_attn.out_proj.bias": "vision_model.encoder.layers.*.self_attn.out_proj.bias",
            "vision_tower.vision_model.encoder.layers.*.mlp.fc1.weight": "vision_model.encoder.layers.*.mlp.fc1.kernel",
            "vision_tower.vision_model.encoder.layers.*.mlp.fc1.bias": "vision_model.encoder.layers.*.mlp.fc1.bias",
            "vision_tower.vision_model.encoder.layers.*.mlp.fc2.weight": "vision_model.encoder.layers.*.mlp.fc2.kernel",
            "vision_tower.vision_model.encoder.layers.*.mlp.fc2.bias": "vision_model.encoder.layers.*.mlp.fc2.bias",
            # post layernorm
            "vision_tower.vision_model.post_layernorm.weight": "vision_model.post_layernorm.scale",
            "vision_tower.vision_model.post_layernorm.bias": "vision_model.post_layernorm.bias",
        }
        vision_metadata = MetadataMap(
            name_map=vision_mappings,
            transpose_map={
                'patch_embedding.weight': (2, 3, 1, 0),
                'q_proj': (1, 0),
                'k_proj': (1, 0),
                'v_proj': (1, 0),
                'out_proj': (1, 0),
                'fc1': (1, 0),
                'fc2': (1, 0),
            },
        )

        load_hf_weights(
            vllm_config=self._preproc_hf_config_for_weight_loading(self.vllm_config),
            model=self,
            metadata_map=vision_metadata,
            mesh=self.mesh,
            filter_regex=r"^vision_tower\.",
            keep_hf_weight_suffix_when_match=['vision_tower'],
        )

    def precompile_vision_encoder(
        self,
        run_compilation_fn: Callable,
    ) -> None: 
        # image is resized to 896 always 
        dummy = jnp.ones((1, 896, 896, 3), dtype=self.vllm_config.model_config.dtype)
        run_compilation_fn(
            "vision_encoder", 
            self.vision_model.__call__, 
            dummy
        )
    