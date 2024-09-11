from dataclasses import dataclass
from typing import Any, Dict, List, TYPE_CHECKING, Optional, Tuple, Type

import torch

try:
    import torch_npu
except:
    raise ImportError("torch-npu not found. 'pip install torch-npu' if using Ascend backend")

import math
import numpy as np

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType,
                                              AttentionMetadataBuilder)
from vllm.attention.backends.utils import (PAD_SLOT_ID, compute_slot_mapping,
                                           compute_slot_mapping_start_idx,
                                           is_block_tables_empty)
from vllm.attention.ops.paged_attn import PagedAttention

if TYPE_CHECKING:
    from vllm.worker.npu_model_runner import ModelInputForNPUBuilder

from vllm.utils import make_tensor_with_pad

SHARE_MASK_TRIL_PREFIX_CACHE = None
SHARE_MASK_TRIL = None


def compute_npu_slot_mapping(is_profile_run: bool, slot_mapping: List[int],
                             seq_id: int, seq_len: int, context_len: int,
                             start_idx: int, block_size: int,
                             block_tables: Dict[int, List[int]],
                             max_query_len: int):
    """
    Compute slot mapping.
    """
    if is_profile_run:
        slot_mapping.extend([[PAD_SLOT_ID, PAD_SLOT_ID]] * seq_len)
        return

    padding_mask_len = max(0, start_idx - context_len)
    slot_mapping.extend([PAD_SLOT_ID] * padding_mask_len)

    range_start = max(start_idx, context_len)
    range_end = seq_len
    block_table = block_tables[seq_id]

    # numpy implementation will be faster than python if we have
    # many elements, otherwise it will be slower.
    for i in range(range_start, range_end):
        block_number = block_table[i // block_size]
        block_offset = i % block_size
        slot_mapping.append([block_number, block_offset])

    slot_mapping.extend([[np.iinfo(np.int_).max, 0]] * (max_query_len - range_end + range_start))


class AscendAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "ascend"

    @staticmethod
    def get_impl_cls() -> Type["AscendAttentionBackendImpl"]:
        return AscendAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["AscendMetadata"]:
        return AscendMetadata

    @staticmethod
    def get_builder_cls() -> Type["AscendMetadataBuilder"]:
        return AscendMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        # return (2, num_blocks, block_size, num_kv_heads * head_size)
        return (2, num_blocks, block_size, num_kv_heads * head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: List[torch.Tensor],
        dst_kv_cache: List[torch.Tensor],
        src_to_dst: Dict[int, int],
    ) -> None:
        # TODO (cmq): check me
        src_key_cache, src_value_cache = src_kv_cache[0], src_kv_cache[1]
        dst_key_cache, dst_value_cache = dst_kv_cache[0], dst_kv_cache[1]
        for src, dst in src_to_dst.items():
            dst_key_cache[dst] = src_key_cache[src].to(dst_key_cache.device)
            dst_value_cache[dst] = src_value_cache[src].to(dst_key_cache.device)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: Dict[int, List[int]],
    ) -> None:
        # TODO (cmq): check me
        key_caches = kv_caches[0]
        value_caches = kv_caches[1]
        layers = len(key_caches)
        for src_id, dsts in src_to_dists.items():
            for dst_id in dsts:
                key_caches[:][dst_id] = key_caches[:][src_id]
                value_caches[:][dst_id] = value_caches[:][src_id]


class AscendPagedAttention(PagedAttention):

    @staticmethod
    def write_to_paged_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_indices: torch.Tensor,
    ) -> None:
        batch_size, seq_len, slot_dim = slot_indices.shape
        key = key.view(batch_size, seq_len, key_cache.shape[-1])
        value = value.view(batch_size, seq_len, value_cache.shape[-1])
        torch_npu.npu_scatter_nd_update_(key_cache, slot_indices, key)
        torch_npu.npu_scatter_nd_update_(value_cache, slot_indices, value)


@dataclass(kw_only=True)
class AscendMetadata(AttentionMetadata, PagedAttentionMetadata):
    # TODO(lqy): just copy from xformers, need to romove unnecessary args.
    seq_lens_tensor: Optional[torch.Tensor]

    # FIXME: It is for flash attn.
    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool

    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]] = None

    # FIXME: It is for flash attn.
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor] = None

    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor] = None

    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int] = None

    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor] = None

    # Self-attention prefill/decode metadata cache
    _cached_prefill_metadata: Optional["XFormersMetadata"] = None
    _cached_decode_metadata: Optional["XFormersMetadata"] = None

    # Begin encoder attn & enc/dec cross-attn fields...

    # Encoder sequence lengths representation
    encoder_seq_lens: Optional[List[int]] = None
    encoder_seq_lens_tensor: Optional[torch.Tensor] = None

    # Maximum sequence length among encoder sequences
    max_encoder_seq_len: Optional[int] = None

    # Number of tokens input to encoder
    num_encoder_tokens: Optional[int] = None

    # Cross-attention memory-mapping data structures: slot mapping
    # and block tables
    cross_slot_mapping: Optional[torch.Tensor] = None
    cross_block_tables: Optional[torch.Tensor] = None

    def __post_init__(self):
        # Set during the execution of the first attention op.
        # It is a list because it is needed to set per prompt
        # when alibi slopes is used. It is because of the limitation
        # from xformer API.
        # will not appear in the __repr__ and __init__
        self.attn_bias: Optional[List[AttentionBias]] = None
        self.encoder_attn_bias: Optional[List[AttentionBias]] = None
        self.cross_attn_bias: Optional[List[AttentionBias]] = None

    @property
    def is_all_encoder_attn_metadata_set(self):
        '''
        All attention metadata required for encoder attention is set.
        '''
        return ((self.encoder_seq_lens is not None)
                and (self.encoder_seq_lens_tensor is not None)
                and (self.max_encoder_seq_len is not None))

    @property
    def is_all_cross_attn_metadata_set(self):
        '''
        All attention metadata required for enc/dec cross-attention is set.

        Superset of encoder attention required metadata.
        '''
        return (self.is_all_encoder_attn_metadata_set
                and (self.cross_slot_mapping is not None)
                and (self.cross_block_tables is not None))

    @property
    def prefill_metadata(self) -> Optional["AscendMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            # Recover cached prefill-phase attention
            # metadata structure
            return self._cached_prefill_metadata

        assert ((self.seq_lens is not None)
                or (self.encoder_seq_lens is not None))
        assert ((self.seq_lens_tensor is not None)
                or (self.encoder_seq_lens_tensor is not None))

        # Compute some attn_metadata fields which default to None
        query_start_loc = (None if self.query_start_loc is None else
                           self.query_start_loc[:self.num_prefills + 1])
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping)
        seq_lens = (None if self.seq_lens is None else
                    self.seq_lens)
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor)
        context_lens_tensor = (None if self.context_lens_tensor is None else
                               self.context_lens_tensor)
        block_tables = (None if self.block_tables is None else
                        self.block_tables)

        # Construct & cache prefill-phase attention metadata structure
        self._cached_prefill_metadata = AscendMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            query_start_loc=query_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables)
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["AscendMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            # Recover cached decode-phase attention
            # metadata structure
            return self._cached_decode_metadata
        assert ((self.seq_lens_tensor is not None)
                or (self.encoder_seq_lens_tensor is not None))

        # Compute some attn_metadata fields which default to None
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping)
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor)
        block_tables = (None if self.block_tables is None else
                        self.block_tables)

        # Construct & cache decode-phase attention metadata structure
        self._cached_decode_metadata = XFormersMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=slot_mapping,
            seq_lens_tensor=seq_lens_tensor,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            block_tables=block_tables,
            use_cuda_graph=self.use_cuda_graph,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables)
        return self._cached_decode_metadata


class AscendMetadataBuilder(CommonMetadataBuilder[AscendMetadata]):
    _metadata_cls = AscendMetadata

    def _add_seq_group(
            self, inter_data: "ModelInputForGPUBuilder.InterDataForSeqGroup",
            chunked_prefill_enabled: bool):
        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables
        computed_block_nums = inter_data.computed_block_nums
        max_query_len = max(inter_data.query_lens)

        for (seq_id, token_len, seq_len, curr_seq_len, query_len, context_len,
             curr_sliding_window_block) in zip(
            inter_data.seq_ids, [len(t) for t in inter_data.input_tokens],
            inter_data.orig_seq_lens, inter_data.seq_lens,
            inter_data.query_lens, inter_data.context_lens,
            inter_data.curr_sliding_window_blocks):
            self.context_lens.append(context_len)
            if is_prompt:
                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                assert query_len == 1, (
                    "seq_len: {}, context_len: {}, query_len: {}".format(
                        seq_len, context_len, query_len))
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # Compute block table.
            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            block_table = []
            if inter_data.prefix_cache_hit:
                block_table = computed_block_nums
            elif ((chunked_prefill_enabled or not is_prompt)
                  and block_tables is not None):
                block_table = block_tables[seq_id][-curr_sliding_window_block:]
            self.block_tables.append(block_table)

            # Compute slot mapping.
            is_profile_run = is_block_tables_empty(block_tables)
            start_idx = compute_slot_mapping_start_idx(
                is_prompt, query_len, context_len, self.sliding_window,
                self.use_v2_block_manager)
            compute_npu_slot_mapping(is_profile_run, self.slot_mapping, seq_id,
                                     seq_len, context_len, start_idx,
                                     self.block_size, inter_data.block_tables,
                                     max_query_len)


class AscendAttentionBackendImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        self.alibi_slopes = alibi_slopes

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def attn_free_mask_pfa(self):
        global SHARE_MASK_TRIL_PREFIX_CACHE
        if SHARE_MASK_TRIL_PREFIX_CACHE is None:
            SHARE_MASK_TRIL_PREFIX_CACHE = torch.triu(
                torch.ones(1, 1, 2048, 2048, dtype=bool, device="npu"),
                diagonal=1,
            )
        return SHARE_MASK_TRIL_PREFIX_CACHE

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: List[torch.Tensor],
        attn_metadata: AscendMetadata,
        kv_scale: float = 1.0,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.

        Args:
            query: shape = [batch_size, seq_len * num_heads * head_size]
            key: shape = [batch_size, seq_len * num_kv_heads * head_size]
            value: shape = [batch_size, seq_len * num_kv_heads * head_size]
            key_cache = [num_blocks, block_size, num_kv_heads * head_size]
            value_cache = [num_blocks, block_size, num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [batch_size, seq_len * num_heads * head_size]
        """
        assert k_scale == 1.0 and v_scale == 1.0
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "PallasAttentionBackendImpl"
            )
        # view q k v to BSH
        batch_size = query.shape[0]

        if kv_cache is not None:
            if attn_metadata.num_prefills > 0:
                slot_indices = attn_metadata.prefill_metadata.slot_indices
            else:
                slot_indices = attn_metadata.decode_metadata.slot_indices
            key_cache, value_cache = kv_cache[0], kv_cache[1]
            AscendPagedAttention.write_to_paged_cache(
                key,
                value,
                key_cache,
                value_cache,
                slot_indices,
            )

        if attn_metadata.num_prefills > 0:
            # TODO (cmq): modify attn_metadata.sparse_mode, attention_mask ...
            # add ENV var to turn on/off maskfree_attn and change 16384
            if attn_metadata.attn_mask is None and query.shape[0] > 16384:
                attn_metadata.attn_mask = self.attn_free_mask_pfa()
                attn_metadata.sparse_mode = 2

            if attn_metadata.attn_mask is None:
                query_len = attn_metadata.seq_lens_tensor
                kv_len = torch.zeros_like(query_len).to(torch.long)
                attention_mask = gen_input_mask(
                    len(attn_metadata.seq_lens),
                    attn_metadata.max_prefill_seq_len,
                    query_len,
                    kv_len,
                )

                if self.sliding_window is not None:
                    attention_mask = ~attention_mask
                    attention_mask = torch.triu(
                        attention_mask, diagonal=1 - self.sliding_window
                    )
                    attention_mask = ~attention_mask
                attn_metadata.attn_mask = attention_mask

            if self.alibi_slopes is not None and attn_metadata.pse_shift is None:
                attn_metadata.pse_shift = _make_alibi_bias(
                    self.alibi_slopes,
                    self.num_kv_heads,
                    dtype=query.dtype,
                    seq_len=attn_metadata.max_prefill_seq_len,
                    batch_size=batch_size,
                )

            # shape of q/k/v [B,S*H] --> [B,S,N,D]
            query = query.view(
                -1, attn_metadata.max_prefill_seq_len, self.num_heads, self.head_size
            ).transpose(1, 2)
            key = key.view(
                -1, attn_metadata.max_prefill_seq_len, self.num_kv_heads, self.head_size
            ).transpose(1, 2)
            value = value.view(
                -1, attn_metadata.max_prefill_seq_len, self.num_kv_heads, self.head_size
            ).transpose(1, 2)

            # FA for prefill phase
            output = torch_npu.npu_prompt_flash_attention(
                query,
                key,
                value,
                pse_shift=attn_metadata.pse_shift,
                atten_mask=attn_metadata.attn_mask,
                num_heads=self.num_heads,
                scale_value=1 / math.sqrt(self.head_size),
                input_layout="BNSD",
                num_key_value_heads=self.num_kv_heads,
                pre_tokens=65535,
                next_tokens=0,
                sparse_mode=attn_metadata.sparse_mode,
            )
            output = output.transpose(1, 2).reshape(
                batch_size, -1, self.num_heads * self.head_size
            )

        elif decode_meta := attn_metadata.decode_metadata:
            # FA for decoding phase
            assert kv_cache is not None
            # shape of query [B,S*H] --> [B,S,H]
            query = query.view(
                -1,
                1,
                self.head_size * self.num_heads,
            )
            output = torch_npu.npu_incre_flash_attention(
                query,
                key_cache,
                value_cache,
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                scale_value=self.scale,
                input_layout="BSH",
                block_table=attn_metadata.block_tables,
                block_size=attn_metadata.block_size,  # max val of block_size == 512
                actual_seq_lengths=attn_metadata.seq_lens,
            )
        # [B,S,H] --> [B,H]
        if output.shape[1] == 1:
            output = output.squeeze(1)
        return output


def gen_input_mask(
        batch_size, seq_len, query_len: torch.LongTensor, kv_len: torch.LongTensor
):
    """
    Generating lower triangular matrix
    """
    global SHARE_MASK_TRIL
    if SHARE_MASK_TRIL is None or SHARE_MASK_TRIL.shape[0] < seq_len:
        SHARE_MASK_TRIL = ~torch.tril(
            torch.ones(seq_len, seq_len, dtype=bool, device="npu")
        )
    range_idx = torch.arange(seq_len, device=query_len.device).expand(batch_size, -1)
    select_idx = range_idx + kv_len.unsqueeze(1)
    attn_mask = torch.index_select(
        SHARE_MASK_TRIL, index=select_idx.view(-1), dim=0
    ).view(batch_size, seq_len, -1)
    padding_idx = range_idx >= query_len.unsqueeze(1)
    padding_idx = padding_idx.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(padding_idx, 1)
    q_len = attn_mask.shape[1]
    attn_mask = attn_mask[:, :, :q_len]

    return attn_mask.unsqueeze(1)[0].unsqueeze(0)


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    dtype: torch.dtype,
    seq_len: int,
    batch_size: int,
):
    bias = torch.arange(seq_len, dtype=dtype, device=alibi_slopes.device)
    # NOTE(zhuohan): HF uses
    #     `bias = bias[None, :].repeat(seq_len, 1)`
    # here. We find that both biases give the same results, but
    # the bias below more accurately follows the original ALiBi
    # paper.
    # Calculate a matrix where each element represents ith element- jth
    # element.
    bias = bias[None, :] - bias[:, None]

    padded_len = (seq_len + 7) // 8 * 8
    num_heads = alibi_slopes.shape[0]
    bias = torch.empty(
        batch_size,
        num_heads,
        seq_len,
        padded_len,
        device=alibi_slopes.device,
        dtype=dtype,
    )[:, :, :, :seq_len].copy_(bias)
    bias.mul_(alibi_slopes[:, None, None])
    if num_heads != num_kv_heads:
        bias = bias.unflatten(1, (num_kv_heads, num_heads // num_kv_heads))

    return bias
