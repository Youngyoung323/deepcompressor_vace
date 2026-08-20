# -*- coding: utf-8 -*-
"""Structure definitions for Wan2.1-VACE (DiffSynth-Studio WanModel).

These structs inherit from the Diffusion-prefixed base classes
so that DeepCompressor's smooth/weight/activation quantization code
(which uses isinstance checks) works without modification.
"""

import typing as tp
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field

import torch.nn as nn

from collections import defaultdict

from deepcompressor.nn.struct.attn import (
    AttentionConfigStruct,
    FeedForwardConfigStruct,
)
from deepcompressor.nn.struct.base import BaseModuleStruct
from deepcompressor.utils.common import join_name

from .struct import (
    DiffusionAttentionStruct,
    DiffusionBlockStruct,
    DiffusionFeedForwardStruct,
    DiffusionModelStruct,
    DiffusionModuleStruct,
    DiffusionTransformerBlockStruct,
)

__all__ = [
    "WanAttentionStruct",
    "WanFeedForwardStruct",
    "WanTransformerBlockStruct",
    "WanDiTStruct",
    "VaceWanTransformerBlockStruct",
    "VaceWanDiTStruct",
]


@dataclass(kw_only=True)
class WanAttentionStruct(DiffusionAttentionStruct):
    """Wraps SelfAttention or CrossAttention from WanModel's DiTBlock.

    Inherits from DiffusionAttentionStruct so isinstance checks in
    smooth.py / weight.py still pass.
    """

    module: nn.Module = field(repr=False, kw_only=False)
    parent: tp.Optional["WanTransformerBlockStruct"] = field(repr=False)

    def filter_kwargs(self, kwargs: dict) -> dict:
        return {}

    @staticmethod
    def _default_construct(
        module: nn.Module,
        /,
        parent: tp.Optional["WanTransformerBlockStruct"] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "WanAttentionStruct":
        q_proj = module.q
        k_proj = module.k
        v_proj = module.v
        o_proj = module.o

        k_img_proj = getattr(module, "k_img", None)
        v_img_proj = getattr(module, "v_img", None)

        hidden_size = q_proj.weight.shape[1]
        inner_size = q_proj.weight.shape[0]
        num_heads = module.num_heads if hasattr(module, "num_heads") else (inner_size // 128)
        has_qk_norm = hasattr(module, "norm_q") and module.norm_q is not None

        is_cross = "cross_attn" in rname or idx == 1
        with_rope = not is_cross

        if is_cross:
            # Cross-attention: k,v map to context (add_k/add_v in the framework)
            add_hidden = k_proj.weight.shape[1]
            config = AttentionConfigStruct(
                hidden_size=hidden_size,
                add_hidden_size=add_hidden,
                inner_size=inner_size,
                num_query_heads=num_heads,
                num_key_value_heads=num_heads,
                with_qk_norm=has_qk_norm,
                with_rope=with_rope,
                linear_attn=False,
            )
            return WanAttentionStruct(
                module=module, parent=parent, fname=fname, idx=idx,
                rname=rname, rkey=rkey, config=config,
                q_proj=q_proj, k_proj=None, v_proj=None, o_proj=o_proj,
                add_q_proj=None, add_k_proj=k_proj, add_v_proj=v_proj, add_o_proj=None,
                q=None, k=None, v=None,
                q_proj_rname="q", k_proj_rname="", v_proj_rname="", o_proj_rname="o",
                add_q_proj_rname="", add_k_proj_rname="k", add_v_proj_rname="v", add_o_proj_rname="",
                q_rname="", k_rname="", v_rname="",
            )
        else:
            # Self-attention: k,v are self, k_img/v_img are additional (if present)
            add_hidden = k_img_proj.weight.shape[1] if k_img_proj is not None else 0
            config = AttentionConfigStruct(
                hidden_size=hidden_size,
                add_hidden_size=add_hidden,
                inner_size=inner_size,
                num_query_heads=num_heads,
                num_key_value_heads=num_heads,
                with_qk_norm=has_qk_norm,
                with_rope=with_rope,
                linear_attn=False,
            )
            return WanAttentionStruct(
                module=module, parent=parent, fname=fname, idx=idx,
                rname=rname, rkey=rkey, config=config,
                q_proj=q_proj, k_proj=k_proj, v_proj=v_proj, o_proj=o_proj,
                add_q_proj=None,
                add_k_proj=k_img_proj, add_v_proj=v_img_proj, add_o_proj=None,
                q=None, k=None, v=None,
                q_proj_rname="q", k_proj_rname="k", v_proj_rname="v", o_proj_rname="o",
                add_q_proj_rname="",
                add_k_proj_rname="k_img" if k_img_proj else "",
                add_v_proj_rname="v_img" if v_img_proj else "",
                add_o_proj_rname="",
                q_rname="", k_rname="", v_rname="",
            )


@dataclass(kw_only=True)
class WanFeedForwardStruct(DiffusionFeedForwardStruct):
    """Wraps the FFN Sequential(Linear, GELU, Linear) from WanModel's DiTBlock.

    Inherits from DiffusionFeedForwardStruct so isinstance checks pass.
    """

    module: nn.Sequential = field(repr=False, kw_only=False)
    parent: tp.Optional["WanTransformerBlockStruct"] = field(repr=False)

    @staticmethod
    def _default_construct(
        module: nn.Sequential,
        /,
        parent: tp.Optional["WanTransformerBlockStruct"] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "WanFeedForwardStruct":
        up_proj = module[0]
        down_proj = module[2]
        assert isinstance(up_proj, nn.Linear)
        assert isinstance(down_proj, nn.Linear)
        config = FeedForwardConfigStruct(
            hidden_size=up_proj.weight.shape[1],
            intermediate_size=down_proj.weight.shape[1],
            intermediate_act_type="gelu",
            num_experts=1,
        )
        return WanFeedForwardStruct(
            module=module, parent=parent, fname=fname, idx=idx,
            rname=rname, rkey=rkey, config=config,
            up_projs=[up_proj], down_projs=[down_proj],
            up_proj_rnames=["0"], down_proj_rnames=["2"],
        )


@dataclass(kw_only=True)
class WanTransformerBlockStruct(DiffusionTransformerBlockStruct):
    """Wraps a single DiTBlock from WanModel.

    Inherits from DiffusionTransformerBlockStruct so isinstance checks pass.
    """

    attn_struct_cls: tp.ClassVar[type[WanAttentionStruct]] = WanAttentionStruct
    ffn_struct_cls: tp.ClassVar[type[WanFeedForwardStruct]] = WanFeedForwardStruct
    _wan_attn_rkeys: tp.ClassVar[list[str]] = ["self_attn", "cross_attn"]

    parent: tp.Optional["WanDiTStruct"] = field(repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.attn_structs = [
            self.attn_struct_cls.construct(
                attn, parent=self, fname="attn", rname=self.attn_rnames[idx],
                rkey=self._wan_attn_rkeys[idx], idx=idx
            )
            for idx, attn in enumerate(self.attns)
        ]

    @classmethod
    def _get_default_key_map(cls) -> dict[str, set[str]]:
        key_map: dict[str, set[str]] = defaultdict(set)
        norm_rkey = norm_key = cls.norm_rkey
        add_norm_rkey = add_norm_key = cls.add_norm_rkey
        key_map[norm_rkey].add(norm_key)
        key_map[add_norm_rkey].add(add_norm_key)

        attn_cls = cls.attn_struct_cls
        old_attn_rkey = cls.attn_rkey  # "attn"

        for attn_prefix in cls._wan_attn_rkeys:
            qkv_key = join_name(attn_prefix, attn_cls.qkv_proj_rkey, sep="_")
            q_key = join_name(attn_prefix, attn_cls.q_rkey, sep="_")
            k_key = join_name(attn_prefix, attn_cls.k_rkey, sep="_")
            v_key = join_name(attn_prefix, attn_cls.v_rkey, sep="_")
            out_key = join_name(attn_prefix, attn_cls.out_proj_rkey, sep="_")
            add_qkv_key = join_name(attn_prefix, attn_cls.add_qkv_proj_rkey, sep="_")
            add_out_key = join_name(attn_prefix, attn_cls.add_out_proj_rkey, sep="_")

            key_map[attn_prefix].update({qkv_key, q_key, k_key, v_key, out_key})
            add_key = join_name(attn_prefix, "add", sep="_")
            key_map[add_key].update({add_qkv_key, add_out_key})
            # qkv_key is retained as a compatibility alias for all three
            # self-attention projections.
            key_map[qkv_key].update({qkv_key, q_key, k_key, v_key})
            key_map[attn_cls.q_rkey].add(q_key)
            key_map[attn_cls.k_rkey].add(k_key)
            key_map[attn_cls.v_rkey].add(v_key)
            key_map[out_key].add(out_key)
            key_map[add_qkv_key].add(add_qkv_key)
            key_map[add_out_key].add(add_out_key)

        all_qkv = {join_name(p, attn_cls.qkv_proj_rkey, sep="_") for p in cls._wan_attn_rkeys}
        all_q = {join_name(p, attn_cls.q_rkey, sep="_") for p in cls._wan_attn_rkeys}
        all_k = {join_name(p, attn_cls.k_rkey, sep="_") for p in cls._wan_attn_rkeys}
        all_v = {join_name(p, attn_cls.v_rkey, sep="_") for p in cls._wan_attn_rkeys}
        all_out = {join_name(p, attn_cls.out_proj_rkey, sep="_") for p in cls._wan_attn_rkeys}
        all_add_qkv = {join_name(p, attn_cls.add_qkv_proj_rkey, sep="_") for p in cls._wan_attn_rkeys}
        all_add_out = {join_name(p, attn_cls.add_out_proj_rkey, sep="_") for p in cls._wan_attn_rkeys}

        old_qkv = join_name(old_attn_rkey, attn_cls.qkv_proj_rkey, sep="_")
        old_out = join_name(old_attn_rkey, attn_cls.out_proj_rkey, sep="_")
        old_add_qkv = join_name(old_attn_rkey, attn_cls.add_qkv_proj_rkey, sep="_")
        old_add_out = join_name(old_attn_rkey, attn_cls.add_out_proj_rkey, sep="_")
        old_add_attn = join_name(old_attn_rkey, "add", sep="_")

        key_map[old_qkv].update(all_qkv | all_q | all_k | all_v)
        key_map[old_out].update(all_out)
        key_map[old_add_qkv].update(all_add_qkv)
        key_map[old_add_out].update(all_add_out)
        key_map[old_attn_rkey].update(all_qkv | all_q | all_k | all_v | all_out)
        key_map[old_add_attn].update(all_add_qkv | all_add_out)

        ffn_cls = cls.ffn_struct_cls
        ffn_key = ffn_rkey = cls.ffn_rkey
        add_ffn_key = add_ffn_rkey = cls.add_ffn_rkey
        up_key = up_rkey = join_name(ffn_key, ffn_cls.up_proj_rkey, sep="_")
        down_key = down_rkey = join_name(ffn_key, ffn_cls.down_proj_rkey, sep="_")
        add_up_key = add_up_rkey = join_name(add_ffn_key, ffn_cls.up_proj_rkey, sep="_")
        add_down_key = add_down_rkey = join_name(add_ffn_key, ffn_cls.down_proj_rkey, sep="_")
        key_map[ffn_rkey].update({up_key, down_key})
        key_map[add_ffn_rkey].update({add_up_key, add_down_key})
        key_map[up_rkey].add(up_key)
        key_map[down_rkey].add(down_key)
        key_map[add_up_rkey].add(add_up_key)
        key_map[add_down_rkey].add(add_down_key)

        return {k: v for k, v in key_map.items() if v}

    @staticmethod
    def _default_construct(
        module: nn.Module,
        /,
        parent: tp.Optional["WanDiTStruct"] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "WanTransformerBlockStruct":
        parallel = False
        norm_type = "ada_norm_mod"
        add_norm_type = "layer_norm"

        # norm1 → pre-norm for self_attn, norm3 → pre-norm for cross_attn
        pre_attn_norms = [module.norm1, module.norm3]
        pre_attn_norm_rnames = ["norm1", "norm3"]
        # pre_attn_add_norms: must match attns length for assertion in
        # TransformerBlockStruct.__post_init__; None = no separate encoder norm
        pre_attn_add_norms = [None, None]
        pre_attn_add_norm_rnames = ["self_attn.norm_cross", "cross_attn.norm_cross"]

        self_attn = module.self_attn
        cross_attn = module.cross_attn
        attns = [self_attn, cross_attn]
        attn_rnames = ["self_attn", "cross_attn"]

        pre_ffn_norm = module.norm2
        pre_ffn_norm_rname = "norm2"
        ffn = module.ffn
        ffn_rname = "ffn"
        pre_add_ffn_norm = None
        pre_add_ffn_norm_rname = ""
        add_ffn = None
        add_ffn_rname = ""

        return WanTransformerBlockStruct(
            module=module, parent=parent, fname=fname, idx=idx,
            rname=rname, rkey=rkey,
            parallel=parallel,
            norm_type=norm_type, add_norm_type=add_norm_type,
            pre_attn_norms=pre_attn_norms, attns=attns,
            pre_ffn_norm=pre_ffn_norm, ffn=ffn,
            pre_attn_add_norms=pre_attn_add_norms,
            pre_add_ffn_norm=pre_add_ffn_norm,
            add_ffn=add_ffn,
            pre_attn_norm_rnames=pre_attn_norm_rnames,
            attn_rnames=attn_rnames,
            pre_ffn_norm_rname=pre_ffn_norm_rname,
            ffn_rname=ffn_rname,
            pre_attn_add_norm_rnames=pre_attn_add_norm_rnames,
            pre_add_ffn_norm_rname=pre_add_ffn_norm_rname,
            add_ffn_rname=add_ffn_rname,
        )


@dataclass(kw_only=True)
class WanDiTStruct(DiffusionModelStruct):
    """Top-level structure for WanModel (DiffSynth-Studio).

    Inherits DiffusionModelStruct so ptq() accepts it directly.
    """

    input_embed_rkey: tp.ClassVar[str] = "input_embed"
    time_embed_rkey: tp.ClassVar[str] = "time_embed"
    text_embed_rkey: tp.ClassVar[str] = "text_embed"
    output_embed_rkey: tp.ClassVar[str] = "output_embed"
    transformer_block_rkey: tp.ClassVar[str] = ""
    transformer_block_struct_cls: tp.ClassVar[type[WanTransformerBlockStruct]] = WanTransformerBlockStruct

    module: nn.Module = field(repr=False, kw_only=False)

    input_embed: nn.Module
    time_embed: nn.Module
    text_embed: nn.Module
    head: nn.Module
    blocks: nn.ModuleList = field(repr=False)

    input_embed_rname: str
    time_embed_rname: str
    text_embed_rname: str
    head_rname: str
    blocks_rname: str

    input_embed_name: str = field(init=False, repr=False)
    time_embed_name: str = field(init=False, repr=False)
    text_embed_name: str = field(init=False, repr=False)
    head_name: str = field(init=False, repr=False)

    input_embed_key: str = field(init=False, repr=False)
    time_embed_key: str = field(init=False, repr=False)
    text_embed_key: str = field(init=False, repr=False)
    head_key: str = field(init=False, repr=False)

    block_structs_list: list[WanTransformerBlockStruct] = field(init=False, repr=False)
    # Must be named ``block_names`` (not ``block_names_list``): BaseModuleStruct
    # validates child ``fname="block"`` via ``parent.block_names[idx]``.
    block_names: list[str] = field(init=False, repr=False)

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def block_structs(self) -> list[WanTransformerBlockStruct]:
        return self.block_structs_list

    def __post_init__(self) -> None:
        if self.parent is None:
            self.name = ""
            self.key = ""
        else:
            self.name = join_name(self.parent.name, self.rname)
            self.key = join_name(self.parent.key, self.rkey, sep="_")

        self.pre_module_structs = OrderedDict()
        for fname, rkey in [
            ("input_embed", self.input_embed_rkey),
            ("time_embed", self.time_embed_rkey),
            ("text_embed", self.text_embed_rkey),
        ]:
            module = getattr(self, fname)
            rname = getattr(self, f"{fname}_rname")
            setattr(self, f"{fname}_key", join_name(self.key, rkey, sep="_"))
            setattr(self, f"{fname}_name", join_name(self.name, rname))
            if module is not None:
                self.pre_module_structs[getattr(self, f"{fname}_name")] = DiffusionModuleStruct(
                    module=module, parent=self, fname=fname, rname=rname, rkey=rkey
                )

        self.post_module_structs = OrderedDict()
        self.head_key = join_name(self.key, self.output_embed_rkey, sep="_")
        self.head_name = join_name(self.name, self.head_rname)
        if self.head is not None:
            self.post_module_structs[self.head_name] = DiffusionModuleStruct(
                module=self.head, parent=self, fname="head", rname=self.head_rname, rkey=self.output_embed_rkey
            )

        block_rnames = [f"{self.blocks_rname}.{i}" for i in range(len(self.blocks))]
        self.block_names = [join_name(self.name, rn) for rn in block_rnames]
        self.block_structs_list = [
            self.transformer_block_struct_cls.construct(
                block, parent=self, fname="block", rname=rn,
                rkey=self.transformer_block_rkey, idx=i,
            )
            for i, (block, rn) in enumerate(zip(self.blocks, block_rnames, strict=True))
        ]

    def get_prev_module_keys(self) -> tuple[str, ...]:
        return (self.input_embed_key, self.time_embed_key, self.text_embed_key)

    def get_post_module_keys(self) -> tuple[str, ...]:
        return (self.head_key,)

    def iter_attention_structs(self) -> tp.Generator[WanAttentionStruct, None, None]:
        for block_struct in self.block_structs_list:
            yield from block_struct.iter_attention_structs()

    def iter_transformer_block_structs(self) -> tp.Generator[WanTransformerBlockStruct, None, None]:
        for block_struct in self.block_structs_list:
            yield from block_struct.iter_transformer_block_structs()

    def _get_iter_block_activations_args(
        self, **input_kwargs
    ) -> tuple[list[nn.Module], list[DiffusionModuleStruct | DiffusionBlockStruct], list[bool], list[bool]]:
        layers = list(self.blocks)
        layer_structs = list(self.block_structs_list)
        use_prev_layer_outputs = [False] + [True] * (len(self.blocks) - 1)
        recomputes = [False] * len(self.blocks)
        return layers, layer_structs, recomputes, use_prev_layer_outputs

    @staticmethod
    def _default_construct(
        module: nn.Module,
        /,
        parent: tp.Optional[BaseModuleStruct] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "WanDiTStruct":
        return WanDiTStruct(
            module=module, parent=parent, fname=fname, idx=idx,
            rname=rname, rkey=rkey,
            input_embed=module.patch_embedding,
            time_embed=module.time_embedding,
            text_embed=module.text_embedding,
            head=module.head,
            blocks=module.blocks,
            input_embed_rname="patch_embedding",
            time_embed_rname="time_embedding",
            text_embed_rname="text_embedding",
            head_rname="head",
            blocks_rname="blocks",
        )

    @classmethod
    def _get_default_key_map(cls) -> dict[str, set[str]]:
        key_map: dict[str, set[str]] = defaultdict(set)
        block_rkey = cls.transformer_block_rkey
        block_cls = cls.transformer_block_struct_cls
        block_key_map = block_cls._get_default_key_map()
        for rkey, keys in block_key_map.items():
            brkey = join_name(block_rkey, rkey, sep="_")
            for key in keys:
                key = join_name(block_rkey, key, sep="_") if block_rkey else key
                key_map[rkey].add(key)
                if brkey != rkey:
                    key_map[brkey].add(key)
                if block_rkey:
                    key_map[block_rkey].add(key)
        keys: set[str] = set()
        keys.add(cls.input_embed_rkey)
        keys.add(cls.time_embed_rkey)
        keys.add(cls.text_embed_rkey)
        keys.add(cls.output_embed_rkey)
        for mapped_keys in key_map.values():
            for key in mapped_keys:
                keys.add(key)
        if "embed" not in keys and "embed" not in key_map:
            key_map["embed"].add(cls.input_embed_rkey)
            key_map["embed"].add(cls.time_embed_rkey)
            key_map["embed"].add(cls.text_embed_rkey)
            key_map["embed"].add(cls.output_embed_rkey)
        for key in keys:
            if key in key_map:
                key_map[key].clear()
            key_map[key].add(key)
        return {k: v for k, v in key_map.items() if v}


@dataclass(kw_only=True)
class VaceWanTransformerBlockStruct(WanTransformerBlockStruct):
    """Structure for a VACE side-branch transformer block.

    ``VaceWanAttentionBlock`` inherits the regular Wan ``DiTBlock`` and adds
    per-branch projection layers around the inherited attention/FFN body.
    """

    before_proj_rkey: tp.ClassVar[str] = "vace_before_proj"
    after_proj_rkey: tp.ClassVar[str] = "vace_after_proj"

    parent: tp.Optional["VaceWanDiTStruct"] = field(repr=False)
    before_proj: nn.Linear | None = field(default=None, repr=False)
    after_proj: nn.Linear | None = field(default=None, repr=False)
    before_proj_rname: str = ""
    after_proj_rname: str = "after_proj"
    before_proj_name: str = field(init=False, repr=False)
    after_proj_name: str = field(init=False, repr=False)
    before_proj_key: str = field(init=False, repr=False)
    after_proj_key: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.before_proj_name = join_name(self.name, self.before_proj_rname)
        self.after_proj_name = join_name(self.name, self.after_proj_rname)
        self.before_proj_key = join_name(self.key, self.before_proj_rkey, sep="_")
        self.after_proj_key = join_name(self.key, self.after_proj_rkey, sep="_")

    def named_key_modules(self) -> tp.Generator[tp.Tuple[str, str, nn.Module, BaseModuleStruct, str], None, None]:
        yield from super().named_key_modules()
        if self.before_proj is not None:
            yield self.before_proj_key, self.before_proj_name, self.before_proj, self, "before_proj"
        if self.after_proj is not None:
            yield self.after_proj_key, self.after_proj_name, self.after_proj, self, "after_proj"

    @classmethod
    def _get_default_key_map(cls) -> dict[str, set[str]]:
        key_map: dict[str, set[str]] = defaultdict(set)
        for rkey, keys in super()._get_default_key_map().items():
            key_map[rkey].update(keys)
        for key in (cls.before_proj_rkey, cls.after_proj_rkey):
            key_map[key].add(key)
        key_map["vace_proj"].update({cls.before_proj_rkey, cls.after_proj_rkey})
        return {k: v for k, v in key_map.items() if v}

    @staticmethod
    def _default_construct(
        module: nn.Module,
        /,
        parent: tp.Optional["VaceWanDiTStruct"] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "VaceWanTransformerBlockStruct":
        parallel = False
        norm_type = "ada_norm_mod"
        add_norm_type = "layer_norm"

        pre_attn_norms = [module.norm1, module.norm3]
        pre_attn_norm_rnames = ["norm1", "norm3"]
        pre_attn_add_norms = [None, None]
        pre_attn_add_norm_rnames = ["self_attn.norm_cross", "cross_attn.norm_cross"]

        attns = [module.self_attn, module.cross_attn]
        attn_rnames = ["self_attn", "cross_attn"]

        return VaceWanTransformerBlockStruct(
            module=module, parent=parent, fname=fname, idx=idx,
            rname=rname, rkey=rkey,
            parallel=parallel,
            norm_type=norm_type, add_norm_type=add_norm_type,
            pre_attn_norms=pre_attn_norms, attns=attns,
            pre_ffn_norm=module.norm2, ffn=module.ffn,
            pre_attn_add_norms=pre_attn_add_norms,
            pre_add_ffn_norm=None,
            add_ffn=None,
            pre_attn_norm_rnames=pre_attn_norm_rnames,
            attn_rnames=attn_rnames,
            pre_ffn_norm_rname="norm2",
            ffn_rname="ffn",
            pre_attn_add_norm_rnames=pre_attn_add_norm_rnames,
            pre_add_ffn_norm_rname="",
            add_ffn_rname="",
            before_proj=getattr(module, "before_proj", None),
            after_proj=getattr(module, "after_proj", None),
            before_proj_rname="before_proj" if hasattr(module, "before_proj") else "",
            after_proj_rname="after_proj",
        )


@dataclass(kw_only=True)
class VaceWanDiTStruct(WanDiTStruct):
    """Top-level structure for the VACE side branch."""

    input_embed_rkey: tp.ClassVar[str] = "vace_input_embed"
    time_embed_rkey: tp.ClassVar[str] = "vace_time_embed"
    text_embed_rkey: tp.ClassVar[str] = "vace_text_embed"
    output_embed_rkey: tp.ClassVar[str] = "vace_output_embed"
    transformer_block_rkey: tp.ClassVar[str] = "vace"
    transformer_block_struct_cls: tp.ClassVar[type[VaceWanTransformerBlockStruct]] = VaceWanTransformerBlockStruct

    dit: nn.Module | None = field(default=None, repr=False)

    def get_prev_module_keys(self) -> tuple[str, ...]:
        return (self.input_embed_key,)

    def get_post_module_keys(self) -> tuple[str, ...]:
        return ()

    @staticmethod
    def _default_construct(
        module: nn.Module,
        /,
        parent: tp.Optional[BaseModuleStruct] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "VaceWanDiTStruct":
        return VaceWanDiTStruct(
            module=module, parent=parent, fname=fname, idx=idx,
            rname=rname, rkey=rkey,
            input_embed=module.vace_patch_embedding,
            time_embed=None,
            text_embed=None,
            head=None,
            blocks=module.vace_blocks,
            input_embed_rname="vace_patch_embedding",
            time_embed_rname="",
            text_embed_rname="",
            head_rname="",
            blocks_rname="vace_blocks",
            dit=kwargs.get("dit", None),
        )


def _register_wan_factories():
    try:
        from diffsynth.models.wan_video_dit import (
            CrossAttention,
            DiTBlock,
            SelfAttention,
            WanModel,
        )
    except ImportError:
        return

    WanAttentionStruct.register_factory(
        (SelfAttention, CrossAttention), WanAttentionStruct._default_construct
    )
    WanFeedForwardStruct.register_factory(nn.Sequential, WanFeedForwardStruct._default_construct)
    WanTransformerBlockStruct.register_factory(DiTBlock, WanTransformerBlockStruct._default_construct)
    WanDiTStruct.register_factory(WanModel, WanDiTStruct._default_construct)
    try:
        from diffsynth.models.wan_video_vace import VaceWanAttentionBlock, VaceWanModel
    except ImportError:
        return
    VaceWanTransformerBlockStruct.register_factory(
        VaceWanAttentionBlock, VaceWanTransformerBlockStruct._default_construct
    )
    VaceWanDiTStruct.register_factory(VaceWanModel, VaceWanDiTStruct._default_construct)


_register_wan_factories()
