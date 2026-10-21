//! Qwen3.8-Flash-Next (GGUF arch "qwen4exp"): qwen35's MoE trunk and
//! GatedDeltaNet (the same converter tiling, so the same transforms) inside
//! hyper-connection streams, a sparse-attention indexer on the full-attention
//! layers and an n-gram per-layer embedding (PLE) at one layer. Its table is
//! the file's `per_layer_token_embd`, which mlx-serve reads row by row off the
//! shard and never loads.
const std = @import("std");
const gguf = @import("../gguf.zig");
const arch = @import("../arch.zig");
const qwen35 = arch.qwen35;
const Transform = arch.Transform;

pub const arch_name = "qwen4exp";

/// Names follow the native mlx-serve packs of the model.
const prefix = "language_model.model.";

/// mlx-serve's n-gram hash knobs the GGUF does not state. Its head vocab sizes
/// are the primes after the base (mlx-serve checks them against the id-math
/// tensors built from `metadataTensor`), its table rows padded to the divisor
/// (checked here against the table).
const ngram_vocab_base = 20_000_000;
const ngram_rows_divisor = 128;
const split_ngram_parts = 128;

pub const VisionTokens = struct { image: u32, video: u32, start: u32, end: u32 };

pub const Hparams = struct {
    base: qwen35.Hparams,
    hc_count: u32,
    hc_lowrank: u32,
    indexer_heads: u32,
    indexer_head_dim: u32,
    indexer_budget: u32,
    compress_ratio: u32,
    /// The GGUF block that holds the PLE (HF's `ple_layer_ids` count from 1).
    ple_layer: u32,
    ngram_size: u32,
    heads_per_ngram: u32,
    ple_conv_kernel: u32,
    /// One n-gram head's embedding width: a table row.
    ple_head_dim: u32,
    table_rows: u64,
    /// The n-gram hash's eos (HF's `text_config.eos_token_id`), not the chat terminator.
    ngram_eos: u32,
    bos: u32,
    vision: VisionTokens,

    pub fn read(f: *const gguf.File) !Hparams {
        const base = try qwen35.Hparams.read(f, arch_name);
        if (base.n_experts == 0) return error.BadMetadata;
        const k = qwen35.Keys{ .f = f, .arch_name = arch_name };
        const hp = Hparams{
            .base = base,
            .hc_count = try k.int("hyper_connection.count"),
            .hc_lowrank = try k.int("hyper_connection.low_rank"),
            .indexer_heads = try k.int("attention.indexer.head_count"),
            .indexer_head_dim = try k.int("attention.indexer.key_length"),
            .indexer_budget = try k.int("attention.indexer.top_k"),
            .compress_ratio = try compressRatio(k.array("attention.compress_ratios") orelse return error.MissingMetadata, base),
            .ple_layer = try pleLayer(k.array("ple.layers") orelse return error.MissingMetadata, base.n_layers),
            .ngram_size = try k.int("ple.ngram_size"),
            .heads_per_ngram = try k.int("ple.heads_per_ngram"),
            .ple_conv_kernel = try k.int("ple.conv_kernel"),
            .ple_head_dim = try k.int("embedding_length_per_layer_input"),
            .ngram_eos = try k.int("ple.eos_token_id"),
            .bos = try arch.u32Key(f, "tokenizer.ggml.bos_token_id"),
            .vision = try visionTokens(f),
            .table_rows = (f.tensors.get("per_layer_token_embd.weight") orelse return error.MissingMetadata).dims[1],
        };
        if (hp.vision.image != try k.int("ple.image_token_id")) return error.BadMetadata;
        const table = f.tensors.get("per_layer_token_embd.weight").?;
        // mlx-serve's table arm reads IQ4_NL rows.
        if (table.n_dims != 2 or table.dims[0] != hp.ple_head_dim or table.ty != .iq4_nl) return error.BadMetadata;
        const ngram = NgramMeta{
            .multipliers = k.array("ple.layer_multipliers") orelse return error.MissingMetadata,
            .offsets = k.array("ple.head_offsets") orelse return error.MissingMetadata,
            .vocab_sizes = k.array("ple.head_vocab_sizes") orelse return error.MissingMetadata,
        };
        try ngram.check(hp, table.dims[1]);
        return hp;
    }

    /// N-gram heads, refused when the header's counts overflow.
    fn nHeads(self: Hparams) !u32 {
        if (self.ngram_size < 2) return error.BadMetadata;
        return std.math.mul(u32, self.ngram_size - 1, self.heads_per_ngram) catch error.BadMetadata;
    }
};

/// Per layer, 0 on the GatedDeltaNet layers and one value on every full-attention layer.
fn compressRatio(ratios: gguf.Array, base: qwen35.Hparams) !u32 {
    if (!ratios.isInt() or ratios.len != base.n_layers or base.full_attn_interval == 0) return error.BadMetadata;
    var ratio: i64 = 0;
    for (0..ratios.len) |i| {
        const v = ratios.int(i);
        if ((i + 1) % base.full_attn_interval != 0) {
            if (v != 0) return error.BadMetadata;
        } else if (v <= 0 or (ratio != 0 and v != ratio)) return error.BadMetadata else ratio = v;
    }
    return std.math.cast(u32, ratio) orelse error.BadMetadata;
}

/// mlx-serve serves one PLE injection point.
fn pleLayer(layers: gguf.Array, n_layers: u32) !u32 {
    if (!layers.isInt() or layers.len != 1) return error.BadMetadata;
    const layer = std.math.cast(u32, layers.int(0)) orelse return error.BadMetadata;
    return if (layer < n_layers) layer else error.BadMetadata;
}

fn visionTokens(f: *const gguf.File) !VisionTokens {
    const names = [_][]const u8{ "<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>" };
    var ids: [names.len]?u32 = @splat(null);
    var it = (f.getArray("tokenizer.ggml.tokens") orelse return error.MissingMetadata).strings();
    var id: u32 = 0;
    while (it.next()) |tok| : (id += 1) {
        for (names, &ids) |n, *slot| if (std.mem.eql(u8, tok, n)) {
            slot.* = id;
        };
    }
    return .{
        .image = ids[0] orelse return error.MissingMetadata,
        .video = ids[1] orelse return error.MissingMetadata,
        .start = ids[2] orelse return error.MissingMetadata,
        .end = ids[3] orelse return error.MissingMetadata,
    };
}

/// The n-gram hash as the GGUF states it: one multiplier per n-gram position,
/// per-head vocab sizes, their running sums as offsets, the table padded to
/// `ngram_rows_divisor` rows.
const NgramMeta = struct {
    multipliers: gguf.Array,
    offsets: gguf.Array,
    vocab_sizes: gguf.Array,

    fn check(self: NgramMeta, hp: Hparams, table_rows: u64) !void {
        if (hp.heads_per_ngram == 0) return error.BadMetadata;
        const n = try hp.nHeads();
        if (!self.multipliers.isInt() or !self.offsets.isInt() or !self.vocab_sizes.isInt()) return error.BadMetadata;
        if (self.multipliers.len != hp.ngram_size or self.offsets.len != n or self.vocab_sizes.len != n) return error.BadMetadata;
        var total: u64 = 0;
        for (0..n) |i| {
            const size = std.math.cast(u64, self.vocab_sizes.int(i)) orelse return error.BadMetadata;
            if (size == 0 or self.offsets.int(i) != std.math.cast(i64, total)) return error.BadMetadata;
            total = std.math.add(u64, total, size) catch return error.BadMetadata;
        }
        const padded = std.math.add(u64, total, ngram_rows_divisor - 1) catch return error.BadMetadata;
        if (table_rows != padded / ngram_rows_divisor * ngram_rows_divisor) return error.BadMetadata;
    }
};

/// Where the n-gram table's rows sit: mlx-serve reads them in place off this file.
pub const TableFile = struct { path: []const u8, offset: u64 };

pub fn tableFile(f: *const gguf.File) !TableFile {
    const at = f.locate(f.tensors.get("per_layer_token_embd.weight") orelse return error.MissingMetadata) orelse return error.MissingMetadata;
    return .{ .path = at.path, .offset = at.offset };
}

/// The config.json of the native packs, without the MTP head (the GGUF has
/// none) and the vision tower (the mmproj's), plus where the table is.
pub fn configJson(allocator: std.mem.Allocator, hp: Hparams, table: TableFile) ![]u8 {
    const b = hp.base;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.print(allocator,
        \\{{"ngram_table":{{"file":{f},"format":"gguf_iq4_nl","offset":{d},"rows":{d},"dim":{d}}},
    , .{ std.json.fmt(table.path, .{}), table.offset, hp.table_rows, hp.ple_head_dim });
    try out.print(allocator,
        \\"architectures":["Qwen4ExpForConditionalGeneration"],"model_type":"qwen4_exp","language_model_only":false,
        \\"tie_word_embeddings":{[tied]},"image_token_id":{[image]d},"video_token_id":{[video]d},
        \\"vision_start_token_id":{[vstart]d},"vision_end_token_id":{[vend]d},
        \\"quantization":{{"group_size":32,"bits":4,"mode":"gguf"}},
        \\"text_config":{{"model_type":"qwen4_exp_text","attention_bias":false,"attention_dropout":0.0,"dtype":"bfloat16",
        \\"bos_token_id":{[bos]d},"eos_token_id":{[eos]d},"pad_token_id":null,"hidden_act":"silu","initializer_range":0.02,
        \\"hidden_size":{[hidden]d},"num_hidden_layers":{[n_layers]d},"layer_types":[
    , .{
        .tied = b.tied,
        .image = hp.vision.image,
        .video = hp.vision.video,
        .vstart = hp.vision.start,
        .vend = hp.vision.end,
        .bos = hp.bos,
        .eos = hp.ngram_eos,
        .hidden = b.hidden,
        .n_layers = b.n_layers,
    });
    for (0..b.n_layers) |i| {
        if (i > 0) try out.append(allocator, ',');
        try out.appendSlice(allocator, if ((i + 1) % b.full_attn_interval == 0) "\"full_attention\"" else "\"linear_attention\"");
    }
    const rotary = @as(f64, @floatFromInt(b.rope_dims)) / @as(f64, @floatFromInt(b.head_dim));
    try out.print(allocator,
        \\],"full_attention_interval":{[interval]d},"num_attention_heads":{[n_heads]d},"num_key_value_heads":{[n_kv]d},
        \\"head_dim":{[head_dim]d},"max_position_embeddings":{[ctx]d},"rms_norm_eps":{[eps]e},
        \\"partial_rotary_factor":{[rotary]d},"rope_parameters":{{"mrope_interleaved":true,"mrope_section":[{[m0]d},{[m1]d},{[m2]d}],
        \\"partial_rotary_factor":{[rotary]d},"rope_theta":{[theta]d},"rope_type":"default"}},
        \\"linear_conv_kernel_dim":{[conv]d},"linear_key_head_dim":{[dk]d},"linear_num_key_heads":{[nk]d},
        \\"linear_num_value_heads":{[nv]d},"linear_value_head_dim":{[dv]d},"mamba_ssm_dtype":"float32","output_gate_type":"sigmoid",
        \\"num_experts":{[experts]d},"num_experts_per_tok":{[topk]d},"moe_intermediate_size":{[expert_ffn]d},
        \\"shared_expert_intermediate_size":{[shared_ffn]d},"output_router_logits":false,"router_aux_loss_coef":0.001,
    , .{
        .interval = b.full_attn_interval,
        .n_heads = b.n_heads,
        .n_kv = b.n_kv_heads,
        .head_dim = b.head_dim,
        .ctx = b.ctx_len,
        // f32 in the file: printed as f32, 1e-6 and not 9.999999974752427e-7.
        .eps = @as(f32, @floatCast(b.rms_eps)),
        .rotary = rotary,
        .m0 = b.mrope[0],
        .m1 = b.mrope[1],
        .m2 = b.mrope[2],
        .theta = b.rope_theta,
        .conv = b.conv_kernel,
        .dk = b.dk,
        .nk = b.nk,
        .nv = b.nv,
        .dv = b.dv,
        .experts = b.n_experts,
        .topk = b.n_experts_used,
        .expert_ffn = b.expert_ffn,
        .shared_ffn = b.shared_ffn,
    });
    try out.print(allocator,
        \\"hc_count":{[hc]d},"hc_lowrank":{[hc_low]d},
        \\"indexer_n_heads":{[idx_heads]d},"indexer_kv_heads":1,"indexer_head_dim":{[idx_dim]d},"indexer_budget":{[idx_budget]d},
        \\"indexer_compress_ratio":{[ratio]d},
        \\"ple_layer_ids":[{[ple_layer]d}],"ple_embed_dim":{[ple_dim]d},"ple_conv_kernel_size":{[ple_conv]d},"ngram_size":{[ngram]d},
        \\"heads_per_ngram":{[hpn]d},"ngram_vocab_size_base":{[vocab_base]d},"make_ngram_vocab_size_divisible_by":{[divisor]d},
        \\"split_ngram_parts":{[parts]d},"tie_word_embeddings":{[tied]},"use_cache":true,"vocab_size":{[vocab]d}}}}}
    , .{
        .hc = hp.hc_count,
        .hc_low = hp.hc_lowrank,
        .idx_heads = hp.indexer_heads,
        .idx_dim = hp.indexer_head_dim,
        .idx_budget = hp.indexer_budget,
        .ratio = hp.compress_ratio,
        .ple_layer = hp.ple_layer + 1,
        .ple_dim = std.math.mul(u32, try hp.nHeads(), hp.ple_head_dim) catch return error.BadMetadata,
        .ple_conv = hp.ple_conv_kernel,
        .ngram = hp.ngram_size,
        .hpn = hp.heads_per_ngram,
        .vocab_base = ngram_vocab_base,
        .divisor = ngram_rows_divisor,
        .parts = split_ngram_parts,
        .tied = b.tied,
        .vocab = b.vocab,
    });
    return out.toOwnedSlice(allocator);
}

/// HF name + load transform for a GGUF tensor, null when the tensor is unknown.
pub fn mapTensor(buf: []u8, name: []const u8, hp: Hparams) ?arch.Mapped {
    const globals = [_]struct { []const u8, []const u8 }{
        .{ "output_hc_up.weight", prefix ++ "hyper_connection_mixer.input_mix_weight_up.weight" },
        .{ "output_hc_down.weight", prefix ++ "hyper_connection_mixer.input_mix_weight_down.weight" },
        .{ "output_hc_norm.weight", prefix ++ "hyper_connection_mixer.hc_norm.weight" },
    };
    for (globals) |g| if (std.mem.eql(u8, name, g[0])) return .{ .name = g[1], .transform = .none };
    if (std.mem.eql(u8, name, "per_layer_token_embd.weight")) return arch.skip;
    // qwen35's map skips the block past the trunk as an MTP head; this GGUF has none, so it is unknown.
    if (arch.splitLayer(name, hp.base.n_layers + 1)) |b| if (b.layer == hp.base.n_layers) return null;
    const blk = arch.splitLayer(name, hp.base.n_layers) orelse return qwen35.mapTensor(buf, name, hp.base);

    const Layers = enum { all, full_attention, ple };
    const qk = struct {
        fn part(p: u8) Transform {
            return .{ .join_rows = .{ .part = p, .parts = 2 } };
        }
    }.part;
    const table = [_]struct { []const u8, []const u8, Transform, Layers }{
        .{ "hc_attn_up.weight", "attn_hyper_connection.input_mix_weight_up.weight", .none, .all },
        .{ "hc_attn_down.weight", "attn_hyper_connection.input_mix_weight_down.weight", .none, .all },
        .{ "hc_attn_inject.weight", "attn_hyper_connection.block_inject_weight.weight", .none, .all },
        .{ "hc_attn_norm.weight", "attn_hyper_connection.hc_norm.weight", .none, .all },
        .{ "hc_ffn_up.weight", "mlp_hyper_connection.input_mix_weight_up.weight", .none, .all },
        .{ "hc_ffn_down.weight", "mlp_hyper_connection.input_mix_weight_down.weight", .none, .all },
        .{ "hc_ffn_inject.weight", "mlp_hyper_connection.block_inject_weight.weight", .none, .all },
        .{ "hc_ffn_norm.weight", "mlp_hyper_connection.hc_norm.weight", .none, .all },
        // HF fuses the indexer's projections, q rows then k rows.
        .{ "indexer.q_proj.weight", "self_attn.indexer.index_qk_proj.weight", qk(0), .full_attention },
        .{ "indexer.k_proj.weight", "self_attn.indexer.index_qk_proj.weight", qk(1), .full_attention },
        .{ "indexer.q_norm.weight", "self_attn.indexer.q_layernorm.weight", .none, .full_attention },
        .{ "indexer.k_norm.weight", "self_attn.indexer.k_layernorm.weight", .none, .full_attention },
        .{ "ple_conv1d.weight", "ple.conv1d.weight", .trailing_axis, .ple },
        .{ "ple_key.weight", "ple.key_proj.weight", .none, .ple },
        .{ "ple_value.weight", "ple.value_proj.weight", .none, .ple },
        .{ "ple_norm_conv.weight", "ple.norm_conv.weight", .none, .ple },
        .{ "ple_norm_key.weight", "ple.norm_key.weight", .none, .ple },
        .{ "ple_norm_query.weight", "ple.norm_query.weight", .none, .ple },
    };
    for (table) |row| if (std.mem.eql(u8, blk.leaf, row[0])) {
        const here = switch (row[3]) {
            .all => true,
            .full_attention => (blk.layer + 1) % hp.base.full_attn_interval == 0,
            .ple => blk.layer == hp.ple_layer,
        };
        if (!here) return null;
        const full = std.fmt.bufPrint(buf, prefix ++ "layers.{d}.{s}", .{ blk.layer, row[1] }) catch return null;
        return .{ .name = full, .transform = row[2] };
    };
    return qwen35.mapTensor(buf, name, hp.base);
}

/// The PLE's id math, which HF checkpoints carry as int64 tensors and the GGUF as metadata.
pub fn metadataTensor(buf: []u8, i: usize, hp: Hparams) ?arch.MetadataTensor {
    const leaves = [_]struct { []const u8, []const u8 }{
        .{ "layer_multipliers", "qwen4exp.ple.layer_multipliers" },
        .{ "ngram_heads_offsets", "qwen4exp.ple.head_offsets" },
        .{ "ngram_heads_vocab_sizes", "qwen4exp.ple.head_vocab_sizes" },
    };
    if (i >= leaves.len) return null;
    const name = std.fmt.bufPrint(buf, prefix ++ "layers.{d}.ple.ple_embedding.{s}", .{ hp.ple_layer, leaves[i][0] }) catch return null;
    return .{ .name = name, .key = leaves[i][1] };
}

/// Qwen3.8-Flash-Next's geometry, as its GGUF metadata states it.
fn testHparams() Hparams {
    var base = std.mem.zeroes(qwen35.Hparams);
    base.n_layers = 48;
    base.hidden = 2560;
    base.n_heads = 24;
    base.n_kv_heads = 2;
    base.head_dim = 256;
    base.rope_dims = 64;
    base.rope_theta = 1e7;
    base.rms_eps = 9.999999974752427e-07;
    base.ctx_len = 262144;
    base.full_attn_interval = 4;
    base.conv_kernel = 4;
    base.nk = 16;
    base.nv = 48;
    base.dk = 128;
    base.dv = 128;
    base.vocab = 248320;
    base.eos_id = 248046;
    base.mrope = .{ 11, 11, 10 };
    base.n_experts = 512;
    base.n_experts_used = 10;
    base.expert_ffn = 640;
    base.shared_ffn = 640;
    return .{
        .base = base,
        .hc_count = 4,
        .hc_lowrank = 320,
        .indexer_heads = 4,
        .indexer_head_dim = 128,
        .indexer_budget = 2048,
        .compress_ratio = 4,
        .ple_layer = 1,
        .ngram_size = 3,
        .heads_per_ngram = 8,
        .ple_conv_kernel = 4,
        .ple_head_dim = 160,
        .table_rows = 320001536,
        .ngram_eos = 248044,
        .bos = 248044,
        .vision = .{ .image = 248056, .video = 248057, .start = 248053, .end = 248054 },
    };
}

/// A GGUF scalar array of i64 values, as `gguf.File` hands them out.
fn testArray(buf: []u8, vals: []const i64) gguf.Array {
    for (vals, 0..) |v, i| std.mem.writeInt(i64, buf[8 * i ..][0..8], v, .little);
    return .{ .elem_type = .i64, .len = vals.len, .bytes = buf[0 .. 8 * vals.len] };
}

test "mapTensor: hyper-connections, the indexer and the PLE by layer kind, the trunk through qwen35" {
    const hp = testHparams();
    var buf: [160]u8 = undefined;
    const Want = struct { []const u8, []const u8 };
    for ([_]Want{
        .{ "output_hc_up.weight", "language_model.model.hyper_connection_mixer.input_mix_weight_up.weight" },
        .{ "output_hc_down.weight", "language_model.model.hyper_connection_mixer.input_mix_weight_down.weight" },
        .{ "output_hc_norm.weight", "language_model.model.hyper_connection_mixer.hc_norm.weight" },
        .{ "blk.0.hc_attn_inject.weight", "language_model.model.layers.0.attn_hyper_connection.block_inject_weight.weight" },
        .{ "blk.0.hc_attn_up.weight", "language_model.model.layers.0.attn_hyper_connection.input_mix_weight_up.weight" },
        .{ "blk.2.hc_ffn_down.weight", "language_model.model.layers.2.mlp_hyper_connection.input_mix_weight_down.weight" },
        .{ "blk.2.hc_ffn_norm.weight", "language_model.model.layers.2.mlp_hyper_connection.hc_norm.weight" },
        .{ "blk.3.indexer.q_norm.weight", "language_model.model.layers.3.self_attn.indexer.q_layernorm.weight" },
        .{ "blk.3.indexer.k_norm.weight", "language_model.model.layers.3.self_attn.indexer.k_layernorm.weight" },
        .{ "blk.1.ple_key.weight", "language_model.model.layers.1.ple.key_proj.weight" },
        .{ "blk.1.ple_value.weight", "language_model.model.layers.1.ple.value_proj.weight" },
        .{ "blk.1.ple_norm_conv.weight", "language_model.model.layers.1.ple.norm_conv.weight" },
        .{ "blk.1.ple_norm_key.weight", "language_model.model.layers.1.ple.norm_key.weight" },
        .{ "blk.1.ple_norm_query.weight", "language_model.model.layers.1.ple.norm_query.weight" },
        .{ "blk.3.attn_q.weight", "language_model.model.layers.3.self_attn.q_proj.weight" },
        .{ "blk.0.ffn_down_exps.weight", "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight" },
    }) |w| {
        const m = mapTensor(&buf, w[0], hp) orelse return error.TestExpectedMapped;
        try std.testing.expectEqualStrings(w[1], m.name);
        try std.testing.expect(m.transform == .none);
    }
    // q rows then k rows, the order the transformer splits index_qk_proj in.
    for ([_][]const u8{ "q_proj", "k_proj" }, 0..) |leaf, part| {
        const name = try std.fmt.allocPrint(std.testing.allocator, "blk.47.indexer.{s}.weight", .{leaf});
        defer std.testing.allocator.free(name);
        const m = mapTensor(&buf, name, hp).?;
        try std.testing.expectEqualStrings("language_model.model.layers.47.self_attn.indexer.index_qk_proj.weight", m.name);
        try std.testing.expectEqual(@as(u8, @intCast(part)), m.transform.join_rows.part);
        try std.testing.expectEqual(@as(u8, 2), m.transform.join_rows.parts);
    }
    const conv = mapTensor(&buf, "blk.1.ple_conv1d.weight", hp).?;
    try std.testing.expectEqualStrings("language_model.model.layers.1.ple.conv1d.weight", conv.name);
    try std.testing.expect(conv.transform == .trailing_axis);
    // The table is read off its shard by the n-gram arm, never loaded.
    try std.testing.expectEqualStrings("", mapTensor(&buf, "per_layer_token_embd.weight", hp).?.name);
    // The trunk keeps qwen35's un-tiling.
    try std.testing.expectEqual(@as(u32, 2 * 16 * 128), mapTensor(&buf, "blk.0.attn_qkv.weight", hp).?.transform.untile_rows.start);
    try std.testing.expect(mapTensor(&buf, "blk.0.ssm_out.weight", hp).?.transform == .tiled_input);
    // An indexer off a full-attention layer, the PLE off its layer, a layer past the trunk: unknown.
    // blk.48 too: qwen35's map skips that block as an MTP head, and this GGUF has none.
    for ([_][]const u8{ "blk.2.indexer.q_proj.weight", "blk.0.ple_key.weight", "blk.48.hc_attn_up.weight", "blk.48.nextn.eh_proj.weight", "blk.49.hc_attn_up.weight", "blk.1.hc_attn_gate.weight" }) |name| {
        try std.testing.expect(mapTensor(&buf, name, hp) == null);
    }
}

test "the indexer compress ratio: one non-zero value on exactly the full-attention layers" {
    var base = std.mem.zeroes(qwen35.Hparams);
    base.n_layers = 8;
    base.full_attn_interval = 4;
    var buf: [8 * 8]u8 = undefined;
    try std.testing.expectEqual(@as(u32, 4), try compressRatio(testArray(&buf, &.{ 0, 0, 0, 4, 0, 0, 0, 4 }), base));
    for ([_][]const i64{
        &.{ 0, 0, 0, 4, 0, 0, 0, 2 }, // two ratios
        &.{ 0, 0, 0, 4, 0, 0, 0, 0 }, // a full-attention layer without one
        &.{ 0, 4, 0, 4, 0, 0, 0, 4 }, // a GatedDeltaNet layer with one
        &.{ 0, 0, 0, 4, 0, 0, 0 }, // not one per layer
    }) |vals| try std.testing.expectError(error.BadMetadata, compressRatio(testArray(&buf, vals), base));
}

test "the n-gram metadata must be the hash mlx-serve derives: one multiplier per position, offsets the running sum, rows padded to 128" {
    var hp = testHparams();
    hp.ngram_size = 2;
    hp.heads_per_ngram = 2;
    var mb: [64]u8 = undefined;
    var ob: [64]u8 = undefined;
    var vb: [64]u8 = undefined;
    const good = NgramMeta{ .multipliers = testArray(&mb, &.{ 3, 5 }), .offsets = testArray(&ob, &.{ 0, 101 }), .vocab_sizes = testArray(&vb, &.{ 101, 103 }) };
    try good.check(hp, 256);
    try std.testing.expectError(error.BadMetadata, good.check(hp, 204)); // 204 rows: not 101 + 103 padded to 128
    const bad_offsets = NgramMeta{ .multipliers = good.multipliers, .offsets = testArray(&ob, &.{ 0, 100 }), .vocab_sizes = good.vocab_sizes };
    try std.testing.expectError(error.BadMetadata, bad_offsets.check(hp, 256));
    const short = NgramMeta{ .multipliers = testArray(&mb, &.{3}), .offsets = testArray(&ob, &.{ 0, 101 }), .vocab_sizes = good.vocab_sizes };
    try std.testing.expectError(error.BadMetadata, short.check(hp, 256));
    // A hand-made header whose sums or head count overflow is refused, not a panic.
    const huge = std.math.maxInt(i64);
    const overflow = NgramMeta{ .multipliers = testArray(&mb, &.{ 3, 5 }), .offsets = testArray(&ob, &.{ 0, huge }), .vocab_sizes = testArray(&vb, &.{ huge, huge }) };
    try std.testing.expectError(error.BadMetadata, overflow.check(hp, 256));
    hp.heads_per_ngram = 1 << 31;
    hp.ngram_size = 3;
    try std.testing.expectError(error.BadMetadata, good.check(hp, 256));
}

test "configJson carries the native pack's text_config, no MTP and no vision_config" {
    const json = try configJson(std.testing.allocator, testHparams(), .{ .path = "/m/x-00002-of-00002.gguf", .offset = 192 });
    defer std.testing.allocator.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    try std.testing.expectEqualStrings("qwen4_exp", root.get("model_type").?.string);
    try std.testing.expectEqualStrings("Qwen4ExpForConditionalGeneration", root.get("architectures").?.array.items[0].string);
    try std.testing.expectEqualStrings("gguf", root.get("quantization").?.object.get("mode").?.string);
    try std.testing.expectEqual(@as(i64, 248056), root.get("image_token_id").?.integer);
    try std.testing.expectEqual(@as(i64, 248057), root.get("video_token_id").?.integer);
    try std.testing.expectEqual(@as(i64, 248053), root.get("vision_start_token_id").?.integer);
    try std.testing.expectEqual(@as(i64, 248054), root.get("vision_end_token_id").?.integer);
    try std.testing.expect(root.get("vision_config") == null);
    // mlx-serve reads the n-gram table's IQ4_NL rows in place, off the shard that holds them.
    const table = root.get("ngram_table").?.object;
    try std.testing.expectEqualStrings("/m/x-00002-of-00002.gguf", table.get("file").?.string);
    try std.testing.expectEqualStrings("gguf_iq4_nl", table.get("format").?.string);
    try std.testing.expectEqual(@as(i64, 192), table.get("offset").?.integer);
    try std.testing.expectEqual(@as(i64, 320001536), table.get("rows").?.integer);
    try std.testing.expectEqual(@as(i64, 160), table.get("dim").?.integer);
    // The GGUF path uses norms as stored (folded, weights.zig's layout oracle): no marker.
    try std.testing.expect(root.get("qwen4_norm_convention") == null);
    const tc = root.get("text_config").?.object;
    try std.testing.expectEqualStrings("qwen4_exp_text", tc.get("model_type").?.string);
    // The n-gram hash's eos, not the chat terminator.
    try std.testing.expectEqual(@as(i64, 248044), tc.get("eos_token_id").?.integer);
    try std.testing.expectEqual(@as(i64, 2), tc.get("ple_layer_ids").?.array.items[0].integer);
    try std.testing.expectEqual(@as(i64, 2560), tc.get("ple_embed_dim").?.integer);
    try std.testing.expectEqual(@as(i64, 2048), tc.get("indexer_budget").?.integer);
    try std.testing.expectEqual(@as(i64, 4), tc.get("indexer_compress_ratio").?.integer);
    try std.testing.expectEqual(@as(usize, 48), tc.get("layer_types").?.array.items.len);
    try std.testing.expectEqualStrings("full_attention", tc.get("layer_types").?.array.items[47].string);
    try std.testing.expectEqualStrings("linear_attention", tc.get("layer_types").?.array.items[46].string);
    try std.testing.expectEqual(@as(f64, 0.25), tc.get("partial_rotary_factor").?.float);
    try std.testing.expect(tc.get("mtp") == null and tc.get("mtp_num_hidden_layers") == null);
}

test "metadataTensor: the PLE's id math as int64 vectors under the native names" {
    const hp = testHparams();
    var buf: [160]u8 = undefined;
    const Want = struct { []const u8, []const u8 };
    for ([_]Want{
        .{ "language_model.model.layers.1.ple.ple_embedding.layer_multipliers", "qwen4exp.ple.layer_multipliers" },
        .{ "language_model.model.layers.1.ple.ple_embedding.ngram_heads_offsets", "qwen4exp.ple.head_offsets" },
        .{ "language_model.model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes", "qwen4exp.ple.head_vocab_sizes" },
    }, 0..) |w, i| {
        const m = metadataTensor(&buf, i, hp).?;
        try std.testing.expectEqualStrings(w[0], m.name);
        try std.testing.expectEqualStrings(w[1], m.key);
    }
    try std.testing.expect(metadataTensor(&buf, 3, hp) == null);
}
