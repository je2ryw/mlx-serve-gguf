//! A llama.cpp `clip` mmproj GGUF holding a Qwen3-VL vision tower and its
//! merger (`clip.projector_type = qwen3vl_merger`), read the way mlx-serve's
//! Qwen3-VL ViT wants it: the HF `vision_config` and vision token ids for the
//! rebuilt config.json, and the GGUF -> `model.visual.*` tensor names. The
//! tower is dense (bf16 weights, f32 biases and norms), nothing is quantized.
const std = @import("std");
const gguf = @import("gguf.zig");
const GgmlType = @import("quants.zig").GgmlType;

const prefix = "model.visual.";

pub const Hparams = struct {
    depth: u32,
    hidden: u32,
    ffn: u32,
    heads: u32,
    patch: u32,
    merge: u32,
    out_hidden: u32,
    /// Slices of the patch embedding: `v.patch_embd.weight`, then `.weight.1`, ...
    temporal_patch: u32,
    /// Rows of `v.position_embd.weight`.
    num_pos: u32,

    pub fn read(f: *const gguf.File) !Hparams {
        const patch = f.tensors.get("v.patch_embd.weight") orelse return error.MissingMetadata;
        const pos = f.tensors.get("v.position_embd.weight") orelse return error.MissingMetadata;
        var temporal: u32 = 1;
        var name_buf: [64]u8 = undefined;
        while (f.tensors.get(std.fmt.bufPrint(&name_buf, "v.patch_embd.weight.{d}", .{temporal}) catch unreachable)) |slice| : (temporal += 1) {
            if (slice.ty != patch.ty or !std.mem.eql(u64, &slice.dims, &patch.dims)) return error.BadMetadata;
        }
        const hp = Hparams{
            .depth = try u32Key(f, "clip.vision.block_count"),
            .hidden = try u32Key(f, "clip.vision.embedding_length"),
            .ffn = try u32Key(f, "clip.vision.feed_forward_length"),
            .heads = try u32Key(f, "clip.vision.attention.head_count"),
            .patch = try u32Key(f, "clip.vision.patch_size"),
            .merge = try u32Key(f, "clip.vision.spatial_merge_size"),
            .out_hidden = try u32Key(f, "clip.vision.projection_dim"),
            .temporal_patch = temporal,
            .num_pos = std.math.cast(u32, pos.dims[1]) orelse return error.BadMetadata,
        };
        // ggml dims are innermost first: [kw, kh, channels, hidden] and [hidden, positions].
        const want_patch = [4]u64{ hp.patch, hp.patch, 3, hp.hidden };
        if (patch.n_dims != 4 or !std.mem.eql(u64, &patch.dims, &want_patch)) return error.BadMetadata;
        if (pos.n_dims != 2 or pos.dims[0] != hp.hidden) return error.BadMetadata;
        if (hp.depth == 0 or hp.heads == 0 or hp.hidden % hp.heads != 0 or hp.merge == 0) return error.BadMetadata;
        // The tower interpolates the table as a square grid of positions.
        const side: u32 = std.math.sqrt(hp.num_pos);
        if (side * side != hp.num_pos) return error.BadMetadata;
        return hp;
    }

    /// Tensors a complete tower holds: 12 per block, the patch slices + bias,
    /// the position table, the merger's norm and two linears.
    fn tensorCount(self: Hparams) usize {
        return @as(usize, self.depth) * 12 + self.temporal_patch + 1 + 1 + 2 + 4;
    }
};

fn u32Key(f: *const gguf.File, key: []const u8) !u32 {
    return std.math.cast(u32, f.getInt(key) orelse return error.MissingMetadata) orelse error.BadMetadata;
}

/// The vision token ids, found in the language model's vocab by name.
pub const TokenIds = struct {
    image: u32,
    video: u32,
    start: u32,
    end: u32,

    pub fn read(llm: *const gguf.File) ?TokenIds {
        const tokens = llm.getArray("tokenizer.ggml.tokens") orelse return null;
        const names = [_][]const u8{ "<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>" };
        var ids: [names.len]?u32 = @splat(null);
        var it = tokens.strings();
        var id: u32 = 0;
        while (it.next()) |tok| : (id += 1) {
            for (names, &ids) |n, *slot| if (slot.* == null and std.mem.eql(u8, tok, n)) {
                slot.* = id;
            };
        }
        return .{ .image = ids[0] orelse return null, .video = ids[1] orelse return null, .start = ids[2] orelse return null, .end = ids[3] orelse return null };
    }
};

/// Why `mm` gives `llm` no Qwen3-VL tower, null when it does. Text keeps
/// serving on a non-null answer, so a partial tower is never offered.
pub fn unsupportedReason(mm: *const gguf.File, llm: *const gguf.File, buf: []u8) ?[]const u8 {
    if (!std.mem.eql(u8, mm.getString("general.architecture") orelse "", "clip")) return "not a clip mmproj";
    if (!(mm.getBool("clip.has_vision_encoder") orelse false)) return "no vision encoder";
    const projector = mm.getString("clip.projector_type") orelse "";
    if (!std.mem.eql(u8, projector, "qwen3vl_merger")) return std.fmt.bufPrint(buf, "projector_type {s}", .{projector}) catch "projector_type";
    if (mm.getArray("clip.vision.is_deepstack_layers")) |ds| {
        if (ds.elem_type != .bool) return "is_deepstack_layers";
        for (0..ds.len) |i| if (ds.boolean(i)) return "deepstack layers (not ported)";
    }
    // Our tower runs GELU and normalizes pixels with mean = std = 0.5.
    if (!(mm.getBool("clip.use_gelu") orelse false)) return "not a GELU tower";
    for ([_][]const u8{ "clip.vision.image_mean", "clip.vision.image_std" }) |key| {
        const arr = mm.getArray(key) orelse return key;
        if (arr.elem_type != .f32 or arr.len != 3) return key;
        for (0..3) |i| if (arr.float(i) != 0.5) return key;
    }
    const hp = Hparams.read(mm) catch |err| return std.fmt.bufPrint(buf, "metadata ({s})", .{@errorName(err)}) catch "metadata";
    const arch = llm.getString("general.architecture") orelse return "model architecture";
    var key_buf: [96]u8 = undefined;
    const hidden = llm.getInt(std.fmt.bufPrint(&key_buf, "{s}.embedding_length", .{arch}) catch return "model hidden size") orelse return "model hidden size";
    if (hidden != hp.out_hidden) return std.fmt.bufPrint(buf, "projection_dim {d} is not the model's hidden size {d}", .{ hp.out_hidden, hidden }) catch "projection_dim";
    if (TokenIds.read(llm) == null) return "no vision tokens in the model's vocab";
    var name_buf: [160]u8 = undefined;
    var it = mm.tensors.iterator();
    while (it.next()) |e| {
        if (mapTensor(&name_buf, e.key_ptr.*, hp) == null) return std.fmt.bufPrint(buf, "tensor {s}", .{e.key_ptr.*}) catch "tensor";
        switch (e.value_ptr.ty) {
            .f32, .f16, .bf16 => {},
            else => return std.fmt.bufPrint(buf, "tensor type of {s} (the tower is dense)", .{e.key_ptr.*}) catch "tensor type",
        }
        const want = wantDims(e.key_ptr.*, hp);
        const t = e.value_ptr.*;
        if (t.n_dims != want.len or !std.mem.eql(u64, t.dims[0..t.n_dims], want.slice()))
            return std.fmt.bufPrint(buf, "shape of {s}", .{e.key_ptr.*}) catch "tensor shape";
    }
    if (mm.tensors.count() != hp.tensorCount()) return std.fmt.bufPrint(buf, "{d} tensors, a full tower has {d}", .{ mm.tensors.count(), hp.tensorCount() }) catch "tensor count";
    return null;
}

const Dims = struct {
    len: u32,
    d: [4]u64,

    fn of(dims: []const u64) Dims {
        var out = Dims{ .len = @intCast(dims.len), .d = @splat(1) };
        @memcpy(out.d[0..dims.len], dims);
        return out;
    }

    fn slice(self: *const Dims) []const u64 {
        return self.d[0..self.len];
    }
};

/// The ggml dims (innermost first) of a tensor `mapTensor` knows, for these hparams.
fn wantDims(name: []const u8, hp: Hparams) Dims {
    const h: u64 = hp.hidden;
    const merged = h * hp.merge * hp.merge;
    const of = Dims.of;
    if (std.mem.startsWith(u8, name, "v.patch_embd.weight")) return of(&.{ hp.patch, hp.patch, 3, h });
    if (std.mem.eql(u8, name, "v.position_embd.weight")) return of(&.{ h, hp.num_pos });
    if (std.mem.eql(u8, name, "mm.0.weight")) return of(&.{ merged, merged });
    if (std.mem.eql(u8, name, "mm.0.bias")) return of(&.{merged});
    if (std.mem.eql(u8, name, "mm.2.weight")) return of(&.{ merged, hp.out_hidden });
    if (std.mem.eql(u8, name, "mm.2.bias")) return of(&.{hp.out_hidden});
    // A block linear is [in, out] with an [out] bias; every other vector (norms, the patch bias) is [hidden].
    const linears = [_]struct { []const u8, u64, u64 }{
        .{ ".attn_qkv.", h, 3 * h }, .{ ".attn_out.", h, h }, .{ ".ffn_up.", h, hp.ffn }, .{ ".ffn_down.", hp.ffn, h },
    };
    for (linears) |l| if (std.mem.indexOf(u8, name, l[0]) != null)
        return if (std.mem.endsWith(u8, name, ".weight")) of(&.{ l[1], l[2] }) else of(&.{l[2]});
    return of(&.{h});
}

/// `"vision_config":{...}` and the four vision token ids: the top-level keys
/// an HF checkpoint of the same model carries, ready to splice into its
/// config.json. `model_type` is the language model's.
pub fn configFragment(allocator: std.mem.Allocator, hp: Hparams, ids: TokenIds, model_type: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator,
        \\"vision_config":{{"deepstack_visual_indexes":[],"depth":{d},"hidden_act":"gelu_pytorch_tanh","hidden_size":{d},
        \\"in_channels":3,"intermediate_size":{d},"model_type":"{s}","num_heads":{d},"num_position_embeddings":{d},
        \\"out_hidden_size":{d},"patch_size":{d},"spatial_merge_size":{d},"temporal_patch_size":{d}}},
        \\"image_token_id":{d},"video_token_id":{d},"vision_start_token_id":{d},"vision_end_token_id":{d}
    , .{ hp.depth, hp.hidden, hp.ffn, model_type, hp.heads, hp.num_pos, hp.out_hidden, hp.patch, hp.merge, hp.temporal_patch, ids.image, ids.video, ids.start, ids.end });
}

/// HF name of an mmproj tensor, "" for a patch-embedding slice after the
/// first (the loader stacks every slice into `patch_embed.proj.weight`),
/// null for anything a Qwen3-VL tower does not have.
pub fn mapTensor(buf: []u8, name: []const u8, hp: Hparams) ?[]const u8 {
    const globals = [_]struct { []const u8, []const u8 }{
        .{ "v.patch_embd.weight", prefix ++ "patch_embed.proj.weight" },
        .{ "v.patch_embd.bias", prefix ++ "patch_embed.proj.bias" },
        .{ "v.position_embd.weight", prefix ++ "pos_embed.weight" },
        .{ "v.post_ln.weight", prefix ++ "merger.norm.weight" },
        .{ "v.post_ln.bias", prefix ++ "merger.norm.bias" },
        .{ "mm.0.weight", prefix ++ "merger.linear_fc1.weight" },
        .{ "mm.0.bias", prefix ++ "merger.linear_fc1.bias" },
        .{ "mm.2.weight", prefix ++ "merger.linear_fc2.weight" },
        .{ "mm.2.bias", prefix ++ "merger.linear_fc2.bias" },
    };
    for (globals) |g| if (std.mem.eql(u8, name, g[0])) return g[1];

    const slice = "v.patch_embd.weight.";
    if (std.mem.startsWith(u8, name, slice)) {
        const t = std.fmt.parseInt(u32, name[slice.len..], 10) catch return null;
        return if (t >= 1 and t < hp.temporal_patch) "" else null;
    }

    if (!std.mem.startsWith(u8, name, "v.blk.")) return null;
    const rest = name["v.blk.".len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const layer = std.fmt.parseInt(u32, rest[0..dot], 10) catch return null;
    if (layer >= hp.depth) return null;
    const leaf = rest[dot + 1 ..];
    const end = std.mem.lastIndexOfScalar(u8, leaf, '.') orelse return null;
    const kind = leaf[end + 1 ..];
    if (!std.mem.eql(u8, kind, "weight") and !std.mem.eql(u8, kind, "bias")) return null;
    const table = [_]struct { []const u8, []const u8 }{
        .{ "attn_qkv", "attn.qkv" },
        .{ "attn_out", "attn.proj" },
        .{ "ffn_up", "mlp.linear_fc1" },
        .{ "ffn_down", "mlp.linear_fc2" },
        .{ "ln1", "norm1" },
        .{ "ln2", "norm2" },
    };
    for (table) |row| if (std.mem.eql(u8, leaf[0..end], row[0])) {
        return std.fmt.bufPrint(buf, prefix ++ "blocks.{d}.{s}.{s}", .{ layer, row[1], kind }) catch null;
    };
    return null;
}

// ---- Tiny GGUF files for tests (here and in mlx-serve's bridge) ----

/// The KV value kinds the test files use.
pub const TestKv = union(enum) {
    str: []const u8,
    u32: u32,
    f32: f32,
    bool: bool,
    f32s: []const f32,
    bools: []const bool,
    i32s: []const i32,
    strs: []const []const u8,
};

pub const TestTensor = struct { name: []const u8, dims: []const u64, ty: GgmlType, data: []const u8 };

/// A GGUF v3 file of `kvs` and `tensors` (data 32-byte aligned). Caller frees.
pub fn writeTestGguf(a: std.mem.Allocator, kvs: []const struct { []const u8, TestKv }, tensors: []const TestTensor) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    const W = struct {
        fn num(o: *std.ArrayList(u8), al: std.mem.Allocator, v: anytype) !void {
            try o.appendSlice(al, std.mem.asBytes(&v));
        }
        fn str(o: *std.ArrayList(u8), al: std.mem.Allocator, s: []const u8) !void {
            try num(o, al, @as(u64, s.len));
            try o.appendSlice(al, s);
        }
    };
    try out.appendSlice(a, "GGUF");
    try W.num(&out, a, @as(u32, 3));
    try W.num(&out, a, @as(u64, tensors.len));
    try W.num(&out, a, @as(u64, kvs.len));
    for (kvs) |kv| {
        try W.str(&out, a, kv[0]);
        switch (kv[1]) {
            .str => |s| {
                try W.num(&out, a, @as(u32, 8));
                try W.str(&out, a, s);
            },
            .u32 => |v| {
                try W.num(&out, a, @as(u32, 4));
                try W.num(&out, a, v);
            },
            .f32 => |v| {
                try W.num(&out, a, @as(u32, 6));
                try W.num(&out, a, v);
            },
            .bool => |v| {
                try W.num(&out, a, @as(u32, 7));
                try W.num(&out, a, @as(u8, @intFromBool(v)));
            },
            .f32s => |vs| {
                try W.num(&out, a, @as(u32, 9));
                try W.num(&out, a, @as(u32, 6));
                try W.num(&out, a, @as(u64, vs.len));
                for (vs) |v| try W.num(&out, a, v);
            },
            .bools => |vs| {
                try W.num(&out, a, @as(u32, 9));
                try W.num(&out, a, @as(u32, 7));
                try W.num(&out, a, @as(u64, vs.len));
                for (vs) |v| try W.num(&out, a, @as(u8, @intFromBool(v)));
            },
            .i32s => |vs| {
                try W.num(&out, a, @as(u32, 9));
                try W.num(&out, a, @as(u32, 5));
                try W.num(&out, a, @as(u64, vs.len));
                for (vs) |v| try W.num(&out, a, v);
            },
            .strs => |vs| {
                try W.num(&out, a, @as(u32, 9));
                try W.num(&out, a, @as(u32, 8));
                try W.num(&out, a, @as(u64, vs.len));
                for (vs) |v| try W.str(&out, a, v);
            },
        }
    }
    var offset: u64 = 0;
    for (tensors) |t| {
        try W.str(&out, a, t.name);
        try W.num(&out, a, @as(u32, @intCast(t.dims.len)));
        for (t.dims) |d| try W.num(&out, a, d);
        try W.num(&out, a, @as(u32, @backingInt(t.ty)));
        try W.num(&out, a, offset);
        offset += std.mem.alignForward(u64, t.data.len, 32);
    }
    while (out.items.len % 32 != 0) try out.append(a, 0);
    for (tensors) |t| {
        try out.appendSlice(a, t.data);
        while (out.items.len % 32 != 0) try out.append(a, 0);
    }
    return out.toOwnedSlice(a);
}

/// What a test tower varies from a valid one.
pub const TestTower = struct {
    depth: u32 = 1,
    out_hidden: u32 = 6,
    projector: []const u8 = "qwen3vl_merger",
    deepstack: bool = false,
    /// A tensor no Qwen3-VL tower has.
    extra_tensor: ?[]const u8 = null,
    /// The second patch slice stored as f16 instead of the first one's f32.
    mismatched_slice: bool = false,
    /// Rows of the position table; a tower's grid is its square root.
    positions: u64 = num_pos,
    /// `mm.2.weight` stored [out, merged]: the right bytes in the wrong dims.
    transposed_merger: bool = false,

    pub const hidden = 4;
    pub const ffn = 8;
    pub const patch = 2;
    pub const num_pos = 4;

    /// f32 element `i` of slice `t` of the patch embedding; small integers, so bf16 holds them exactly.
    pub fn patchValue(t: usize, i: usize) f32 {
        return @floatFromInt(t * 64 + i);
    }

    /// The mmproj GGUF bytes. Caller frees.
    pub fn write(self: TestTower, a: std.mem.Allocator) ![]u8 {
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const ar = arena.allocator();
        const h: u64 = hidden;
        const merged: u64 = h * 4;
        var tensors: std.ArrayList(TestTensor) = .empty;
        const f32Data = struct {
            fn make(al: std.mem.Allocator, n: u64, base: f32) ![]const u8 {
                const v = try al.alloc(f32, n);
                for (v, 0..) |*x, i| x.* = base + @as(f32, @floatFromInt(i % 7));
                return std.mem.sliceAsBytes(v);
            }
        }.make;
        const bf16Data = struct {
            fn make(al: std.mem.Allocator, n: u64) ![]const u8 {
                const v = try al.alloc(u16, n);
                for (v, 0..) |*x, i| x.* = @truncate(@as(u32, @bitCast(@as(f32, @floatFromInt(i % 5)))) >> 16);
                return std.mem.sliceAsBytes(v);
            }
        }.make;
        for (0..self.depth) |l| {
            const blk = struct {
                leaf: []const u8,
                in: u64,
                out: u64,
            };
            for ([_]blk{ .{ .leaf = "attn_qkv", .in = h, .out = 3 * h }, .{ .leaf = "attn_out", .in = h, .out = h }, .{ .leaf = "ffn_up", .in = h, .out = ffn }, .{ .leaf = "ffn_down", .in = ffn, .out = h } }) |b| {
                try tensors.append(ar, .{ .name = try std.fmt.allocPrint(ar, "v.blk.{d}.{s}.weight", .{ l, b.leaf }), .dims = try ar.dupe(u64, &.{ b.in, b.out }), .ty = .bf16, .data = try bf16Data(ar, b.in * b.out) });
                try tensors.append(ar, .{ .name = try std.fmt.allocPrint(ar, "v.blk.{d}.{s}.bias", .{ l, b.leaf }), .dims = try ar.dupe(u64, &.{b.out}), .ty = .f32, .data = try f32Data(ar, b.out, 1) });
            }
            for ([_][]const u8{ "ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias" }) |leaf| {
                try tensors.append(ar, .{ .name = try std.fmt.allocPrint(ar, "v.blk.{d}.{s}", .{ l, leaf }), .dims = try ar.dupe(u64, &.{h}), .ty = .f32, .data = try f32Data(ar, h, 2) });
            }
        }
        const patch_n: u64 = patch * patch * 3 * h;
        for ([_][]const u8{ "v.patch_embd.weight", "v.patch_embd.weight.1" }, 0..) |name, t| {
            const v = try ar.alloc(f32, patch_n);
            for (v, 0..) |*x, i| x.* = patchValue(t, i);
            const f16_slice = self.mismatched_slice and t == 1;
            const data = std.mem.sliceAsBytes(v);
            try tensors.append(ar, .{ .name = name, .dims = &.{ patch, patch, 3, h }, .ty = if (f16_slice) .f16 else .f32, .data = if (f16_slice) data[0 .. data.len / 2] else data });
        }
        try tensors.append(ar, .{ .name = "v.patch_embd.bias", .dims = try ar.dupe(u64, &.{h}), .ty = .f32, .data = try f32Data(ar, h, 3) });
        try tensors.append(ar, .{ .name = "v.position_embd.weight", .dims = try ar.dupe(u64, &.{ h, self.positions }), .ty = .f32, .data = try f32Data(ar, h * self.positions, 4) });
        try tensors.append(ar, .{ .name = "v.post_ln.weight", .dims = try ar.dupe(u64, &.{h}), .ty = .f32, .data = try f32Data(ar, h, 5) });
        try tensors.append(ar, .{ .name = "v.post_ln.bias", .dims = try ar.dupe(u64, &.{h}), .ty = .f32, .data = try f32Data(ar, h, 6) });
        try tensors.append(ar, .{ .name = "mm.0.weight", .dims = try ar.dupe(u64, &.{ merged, merged }), .ty = .bf16, .data = try bf16Data(ar, merged * merged) });
        try tensors.append(ar, .{ .name = "mm.0.bias", .dims = try ar.dupe(u64, &.{merged}), .ty = .f32, .data = try f32Data(ar, merged, 7) });
        const mm2: [2]u64 = if (self.transposed_merger) .{ self.out_hidden, merged } else .{ merged, self.out_hidden };
        try tensors.append(ar, .{ .name = "mm.2.weight", .dims = try ar.dupe(u64, &mm2), .ty = .bf16, .data = try bf16Data(ar, merged * self.out_hidden) });
        try tensors.append(ar, .{ .name = "mm.2.bias", .dims = try ar.dupe(u64, &.{self.out_hidden}), .ty = .f32, .data = try f32Data(ar, self.out_hidden, 8) });
        if (self.extra_tensor) |name| try tensors.append(ar, .{ .name = name, .dims = try ar.dupe(u64, &.{h}), .ty = .f32, .data = try f32Data(ar, h, 9) });

        const deepstack = try ar.alloc(bool, self.depth);
        @memset(deepstack, false);
        deepstack[0] = self.deepstack;
        return writeTestGguf(a, &.{
            .{ "general.architecture", .{ .str = "clip" } },
            .{ "clip.has_vision_encoder", .{ .bool = true } },
            .{ "clip.projector_type", .{ .str = self.projector } },
            .{ "clip.use_gelu", .{ .bool = true } },
            .{ "clip.vision.block_count", .{ .u32 = self.depth } },
            .{ "clip.vision.embedding_length", .{ .u32 = hidden } },
            .{ "clip.vision.feed_forward_length", .{ .u32 = ffn } },
            .{ "clip.vision.attention.head_count", .{ .u32 = 2 } },
            .{ "clip.vision.patch_size", .{ .u32 = patch } },
            .{ "clip.vision.spatial_merge_size", .{ .u32 = 2 } },
            .{ "clip.vision.projection_dim", .{ .u32 = self.out_hidden } },
            .{ "clip.vision.image_mean", .{ .f32s = &.{ 0.5, 0.5, 0.5 } } },
            .{ "clip.vision.image_std", .{ .f32s = &.{ 0.5, 0.5, 0.5 } } },
            .{ "clip.vision.is_deepstack_layers", .{ .bools = deepstack } },
        }, tensors.items);
    }
};

/// A tensor-less qwen35 language model of hidden size `hidden` whose vocab
/// holds the four vision tokens (ids 3..6): enough metadata for the config,
/// tokenizer and vision pairing, nothing to run. Caller frees.
pub fn writeTestLlm(a: std.mem.Allocator, hidden: u32) ![]u8 {
    return writeTestModel(a, .{ .hidden = hidden });
}

/// What a test language model varies from `writeTestLlm`'s.
pub const TestLlm = struct {
    hidden: u32,
    /// Blocks after the 4 trunk layers that hold the MTP head (`nextn_predict_layers`).
    nextn: u32 = 0,
    tensors: []const TestTensor = &.{},
};

pub fn writeTestModel(a: std.mem.Allocator, opts: TestLlm) ![]u8 {
    const hidden = opts.hidden;
    var kvs: std.ArrayList(struct { []const u8, TestKv }) = .empty;
    defer kvs.deinit(a);
    if (opts.nextn > 0) try kvs.append(a, .{ "qwen35.nextn_predict_layers", .{ .u32 = opts.nextn } });
    try kvs.appendSlice(a, &.{
        .{ "general.architecture", .{ .str = "qwen35" } },
        .{ "qwen35.block_count", .{ .u32 = 4 + opts.nextn } },
        .{ "qwen35.context_length", .{ .u32 = 4096 } },
        .{ "qwen35.embedding_length", .{ .u32 = hidden } },
        .{ "qwen35.feed_forward_length", .{ .u32 = 16 } },
        .{ "qwen35.attention.head_count", .{ .u32 = 2 } },
        .{ "qwen35.attention.head_count_kv", .{ .u32 = 1 } },
        .{ "qwen35.attention.key_length", .{ .u32 = 4 } },
        .{ "qwen35.rope.dimension_count", .{ .u32 = 4 } },
        .{ "qwen35.rope.dimension_sections", .{ .i32s = &.{ 1, 1, 0, 0 } } },
        .{ "qwen35.rope.freq_base", .{ .f32 = 10000.0 } },
        .{ "qwen35.attention.layer_norm_rms_epsilon", .{ .f32 = 1e-6 } },
        .{ "qwen35.full_attention_interval", .{ .u32 = 4 } },
        .{ "qwen35.ssm.conv_kernel", .{ .u32 = 4 } },
        .{ "qwen35.ssm.state_size", .{ .u32 = 4 } },
        .{ "qwen35.ssm.group_count", .{ .u32 = 1 } },
        .{ "qwen35.ssm.time_step_rank", .{ .u32 = 2 } },
        .{ "qwen35.ssm.inner_size", .{ .u32 = 8 } },
        .{ "tokenizer.ggml.model", .{ .str = "gpt2" } },
        .{ "tokenizer.ggml.pre", .{ .str = "qwen35" } },
        .{ "tokenizer.ggml.tokens", .{ .strs = &.{ "a", "b", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>" } } },
        .{ "tokenizer.ggml.token_type", .{ .i32s = &.{ 1, 1, 3, 3, 3, 3, 3 } } },
        .{ "tokenizer.ggml.merges", .{ .strs = &.{"a b"} } },
        .{ "tokenizer.ggml.eos_token_id", .{ .u32 = 2 } },
        .{ "tokenizer.chat_template", .{ .str = "{{ messages[0]['content'] }}" } },
    });
    return writeTestGguf(a, kvs.items, opts.tensors);
}

/// A qwen35 test model (hidden 32) whose blk.4 is a 15-tensor MTP head, its
/// matrices Q8_0 (scale 1/16, codes -3..3), plus the tensor `extra` names. Caller frees.
pub fn writeTestMtpModel(a: std.mem.Allocator, extra: ?[]const u8) ![]u8 {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const ar = arena.allocator();
    var tensors: std.ArrayList(TestTensor) = .empty;
    const mats = [_]struct { []const u8, u64, u64 }{
        .{ "nextn.eh_proj", 64, 32 }, .{ "attn_q", 32, 16 },   .{ "attn_k", 32, 8 },  .{ "attn_v", 32, 8 },
        .{ "attn_output", 32, 32 },   .{ "ffn_gate", 32, 64 }, .{ "ffn_up", 32, 64 }, .{ "ffn_down", 64, 32 },
    };
    for (mats) |m| {
        const data = try ar.alloc(u8, m[2] * m[1] / 32 * 34);
        for (0..data.len / 34) |blk| {
            const b = data[blk * 34 ..][0..34];
            std.mem.writeInt(u16, b[0..2], 0x2C00, .little); // f16 0.0625
            for (b[2..], 0..) |*q, i| q.* = @bitCast(@as(i8, @intCast(@as(i64, @intCast((blk + i) % 7)) - 3)));
        }
        try tensors.append(ar, .{ .name = try std.fmt.allocPrint(ar, "blk.4.{s}.weight", .{m[0]}), .dims = try ar.dupe(u64, &.{ m[1], m[2] }), .ty = .q8_0, .data = data });
    }
    const norms = [_]struct { []const u8, u64 }{
        .{ "nextn.enorm", 32 },         .{ "nextn.hnorm", 32 }, .{ "nextn.shared_head_norm", 32 }, .{ "attn_norm", 32 },
        .{ "post_attention_norm", 32 }, .{ "attn_q_norm", 4 },  .{ "attn_k_norm", 4 },
    };
    for (norms ++ [_]struct { []const u8, u64 }{.{ extra orelse "", 32 }}) |n| {
        if (n[0].len == 0) continue;
        const vals = try ar.alloc(f32, n[1]);
        @memset(vals, 1.5);
        try tensors.append(ar, .{ .name = try std.fmt.allocPrint(ar, "blk.4.{s}.weight", .{n[0]}), .dims = try ar.dupe(u64, &.{n[1]}), .ty = .f32, .data = std.mem.sliceAsBytes(vals) });
    }
    return writeTestModel(a, .{ .hidden = 32, .nextn = 1, .tensors = tensors.items });
}

fn testFiles(tower: TestTower, llm_hidden: u32, mm_bytes: *[]u8, llm_bytes: *[]u8) !struct { gguf.File, gguf.File } {
    const a = std.testing.allocator;
    mm_bytes.* = try tower.write(a);
    errdefer a.free(mm_bytes.*);
    llm_bytes.* = try writeTestLlm(a, llm_hidden);
    errdefer a.free(llm_bytes.*);
    var mm = try gguf.File.parse(a, mm_bytes.*);
    errdefer mm.deinit();
    return .{ mm, try gguf.File.parse(a, llm_bytes.*) };
}

test "the name map covers every leaf of a real Qwen3-VL mmproj and refuses the rest" {
    // The 22 distinct leaves of mmproj-Qwen3.8-27B-BF16.gguf, against the names the native packs carry.
    const hp = Hparams{ .depth = 27, .hidden = 1152, .ffn = 4304, .heads = 16, .patch = 16, .merge = 2, .out_hidden = 5120, .temporal_patch = 2, .num_pos = 2304 };
    const want = [_]struct { []const u8, []const u8 }{
        .{ "mm.0.bias", "model.visual.merger.linear_fc1.bias" },
        .{ "mm.0.weight", "model.visual.merger.linear_fc1.weight" },
        .{ "mm.2.bias", "model.visual.merger.linear_fc2.bias" },
        .{ "mm.2.weight", "model.visual.merger.linear_fc2.weight" },
        .{ "v.blk.26.attn_out.bias", "model.visual.blocks.26.attn.proj.bias" },
        .{ "v.blk.26.attn_out.weight", "model.visual.blocks.26.attn.proj.weight" },
        .{ "v.blk.26.attn_qkv.bias", "model.visual.blocks.26.attn.qkv.bias" },
        .{ "v.blk.26.attn_qkv.weight", "model.visual.blocks.26.attn.qkv.weight" },
        .{ "v.blk.26.ffn_down.bias", "model.visual.blocks.26.mlp.linear_fc2.bias" },
        .{ "v.blk.26.ffn_down.weight", "model.visual.blocks.26.mlp.linear_fc2.weight" },
        .{ "v.blk.26.ffn_up.bias", "model.visual.blocks.26.mlp.linear_fc1.bias" },
        .{ "v.blk.26.ffn_up.weight", "model.visual.blocks.26.mlp.linear_fc1.weight" },
        .{ "v.blk.26.ln1.bias", "model.visual.blocks.26.norm1.bias" },
        .{ "v.blk.26.ln1.weight", "model.visual.blocks.26.norm1.weight" },
        .{ "v.blk.26.ln2.bias", "model.visual.blocks.26.norm2.bias" },
        .{ "v.blk.26.ln2.weight", "model.visual.blocks.26.norm2.weight" },
        .{ "v.patch_embd.bias", "model.visual.patch_embed.proj.bias" },
        .{ "v.patch_embd.weight", "model.visual.patch_embed.proj.weight" },
        .{ "v.patch_embd.weight.1", "" },
        .{ "v.position_embd.weight", "model.visual.pos_embed.weight" },
        .{ "v.post_ln.bias", "model.visual.merger.norm.bias" },
        .{ "v.post_ln.weight", "model.visual.merger.norm.weight" },
    };
    var buf: [160]u8 = undefined;
    for (want) |w| try std.testing.expectEqualStrings(w[1], mapTensor(&buf, w[0], hp) orelse return error.TestUnmapped);
    for ([_][]const u8{ "v.blk.27.ln1.weight", "v.blk.0.attn_q.weight", "v.blk.0.ln1.scale", "v.patch_embd.weight.2", "v.patch_embd.weight.0", "mm.1.weight", "v.deepstack.0.weight", "v.blk.x.ln1.weight" }) |name|
        try std.testing.expectEqual(@as(?[]const u8, null), mapTensor(&buf, name, hp));
}

test "a valid tower pairs with its model; every other file is declined by name" {
    const a = std.testing.allocator;
    var buf: [200]u8 = undefined;
    const cases = [_]struct { TestTower, u32, ?[]const u8 }{
        .{ .{}, 6, null },
        .{ .{ .deepstack = true }, 6, "deepstack" },
        .{ .{ .projector = "mlp" }, 6, "projector_type mlp" },
        .{ .{ .extra_tensor = "v.blk.0.attn_q.weight" }, 6, "tensor v.blk.0.attn_q.weight" },
        .{ .{}, 8, "projection_dim 6 is not the model's hidden size 8" },
        .{ .{ .mismatched_slice = true }, 6, "metadata (BadMetadata)" },
        .{ .{ .positions = 6 }, 6, "metadata (BadMetadata)" },
        .{ .{ .positions = 65536 }, 6, null },
        .{ .{ .transposed_merger = true }, 6, "shape of mm.2.weight" },
    };
    for (cases) |c| {
        var mm_bytes: []u8 = undefined;
        var llm_bytes: []u8 = undefined;
        var files = try testFiles(c[0], c[1], &mm_bytes, &llm_bytes);
        defer {
            files[0].deinit();
            files[1].deinit();
            a.free(mm_bytes);
            a.free(llm_bytes);
        }
        const why = unsupportedReason(&files[0], &files[1], &buf);
        if (c[2]) |want| {
            try std.testing.expect(std.mem.indexOf(u8, why orelse return error.TestNotDeclined, want) != null);
        } else try std.testing.expectEqual(@as(?[]const u8, null), why);
    }
}

test "the config fragment mirrors an HF checkpoint's vision_config and token ids" {
    const a = std.testing.allocator;
    var mm_bytes: []u8 = undefined;
    var llm_bytes: []u8 = undefined;
    var files = try testFiles(.{ .depth = 2 }, 6, &mm_bytes, &llm_bytes);
    defer {
        files[0].deinit();
        files[1].deinit();
        a.free(mm_bytes);
        a.free(llm_bytes);
    }
    const hp = try Hparams.read(&files[0]);
    try std.testing.expectEqual(@as(u32, 2), hp.temporal_patch);
    try std.testing.expectEqual(@as(u32, TestTower.num_pos), hp.num_pos);
    const frag = try configFragment(a, hp, TokenIds.read(&files[1]).?, "qwen3_5");
    defer a.free(frag);
    const json = try std.mem.concat(a, u8, &.{ "{", frag, "}" });
    defer a.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, json, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    try std.testing.expectEqual(@as(i64, 5), root.get("image_token_id").?.integer);
    try std.testing.expectEqual(@as(i64, 6), root.get("video_token_id").?.integer);
    try std.testing.expectEqual(@as(i64, 3), root.get("vision_start_token_id").?.integer);
    try std.testing.expectEqual(@as(i64, 4), root.get("vision_end_token_id").?.integer);
    const vc = root.get("vision_config").?.object;
    try std.testing.expectEqualStrings("qwen3_5", vc.get("model_type").?.string);
    try std.testing.expectEqual(@as(usize, 0), vc.get("deepstack_visual_indexes").?.array.items.len);
    const want = [_]struct { []const u8, i64 }{
        .{ "depth", 2 },                    .{ "hidden_size", TestTower.hidden },              .{ "intermediate_size", TestTower.ffn },
        .{ "num_heads", 2 },                .{ "num_position_embeddings", TestTower.num_pos }, .{ "out_hidden_size", 6 },
        .{ "patch_size", TestTower.patch }, .{ "spatial_merge_size", 2 },                      .{ "temporal_patch_size", 2 },
        .{ "in_channels", 3 },
    };
    for (want) |w| try std.testing.expectEqual(w[1], vc.get(w[0]).?.integer);
}
