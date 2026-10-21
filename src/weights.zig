//! GGUF tensors -> the name -> array map mlx-serve's Transformer binds from.
//! Quantized tensors keep their bytes (see kernels.Info for how they travel),
//! float tensors become bf16 like an mlx-community checkpoint.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
const gguf = @import("gguf.zig");
const kernels = @import("kernels.zig");
const arch_mod = @import("arch.zig");
const qwen35 = arch_mod.qwen35;
const vision = @import("vision.zig");
const Arch = arch_mod.Arch;

pub const WeightMap = std.StringHashMap(mlx.mlx_array);

/// Fills `out` (keys owned by `allocator`, the contract of mlx-serve's Weights).
pub fn load(allocator: std.mem.Allocator, f: *const gguf.File, out: *WeightMap, s: mlx.mlx_stream) !void {
    const arch = try Arch.read(f);
    const hp = arch.gdn();
    var joins = Joins{ .allocator = allocator };
    defer joins.deinit();
    var name_buf: [160]u8 = undefined;
    var it = f.tensors.iterator();
    while (it.next()) |e| {
        const mapped = arch.mapTensor(&name_buf, e.key_ptr.*) orelse return error.UnsupportedTensor;
        if (mapped.name.len == 0) continue;
        const t = e.value_ptr.*;
        if (mapped.transform == .join_rows) {
            const joined = try joins.add(mapped.name, mapped.transform.join_rows, t) orelse continue;
            defer allocator.free(joined.data);
            try loadTensor(allocator, joined, .{ .name = mapped.name, .transform = .none }, hp, out, s);
        } else try loadTensor(allocator, t, mapped, hp, out, s);
    }
    if (joins.pending() > 0) return error.UnsupportedTensor;
    var i: usize = 0;
    while (arch.metadataTensor(&name_buf, i)) |m| : (i += 1) {
        try put(allocator, out, m.name, try int64Vector(f.getArray(m.key) orelse return error.MissingMetadata));
    }
}

fn loadTensor(allocator: std.mem.Allocator, t: gguf.Tensor, mapped: arch_mod.Mapped, hp: qwen35.Hparams, out: *WeightMap, s: mlx.mlx_stream) !void {
    if (t.ty.isQuantized()) return loadQuantized(allocator, t, mapped, hp, out, s);
    try put(allocator, out, mapped.name, try loadFloat(allocator, t, mapped.transform, hp, s));
}

const max_join_parts = 4;

/// Tensors HF stores fused (`join_rows`), held until every part has been seen.
const Joins = struct {
    allocator: std.mem.Allocator,
    parts: std.StringHashMapUnmanaged([max_join_parts]?gguf.Tensor) = .empty,

    /// The joined [in, rows] tensor once `t` completes `name` (caller frees its data), else null.
    fn add(self: *Joins, name: []const u8, at: @FieldType(arch_mod.Transform, "join_rows"), t: gguf.Tensor) !?gguf.Tensor {
        if (at.part >= at.parts or at.parts > max_join_parts or t.n_dims != 2) return error.UnsupportedTensor;
        const gop = try self.parts.getOrPut(self.allocator, name);
        if (!gop.found_existing) {
            gop.key_ptr.* = self.allocator.dupe(u8, name) catch |err| {
                self.parts.removeByPtr(gop.key_ptr);
                return err;
            };
            gop.value_ptr.* = @splat(null);
        }
        const slots = gop.value_ptr[0..at.parts];
        if (slots[at.part] != null) return error.UnsupportedTensor;
        for (slots) |p| if (p) |other| if (other.ty != t.ty or other.dims[0] != t.dims[0]) return error.BadShape;
        slots[at.part] = t;
        var rows: u64 = 0;
        var bytes: usize = 0;
        for (slots) |p| {
            rows += (p orelse return null).dims[1];
            bytes += p.?.data.len;
        }
        const data = try self.allocator.alloc(u8, bytes);
        var at_byte: usize = 0;
        for (slots) |p| {
            @memcpy(data[at_byte..][0..p.?.data.len], p.?.data);
            at_byte += p.?.data.len;
        }
        const key = gop.key_ptr.*;
        self.parts.removeByPtr(gop.key_ptr);
        self.allocator.free(key);
        return .{ .dims = .{ t.dims[0], rows, 1, 1 }, .n_dims = 2, .ty = t.ty, .data = data };
    }

    fn pending(self: *const Joins) usize {
        return self.parts.count();
    }

    fn deinit(self: *Joins) void {
        var it = self.parts.keyIterator();
        while (it.next()) |k| self.allocator.free(k.*);
        self.parts.deinit(self.allocator);
    }
};

/// A metadata integer array as an int64 [len] array.
fn int64Vector(arr: gguf.Array) !mlx.mlx_array {
    if (!arr.isInt()) return error.BadMetadata;
    var vals: [64]i64 = undefined;
    if (arr.len > vals.len) return error.BadMetadata;
    for (vals[0..arr.len], 0..) |*v, j| v.* = arr.int(j);
    return mlx.mlx_array_new_data(&vals, &[_]c_int{@intCast(arr.len)}, 1, .int64);
}

/// The MTP head a dense qwen35 file keeps after its trunk (`qwen35.hasMtpHead`),
/// loaded like the trunk under the `language_model.mtp.*` names (keys owned by
/// `allocator`). Every head leaf is checked before any loads.
pub fn loadMtp(allocator: std.mem.Allocator, f: *const gguf.File, out: *WeightMap, s: mlx.mlx_stream) !void {
    const arch = try Arch.read(f);
    if (arch != .qwen35 or !qwen35.hasMtpHead(f, arch.qwen35)) return error.NoMtpHead;
    const hp = arch.qwen35;
    var name_buf: [160]u8 = undefined;
    for ([_]bool{ false, true }) |load_now| {
        var it = f.tensors.iterator();
        while (it.next()) |e| {
            const b = arch_mod.splitLayer(e.key_ptr.*, hp.n_layers + 1) orelse continue;
            if (b.layer != hp.n_layers) continue;
            const mapped = qwen35.mapMtpTensor(&name_buf, e.key_ptr.*, hp) orelse return error.UnsupportedTensor;
            if (!load_now) continue;
            const t = e.value_ptr.*;
            if (t.ty.isQuantized()) {
                try loadQuantized(allocator, t, mapped, hp, out, s);
            } else {
                try put(allocator, out, mapped.name, try loadFloat(allocator, t, mapped.transform, hp, s));
            }
        }
    }
}

/// The tower of an mmproj `vision.unsupportedReason` accepted, as the dense
/// bf16 `model.visual.*` tensors of an HF checkpoint (keys owned by `allocator`).
pub fn loadVision(allocator: std.mem.Allocator, mm: *const gguf.File, out: *WeightMap, s: mlx.mlx_stream) !void {
    const hp = try vision.Hparams.read(mm);
    const no_tiling = std.mem.zeroes(qwen35.Hparams);
    var name_buf: [160]u8 = undefined;
    var it = mm.tensors.iterator();
    while (it.next()) |e| {
        const name = vision.mapTensor(&name_buf, e.key_ptr.*, hp) orelse return error.UnsupportedTensor;
        if (name.len == 0) continue;
        const arr = if (std.mem.eql(u8, e.key_ptr.*, "v.patch_embd.weight"))
            try stackPatchSlices(allocator, mm, hp.temporal_patch, s)
        else
            try loadFloat(allocator, e.value_ptr.*, .none, no_tiling, s);
        try put(allocator, out, name, arr);
    }
}

/// llama.cpp keeps one [kw, kh, C, hidden] patch kernel per time slice
/// (`v.patch_embd.weight`, `.weight.1`, ...); HF's Conv3d is one [hidden, C, T, kh, kw].
fn stackPatchSlices(allocator: std.mem.Allocator, mm: *const gguf.File, temporal: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const first = mm.tensors.get("v.patch_embd.weight") orelse return error.UnsupportedTensor;
    if (first.n_dims != 4) return error.BadShape;
    const plane = @as(usize, @intCast(first.dims[0] * first.dims[1])) * first.ty.blockBytes();
    const planes: usize = @intCast(first.dims[2] * first.dims[3]);
    const tmp = try allocator.alignedAlloc(u8, .@"4", first.data.len * temporal);
    defer allocator.free(tmp);
    var name_buf: [64]u8 = undefined;
    for (0..temporal) |t| {
        const slice = if (t == 0) first else mm.tensors.get(std.fmt.bufPrint(&name_buf, "v.patch_embd.weight.{d}", .{t}) catch unreachable) orelse return error.UnsupportedTensor;
        if (slice.ty != first.ty or !std.mem.eql(u64, &slice.dims, &first.dims)) return error.BadShape;
        for (0..planes) |p| @memcpy(tmp[(p * temporal + t) * plane ..][0..plane], slice.data[p * plane ..][0..plane]);
    }
    const shape = [_]c_int{ @intCast(first.dims[3]), @intCast(first.dims[2]), @intCast(temporal), @intCast(first.dims[1]), @intCast(first.dims[0]) };
    const dtype: mlx.mlx_dtype = switch (first.ty) {
        .f32 => .float32,
        .f16 => .float16,
        .bf16 => .bfloat16,
        else => return error.UnsupportedTensor,
    };
    return asBf16(mlx.mlx_array_new_data(tmp.ptr, &shape, shape.len, dtype), s);
}

/// `raw` as bf16, evaluated; consumes `raw`.
fn asBf16(raw: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    if (mlx.mlx_array_dtype(raw) == .bfloat16) return raw;
    defer _ = mlx.mlx_array_free(raw);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, raw, .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// Runs the decode and the prefill kernel of every distinct (type, shape) in
/// `map` once, so Metal compiles them during load and not inside the first
/// request (a few hundred ms there).
pub fn warmKernels(allocator: std.mem.Allocator, map: *const WeightMap, s: mlx.mlx_stream) !void {
    var seen = std.AutoHashMap([4]c_int, void).init(allocator);
    defer seen.deinit();
    var it = map.iterator();
    while (it.next()) |e| {
        if (!std.mem.endsWith(u8, e.key_ptr.*, ".weight")) continue;
        var sc_name: [160]u8 = undefined;
        const base = e.key_ptr.*[0 .. e.key_ptr.len - "weight".len];
        const sc = map.get(std.fmt.bufPrint(&sc_name, "{s}scales", .{base}) catch continue) orelse continue;
        const w = e.value_ptr.*;
        const info = kernels.infoOf(w, sc) orelse continue;
        const ws = mlx.getShape(w);
        const bank = ws.len == 3;
        const key = [4]c_int{ @intCast(@backingInt(info.ty)), @intFromBool(bank), ws[ws.len - 2], ws[ws.len - 1] };
        if ((try seen.getOrPut(key)).found_existing) continue;

        const in: c_int = @intCast(@as(usize, @intCast(ws[ws.len - 1])) / info.ty.blockBytes() * info.ty.blockElems());
        // One row = the decode matvec, 16 = the prefill tile kernel.
        for ([_]c_int{ 1, 16 }) |m| {
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            const x_shape: []const c_int = if (bank) &.{ m, 1, in } else &.{ m, in };
            try mlx.check(mlx.mlx_zeros(&x, x_shape.ptr, x_shape.len, .bfloat16, s));
            const y = if (bank) blk: {
                var experts = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(experts);
                try mlx.check(mlx.mlx_zeros(&experts, &[_]c_int{m}, 1, .uint32, s));
                break :blk try kernels.gatherLinear(.{ .ty = info.ty }, x, w, .{ .ctx = null }, experts, s);
            } else try kernels.linear(.{ .ty = info.ty }, x, w, s);
            defer _ = mlx.mlx_array_free(y);
            try mlx.check(mlx.mlx_array_eval(y));
        }
    }
}

fn put(allocator: std.mem.Allocator, out: *WeightMap, name: []const u8, arr: mlx.mlx_array) !void {
    errdefer _ = mlx.mlx_array_free(arr);
    const key = try allocator.dupe(u8, name);
    errdefer allocator.free(key);
    try out.putNoClobber(key, arr);
}

fn loadQuantized(allocator: std.mem.Allocator, t: gguf.Tensor, mapped: arch_mod.Mapped, hp: qwen35.Hparams, out: *WeightMap, s: mlx.mlx_stream) !void {
    if (!std.mem.endsWith(u8, mapped.name, ".weight")) return error.UnsupportedTensor;
    const rows: usize = @intCast(t.dims[1]);
    if (t.n_dims == 3) {
        // An expert bank, experts outermost: [experts, rows, row bytes].
        if (mapped.transform != .none) return error.UnsupportedTensor;
        const experts: usize = @intCast(t.dims[2]);
        const bank = [_]c_int{ @intCast(experts), @intCast(rows), @intCast(t.data.len / experts / rows) };
        try put(allocator, out, mapped.name, try packedRows(allocator, t.ty, t.data, &bank, s));
        return putSentinel(allocator, out, mapped.name, .{ .ty = t.ty }, s);
    }
    if (t.n_dims != 2) return error.UnsupportedTensor;
    const row_bytes = t.data.len / rows;
    const shape = [_]c_int{ @intCast(rows), @intCast(row_bytes) };

    if (mapped.transform == .split_ba) {
        const g = mapped.transform.split_ba;
        const nv = g.nk * g.r;
        if (rows != 2 * nv) return error.BadShape;
        const marker = ".in_proj_ba.";
        const at = std.mem.indexOf(u8, mapped.name, marker) orelse return error.UnsupportedTensor;
        const tmp = try allocator.alloc(u8, nv * row_bytes);
        defer allocator.free(tmp);
        for ([_][]const u8{ ".in_proj_b.", ".in_proj_a." }, 0..) |half_name, half| {
            for (0..g.nk) |k| for (0..g.r) |j| {
                const src = (k * 2 * g.r + half * g.r + j) * row_bytes;
                @memcpy(tmp[(k * g.r + j) * row_bytes ..][0..row_bytes], t.data[src..][0..row_bytes]);
            };
            var name_buf: [160]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "{s}{s}{s}", .{ mapped.name[0..at], half_name, mapped.name[at + marker.len ..] });
            try put(allocator, out, name, try packedRows(allocator, t.ty, tmp, &[_]c_int{ @intCast(nv), @intCast(row_bytes) }, s));
            try putSentinel(allocator, out, name, .{ .ty = t.ty }, s);
        }
        return;
    }

    var info = kernels.Info{ .ty = t.ty };
    const w = switch (mapped.transform) {
        .none => try packedRows(allocator, t.ty, t.data, &shape, s),
        .tiled_input => blk: {
            info.tiled = .{ .nk = @intCast(hp.nk), .r = @intCast(hp.r()) };
            break :blk try packedRows(allocator, t.ty, t.data, &shape, s);
        },
        .untile_rows => |u| blk: {
            // Rows are whole blocks, so moving a row moves its bytes and nothing else.
            if (u.start + hp.nv * u.unit != rows) return error.BadShape;
            const tmp = try allocator.alloc(u8, t.data.len);
            defer allocator.free(tmp);
            untile(tmp, t.data, u.start * row_bytes, u.unit * row_bytes, hp);
            break :blk try packedRows(allocator, t.ty, tmp, &shape, s);
        },
        else => return error.UnsupportedTensor,
    };
    try put(allocator, out, mapped.name, w);
    try putSentinel(allocator, out, mapped.name, info, s);
}

/// The weight as an MLX array, rows in the split layout when the kernels want it (see kernels.isSplit).
fn packedRows(allocator: std.mem.Allocator, ty: @import("quants.zig").GgmlType, bytes: []const u8, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    _ = s;
    const row_bytes: usize = @intCast(shape[shape.len - 1]);
    const in = row_bytes / ty.blockBytes() * ty.blockElems();
    if (!kernels.isSplit(ty, in)) return mlx.mlx_array_new_data(bytes.ptr, shape.ptr, @intCast(shape.len), .uint8);
    const rows = bytes.len / row_bytes;
    const tmp = try allocator.alloc(u8, rows * kernels.rowBytes(ty, in));
    defer allocator.free(tmp);
    kernels.packRows(ty, in, tmp, bytes);
    var padded: [3]c_int = undefined;
    @memcpy(padded[0..shape.len], shape);
    padded[shape.len - 1] = @intCast(kernels.rowBytes(ty, in));
    return mlx.mlx_array_new_data(tmp.ptr, &padded, @intCast(shape.len), .uint8);
}

/// `<base>.scales` next to `<base>.weight`, see kernels.Info.
fn putSentinel(allocator: std.mem.Allocator, out: *WeightMap, weight_name: []const u8, info: kernels.Info, s: mlx.mlx_stream) !void {
    var sc_name: [160]u8 = undefined;
    const base = weight_name[0 .. weight_name.len - "weight".len];
    try put(allocator, out, try std.fmt.bufPrint(&sc_name, "{s}scales", .{base}), try kernels.sentinel(info, s));
}

/// Copy `src` to `dst` moving each tiled value-head chunk (`unit` bytes, the
/// region starting at `start`) to its grouped position.
fn untile(dst: []u8, src: []const u8, start: usize, unit: usize, hp: qwen35.Hparams) void {
    @memcpy(dst[0..start], src[0..start]);
    for (0..hp.nv) |tiled| {
        const grouped = qwen35.groupedIndex(@intCast(tiled), hp.nk, hp.r());
        @memcpy(dst[start + grouped * unit ..][0..unit], src[start + tiled * unit ..][0..unit]);
    }
}

fn loadFloat(allocator: std.mem.Allocator, t: gguf.Tensor, transform: arch_mod.Transform, hp: qwen35.Hparams, s: mlx.mlx_stream) !mlx.mlx_array {
    // ggml dims are innermost first.
    var shape_buf: [5]c_int = undefined;
    for (0..t.n_dims) |i| shape_buf[i] = @intCast(t.dims[t.n_dims - 1 - i]);
    var shape: []const c_int = shape_buf[0..t.n_dims];
    const dtype: mlx.mlx_dtype = switch (t.ty) {
        .f32 => .float32,
        .f16 => .float16,
        .bf16 => .bfloat16,
        else => unreachable,
    };

    const elem = t.ty.blockBytes();
    const raw = switch (transform) {
        .none => mlx.mlx_array_new_data(t.data.ptr, shape.ptr, @intCast(shape.len), dtype),
        .shared_gate => blk: {
            if (t.n_dims != 1) return error.BadShape;
            break :blk mlx.mlx_array_new_data(t.data.ptr, &[_]c_int{ 1, shape[0] }, 2, dtype);
        },
        .trailing_axis => blk: {
            if (t.n_dims + 1 > shape_buf.len) return error.BadShape;
            shape_buf[t.n_dims] = 1;
            shape = shape_buf[0 .. t.n_dims + 1];
            break :blk mlx.mlx_array_new_data(t.data.ptr, shape.ptr, @intCast(shape.len), dtype);
        },
        .minus_one, .neg_log => blk: {
            if (t.ty != .f32) return error.UnsupportedTensor;
            const tmp = try allocator.alloc(f32, t.elems());
            defer allocator.free(tmp);
            for (tmp, std.mem.bytesAsSlice(f32, t.data)) |*o, v| o.* = if (transform == .neg_log) @log(-v) else v - 1;
            const flat = [_]c_int{@intCast(t.elems())};
            break :blk mlx.mlx_array_new_data(tmp.ptr, if (transform == .neg_log) &flat else shape.ptr, if (transform == .neg_log) 1 else @intCast(shape.len), dtype);
        },
        .flatten => mlx.mlx_array_new_data(t.data.ptr, &[_]c_int{@intCast(t.elems())}, 1, dtype),
        .split_ba, .join_rows => return error.UnsupportedTensor,
        // Un-tiling only moves bytes, so it works the same on any float type.
        .untile_rows, .untile_vec, .a_log, .conv1d, .tiled_input => blk: {
            const tmp = try allocator.alignedAlloc(u8, .@"4", t.data.len);
            defer allocator.free(tmp);
            const row: usize = if (t.n_dims == 2) @intCast(t.dims[0]) else 1;
            switch (transform) {
                .untile_rows => |u| {
                    if (t.n_dims != 2 or u.start + hp.nv * u.unit != t.dims[1]) return error.BadShape;
                    untile(tmp, t.data, u.start * row * elem, u.unit * row * elem, hp);
                },
                // [channels, kernel]: q and k channels first, then the value channels.
                .conv1d => {
                    if (t.n_dims != 2 or 2 * hp.nk * hp.dk + hp.nv * hp.dv != t.dims[1]) return error.BadShape;
                    untile(tmp, t.data, 2 * hp.nk * hp.dk * row * elem, hp.dv * row * elem, hp);
                    shape_buf[t.n_dims] = 1;
                    shape = shape_buf[0 .. t.n_dims + 1];
                },
                // A float weight can simply have its columns moved, row by row.
                .tiled_input => {
                    if (t.n_dims != 2 or hp.nv * hp.dv != row) return error.BadShape;
                    for (0..@intCast(t.dims[1])) |r| untile(tmp[r * row * elem ..][0 .. row * elem], t.data[r * row * elem ..][0 .. row * elem], 0, hp.dv * elem, hp);
                },
                else => {
                    if (t.elems() != hp.nv) return error.BadShape;
                    untile(tmp, t.data, 0, elem, hp);
                },
            }
            if (transform == .a_log) {
                if (t.ty != .f32) return error.UnsupportedTensor;
                for (std.mem.bytesAsSlice(f32, tmp)) |*v| v.* = @log(-v.*);
            }
            break :blk mlx.mlx_array_new_data(tmp.ptr, shape.ptr, @intCast(shape.len), dtype);
        },
    };
    // A_log stays f32 like the reference checkpoints, the gate math runs in f32.
    if (transform == .a_log or transform == .neg_log) return raw;
    return asBf16(raw, s);
}

test "untile moves tiled value-head chunks to grouped order and keeps the prefix" {
    var hp = std.mem.zeroes(qwen35.Hparams);
    hp.nk = 2;
    hp.nv = 6;
    // prefix "PP", then heads tiled as g0j0 g1j0 g0j1 g1j1 g0j2 g1j2, two bytes each.
    const src = "PP" ++ "a0" ++ "b0" ++ "a1" ++ "b1" ++ "a2" ++ "b2";
    var dst: [src.len]u8 = undefined;
    untile(&dst, src, 2, 2, hp);
    try std.testing.expectEqualStrings("PPa0a1a2b0b1b2", &dst);
}

test "an mmproj tower loads as bf16 model.visual.* tensors, patch slices stacked on the temporal axis" {
    const a = std.testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const Tower = vision.TestTower;
    const bytes = try (Tower{ .depth = 2 }).write(a);
    defer a.free(bytes);
    var f = try gguf.File.parse(a, bytes);
    defer f.deinit();
    var map = WeightMap.init(a);
    defer {
        var it = map.iterator();
        while (it.next()) |e| {
            _ = mlx.mlx_array_free(e.value_ptr.*);
            a.free(e.key_ptr.*);
        }
        map.deinit();
    }
    try loadVision(a, &f, &map, s);

    try std.testing.expectEqual(@as(u32, 2 * 12 + 9), map.count());
    var it = map.iterator();
    while (it.next()) |e| {
        try std.testing.expect(std.mem.startsWith(u8, e.key_ptr.*, "model.visual."));
        try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(e.value_ptr.*));
    }
    const h = Tower.hidden;
    const p = Tower.patch;
    const shapes = [_]struct { []const u8, []const c_int }{
        .{ "model.visual.patch_embed.proj.weight", &.{ h, 3, 2, p, p } },
        .{ "model.visual.patch_embed.proj.bias", &.{h} },
        .{ "model.visual.pos_embed.weight", &.{ Tower.num_pos, h } },
        .{ "model.visual.blocks.1.attn.qkv.weight", &.{ 3 * h, h } },
        .{ "model.visual.blocks.1.attn.qkv.bias", &.{3 * h} },
        .{ "model.visual.blocks.1.attn.proj.weight", &.{ h, h } },
        .{ "model.visual.blocks.1.mlp.linear_fc1.weight", &.{ Tower.ffn, h } },
        .{ "model.visual.blocks.1.mlp.linear_fc2.weight", &.{ h, Tower.ffn } },
        .{ "model.visual.blocks.1.norm2.bias", &.{h} },
        .{ "model.visual.merger.norm.weight", &.{h} },
        .{ "model.visual.merger.linear_fc1.weight", &.{ 4 * h, 4 * h } },
        .{ "model.visual.merger.linear_fc2.weight", &.{ 6, 4 * h } },
        .{ "model.visual.merger.linear_fc2.bias", &.{6} },
    };
    for (shapes) |sh| try std.testing.expectEqualSlices(c_int, sh[1], mlx.getShape(map.get(sh[0]) orelse return error.TestMissing));

    // HF order [hidden, C, T, kh, kw]: element (o, c, t, y, x) is slice t's (o, c, y, x).
    var w32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w32);
    try mlx.check(mlx.mlx_astype(&w32, map.get("model.visual.patch_embed.proj.weight").?, .float32, s));
    try mlx.check(mlx.mlx_array_eval(w32));
    const got = (mlx.mlx_array_data_float32(w32) orelse return error.TestNoData)[0 .. h * 3 * 2 * p * p];
    for (0..h * 3) |oc| for (0..2) |t| for (0..p * p) |yx| {
        try std.testing.expectEqual(Tower.patchValue(t, oc * p * p + yx), got[(oc * 2 + t) * p * p + yx]);
    };
}

test "the MTP block loads as the language_model.mtp.* head, quantized weights with their sentinel" {
    const a = std.testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const bytes = try vision.writeTestMtpModel(a, null);
    defer a.free(bytes);
    var f = try gguf.File.parse(a, bytes);
    defer f.deinit();
    var map = WeightMap.init(a);
    defer {
        var it = map.iterator();
        while (it.next()) |e| {
            _ = mlx.mlx_array_free(e.value_ptr.*);
            a.free(e.key_ptr.*);
        }
        map.deinit();
    }
    try loadMtp(a, &f, &map, s);
    // 15 tensors, and a `.scales` sentinel beside each of the 8 quantized ones.
    try std.testing.expectEqual(@as(u32, 15 + 8), map.count());
    var it = map.iterator();
    while (it.next()) |e| try std.testing.expect(std.mem.startsWith(u8, e.key_ptr.*, "language_model.mtp."));
    const fc = kernels.infoOf(map.get("language_model.mtp.fc.weight").?, map.get("language_model.mtp.fc.scales").?) orelse return error.TestNotGguf;
    try std.testing.expectEqual(@import("quants.zig").GgmlType.q8_0, fc.ty);
    const norm = map.get("language_model.mtp.layers.0.self_attn.q_norm.weight").?;
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(norm));
    try std.testing.expectEqualSlices(c_int, &.{4}, mlx.getShape(norm));

    // No head, or a head leaf the map does not know: nothing half-loaded.
    const plain = try vision.writeTestModel(a, .{ .hidden = 32 });
    defer a.free(plain);
    var pf = try gguf.File.parse(a, plain);
    defer pf.deinit();
    try std.testing.expectError(error.NoMtpHead, loadMtp(a, &pf, &map, s));
    const odd = try vision.writeTestMtpModel(a, "nextn.embed_tokens");
    defer a.free(odd);
    var of = try gguf.File.parse(a, odd);
    defer of.deinit();
    var odd_map = WeightMap.init(a);
    defer {
        var oit = odd_map.iterator();
        while (oit.next()) |e| {
            _ = mlx.mlx_array_free(e.value_ptr.*);
            a.free(e.key_ptr.*);
        }
        odd_map.deinit();
    }
    try std.testing.expectError(error.UnsupportedTensor, loadMtp(a, &of, &odd_map, s));
    try std.testing.expectEqual(@as(u32, 0), odd_map.count());
}

test "an mmproj tower is bit-equal to an MLX pack of the same checkpoint (set MLX_SERVE_GGUF_VISION_MMPROJ, MLX_SERVE_GGUF_VISION_ORACLE)" {
    const mm_path = std.mem.span(std.c.getenv("MLX_SERVE_GGUF_VISION_MMPROJ") orelse return error.SkipZigTest);
    const oracle = std.c.getenv("MLX_SERVE_GGUF_VISION_ORACLE") orelse return error.SkipZigTest;
    const a = std.testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var f = try gguf.File.open(a, mm_path);
    defer f.deinit();
    var map = WeightMap.init(a);
    defer {
        var it = map.iterator();
        while (it.next()) |e| {
            _ = mlx.mlx_array_free(e.value_ptr.*);
            a.free(e.key_ptr.*);
        }
        map.deinit();
    }
    try loadVision(a, &f, &map, s);

    var want = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(want);
    var meta_map = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta_map);
    try mlx.check(mlx.mlx_load_safetensors(&want, &meta_map, oracle, cpu));
    // Our spelling, or mlx-vlm's (`vision_tower.`, the patch kernel channels-last [hidden, T, kh, kw, C]) inside a whole pack.
    var probe = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(probe);
    const ours = "model.visual.";
    const prefix: []const u8 = if (mlx.mlx_map_string_to_array_get(&probe, want, ours ++ "pos_embed.weight") == 0) ours else "vision_tower.";
    var n_want: u32 = 0;
    const wit = mlx.mlx_map_string_to_array_iterator_new(want);
    defer _ = mlx.mlx_map_string_to_array_iterator_free(wit);
    while (true) {
        var key: ?[*:0]const u8 = null;
        var v = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v);
        if (mlx.mlx_map_string_to_array_iterator_next(&key, &v, wit) != 0 or key == null) break;
        n_want += @intFromBool(std.mem.startsWith(u8, std.mem.span(key.?), prefix));
    }
    try std.testing.expectEqual(n_want, map.count());

    var differing: usize = 0;
    var it = map.iterator();
    while (it.next()) |e| {
        const key = try std.fmt.allocPrintSentinel(a, "{s}{s}", .{ prefix, e.key_ptr.*[ours.len..] }, 0);
        defer a.free(key);
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        try mlx.check(mlx.mlx_map_string_to_array_get(&w, want, key));
        try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(w));
        if (mlx.getShape(w).len == 5 and mlx.getShape(w)[4] == 3) {
            var t = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_transpose_axes(&t, w, &[_]c_int{ 0, 4, 1, 2, 3 }, 5, cpu));
            _ = mlx.mlx_array_free(w);
            w = t;
        }
        try std.testing.expectEqualSlices(c_int, mlx.getShape(w), mlx.getShape(e.value_ptr.*));
        // Bits, not values: both as uint16.
        var wb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wb);
        var gb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gb);
        var eq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eq);
        try mlx.check(mlx.mlx_view(&wb, w, .uint16, cpu));
        try mlx.check(mlx.mlx_view(&gb, e.value_ptr.*, .uint16, cpu));
        try mlx.check(mlx.mlx_array_equal(&eq, wb, gb, false, cpu));
        try mlx.check(mlx.mlx_array_eval(eq));
        var same = false;
        try mlx.check(mlx.mlx_array_item_bool(&same, eq));
        if (!same) {
            std.debug.print("not bit-equal: {s}\n", .{e.key_ptr.*});
            differing += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), differing);
}

test "weights the loader rearranges are stored split like the rest (tiled input, fused b / a)" {
    const a = std.testing.allocator;
    const quants = @import("quants.zig");
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    // Q6_K, 8 blocks per row: 1680 bytes, a multiple of 16, so no row padding tells the layouts apart.
    const ty = quants.GgmlType.q6_k;
    const be = comptime ty.blockElems();
    const bb = comptime ty.blockBytes();
    const nb = 8;
    const in = nb * be;
    const n_rows = 2;
    const data = @embedFile("fixtures/q6_k.bin");
    var ref: [4 * 256]f32 = undefined;
    try quants.dequantize(ty, data[0 .. 4 * bb], &ref);
    // Block b of row r is fixture block (r + b) % 4.
    var bytes: [n_rows * nb * bb]u8 = undefined;
    for (0..n_rows * nb) |i| @memcpy(bytes[i * bb ..][0..bb], data[((i / nb + i % nb) % 4) * bb ..][0..bb]);
    const t = gguf.Tensor{ .dims = .{ in, n_rows, 1, 1 }, .n_dims = 2, .ty = ty, .data = &bytes };
    var hp = std.mem.zeroes(qwen35.Hparams);
    hp.nk = 2;
    hp.nv = 4;

    var map = WeightMap.init(a);
    defer {
        var it = map.iterator();
        while (it.next()) |e| {
            _ = mlx.mlx_array_free(e.value_ptr.*);
            a.free(e.key_ptr.*);
        }
        map.deinit();
    }
    try loadQuantized(a, t, .{ .name = "out_proj.weight", .transform = .tiled_input }, hp, &map, s);
    try loadQuantized(a, t, .{ .name = "l.in_proj_ba.weight", .transform = .{ .split_ba = .{ .nk = 1, .r = 1 } } }, hp, &map, s);

    var prng = std.Random.DefaultPrng.init(3);
    var xv: [in]f32 = undefined;
    for (&xv) |*v| v.* = prng.random().float(f32) - 0.5;
    // out_proj gets x grouped [nk, r, d] and reads it tiled [r, nk, d].
    const d = in / hp.nv;
    var xt: [in]f32 = undefined;
    for (0..hp.nk) |k| for (0..hp.r()) |j| @memcpy(xt[(j * hp.nk + k) * d ..][0..d], xv[(k * hp.r() + j) * d ..][0..d]);
    const x = mlx.mlx_array_new_data(&xv, &[_]c_int{ 1, in }, 2, .float32);
    defer _ = mlx.mlx_array_free(x);
    const Case = struct { name: []const u8, rows: []const usize, x_seen: *const [in]f32 };
    for ([_]Case{
        .{ .name = "out_proj.", .rows = &.{ 0, 1 }, .x_seen = &xt },
        .{ .name = "l.in_proj_b.", .rows = &.{0}, .x_seen = &xv },
        .{ .name = "l.in_proj_a.", .rows = &.{1}, .x_seen = &xv },
    }) |c| {
        var nbuf: [64]u8 = undefined;
        const w = map.get(try std.fmt.bufPrint(&nbuf, "{s}weight", .{c.name})).?;
        const info = kernels.infoOf(w, map.get(try std.fmt.bufPrint(&nbuf, "{s}scales", .{c.name})).?).?;
        const y = try kernels.linear(info, x, w, s);
        defer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_array_eval(y));
        const got = mlx.mlx_array_data_float32(y).?[0..c.rows.len];
        for (c.rows, got) |row, g| {
            var want: f32 = 0;
            var scale: f32 = 0;
            for (c.x_seen, 0..) |xc, col| {
                const term = ref[((row + col / be) % 4) * be + col % be] * xc;
                want += term;
                scale += @abs(term);
            }
            try std.testing.expectApproxEqAbs(want, g, 1e-5 * scale);
        }
    }
}

test "fused rows are joined in part order, whichever part the file lists first" {
    const a = std.testing.allocator;
    var joins = Joins{ .allocator = a };
    defer joins.deinit();
    const q_bytes = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 }; // bf16 [2 rows x 2]
    const k_bytes = [_]u8{ 9, 10, 11, 12 }; // bf16 [1 row x 2]
    const q = gguf.Tensor{ .dims = .{ 2, 2, 1, 1 }, .n_dims = 2, .ty = .bf16, .data = &q_bytes };
    const k = gguf.Tensor{ .dims = .{ 2, 1, 1, 1 }, .n_dims = 2, .ty = .bf16, .data = &k_bytes };
    try std.testing.expect(try joins.add("qk.weight", .{ .part = 1, .parts = 2 }, k) == null);
    try std.testing.expectEqual(@as(usize, 1), joins.pending());
    const joined = (try joins.add("qk.weight", .{ .part = 0, .parts = 2 }, q)).?;
    defer a.free(joined.data);
    try std.testing.expectEqual(@as(usize, 0), joins.pending());
    try std.testing.expectEqualSlices(u64, &.{ 2, 3 }, joined.dims[0..joined.n_dims]);
    try std.testing.expectEqualSlices(u8, &(q_bytes ++ k_bytes), joined.data);

    // A part seen twice, or parts of different widths, cannot be joined.
    try std.testing.expect(try joins.add("x.weight", .{ .part = 0, .parts = 2 }, q) == null);
    try std.testing.expectError(error.UnsupportedTensor, joins.add("x.weight", .{ .part = 0, .parts = 2 }, q));
    const wide = gguf.Tensor{ .dims = .{ 4, 1, 1, 1 }, .n_dims = 2, .ty = .bf16, .data = &q_bytes };
    try std.testing.expectError(error.BadShape, joins.add("x.weight", .{ .part = 1, .parts = 2 }, wide));
}

test "a metadata array becomes an int64 vector" {
    var buf: [24]u8 = undefined;
    for ([_]i64{ 23703573157769, -2, 7 }, 0..) |v, i| std.mem.writeInt(i64, buf[8 * i ..][0..8], v, .little);
    const arr = try int64Vector(.{ .elem_type = .i64, .len = 3, .bytes = &buf });
    defer _ = mlx.mlx_array_free(arr);
    try std.testing.expectEqual(mlx.mlx_dtype.int64, mlx.mlx_array_dtype(arr));
    try std.testing.expectEqualSlices(c_int, &.{3}, mlx.getShape(arr));
    try mlx.check(mlx.mlx_array_eval(arr));
    // mlx-c's typed data accessors only reinterpret the buffer.
    const got = std.mem.bytesAsSlice(i64, @as([*]align(8) const u8, @ptrCast(@alignCast(mlx.mlx_array_data_uint8(arr).?)))[0..24]);
    try std.testing.expectEqualSlices(i64, &.{ 23703573157769, -2, 7 }, got);
    var fbuf: [4]u8 = undefined;
    try std.testing.expectError(error.BadMetadata, int64Vector(.{ .elem_type = .f32, .len = 1, .bytes = &fbuf }));
}

test "real model loads into a weight map (set MLX_SERVE_GGUF_TEST_MODEL)" {
    const path = std.mem.span(std.c.getenv("MLX_SERVE_GGUF_TEST_MODEL") orelse return error.SkipZigTest);
    const a = std.testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var f = try gguf.File.open(a, path);
    defer f.deinit();
    var map = WeightMap.init(a);
    defer {
        var it = map.iterator();
        while (it.next()) |e| {
            _ = mlx.mlx_array_free(e.value_ptr.*);
            a.free(e.key_ptr.*);
        }
        map.deinit();
    }
    try load(a, &f, &map, s);
    try warmKernels(a, &map, s);

    const arch = try Arch.read(&f);
    const p = "language_model.model.";
    const prefix: []const u8 = switch (arch) {
        .gemma3, .lfm2, .gpt_oss, .qwen3next => "model.",
        .nemotron_h => "backbone.",
        else => p,
    };
    const norm_name = try std.fmt.allocPrint(a, "{s}{s}norm{s}.weight", .{ prefix, if (arch == .lfm2) "embedding_" else "", if (arch == .nemotron_h) "_f" else "" });
    defer a.free(norm_name);
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(map.get(norm_name).?));
    switch (arch) {
        .qwen35 => |hp| {
            const qkv = map.get(p ++ "layers.0.linear_attn.in_proj_qkv.weight").?;
            try std.testing.expectEqual(@as(c_int, @intCast(2 * hp.nk * hp.dk + hp.nv * hp.dv)), mlx.getShape(qkv)[0]);
            const out_sc = map.get(p ++ "layers.0.linear_attn.out_proj.scales").?;
            const out_w = map.get(p ++ "layers.0.linear_attn.out_proj.weight").?;
            try std.testing.expectEqual(hp.r() > 1, kernels.infoOf(out_w, out_sc).?.tiled != null);
            try std.testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(map.get(p ++ "layers.0.linear_attn.A_log").?));
            try std.testing.expectEqualSlices(c_int, &.{ @intCast(2 * hp.nk * hp.dk + hp.nv * hp.dv), @intCast(hp.conv_kernel), 1 }, mlx.getShape(map.get(p ++ "layers.0.linear_attn.conv1d.weight").?));
        },
        .gemma4 => |hp| {
            try std.testing.expectEqualSlices(c_int, &.{1}, mlx.getShape(map.get(p ++ "layers.0.layer_scalar").?));
            try std.testing.expect(map.get("rope_freqs.weight") == null);
            try std.testing.expectEqual(hp.ple_dim > 0, map.get(p ++ "embed_tokens_per_layer.weight") != null);
            try std.testing.expectEqual(hp.ple_dim > 0, map.get(p ++ "layers.0.per_layer_input_gate.weight") != null);
            const q = map.get(p ++ "layers.0.self_attn.q_proj.weight").?;
            try std.testing.expectEqual(@as(c_int, @intCast(hp.n_heads * hp.head_dim)), mlx.getShape(q)[0]);
        },
        .lfm2 => |hp| {
            try std.testing.expectEqualSlices(c_int, &.{ @intCast(hp.hidden), @intCast(hp.conv_cache), 1 }, mlx.getShape(map.get("model.layers.0.conv.conv.weight").?));
            try std.testing.expectEqual(hp.moe, map.get("model.layers.2.feed_forward.switch_mlp.up_proj.weight") != null);
        },
        .qwen3next => |hp| {
            try std.testing.expectEqualSlices(c_int, &.{@intCast(hp.nv)}, mlx.getShape(map.get("model.layers.0.linear_attn.A_log").?));
            try std.testing.expectEqual(@as(c_int, @intCast(hp.nv)), mlx.getShape(map.get("model.layers.0.linear_attn.in_proj_b.weight").?)[0]);
            try std.testing.expect(map.get("model.layers.0.linear_attn.in_proj_ba.weight") == null);
        },
        .nemotron_h => |hp| {
            try std.testing.expectEqualSlices(c_int, &.{@intCast(hp.mamba_heads)}, mlx.getShape(map.get("backbone.layers.0.mixer.A_log").?));
            try std.testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(map.get("backbone.layers.0.mixer.A_log").?));
        },
        .gpt_oss => |hp| {
            try std.testing.expectEqualSlices(c_int, &.{@intCast(hp.n_heads)}, mlx.getShape(map.get("model.layers.0.self_attn.sinks").?));
            try std.testing.expect(map.get("model.layers.0.mlp.experts.gate_proj.bias") != null);
        },
        .qwen4exp => |hp| {
            const idx = try std.fmt.allocPrint(a, p ++ "layers.{d}.self_attn.indexer.index_qk_proj.weight", .{hp.base.full_attn_interval - 1});
            defer a.free(idx);
            try std.testing.expectEqual(@as(c_int, @intCast((hp.indexer_heads + 1) * hp.indexer_head_dim)), mlx.getShape(map.get(idx).?)[0]);
        },
        .gemma3 => |hp| {
            const q = map.get("model.layers.0.self_attn.q_proj.weight").?;
            try std.testing.expectEqual(@as(c_int, @intCast(hp.n_heads * hp.head_dim)), mlx.getShape(q)[0]);
            try std.testing.expect(map.get("model.layers.0.input_layernorm.weight") != null);
        },
    }
}
