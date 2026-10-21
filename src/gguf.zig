//! Zero-copy GGUF reader: the file is mmapped, metadata strings, arrays and
//! tensor data are all slices into the map.
const std = @import("std");
const quants = @import("quants.zig");
const GgmlType = quants.GgmlType;

pub const Error = error{
    BadMagic,
    UnsupportedVersion,
    Truncated,
    BadValueType,
    BadTensor,
    DuplicateKey,
    /// A split GGUF opens from its first shard only.
    SplitNotFirstShard,
    /// A split GGUF whose shards, names or tensor counts disagree.
    BadSplit,
} || std.mem.Allocator.Error;

const Map = []align(std.heap.page_size_min) const u8;

const ValueType = enum(u32) { u8, i8, u16, i16, u32, i32, f32, bool, string, array, u64, i64, f64, _ };

fn scalarSize(t: ValueType) ?usize {
    return switch (t) {
        .u8, .i8, .bool => 1,
        .u16, .i16 => 2,
        .u32, .i32, .f32 => 4,
        .u64, .i64, .f64 => 8,
        else => null,
    };
}

pub const Array = struct {
    elem_type: ValueType,
    len: usize,
    /// Raw payload. Strings are (u64 len, bytes) back to back, walk them with `strings()`.
    bytes: []const u8,

    pub fn strings(self: Array) StringIterator {
        std.debug.assert(self.elem_type == .string);
        return .{ .r = .{ .buf = self.bytes }, .left = self.len };
    }

    pub fn boolean(self: Array, i: usize) bool {
        std.debug.assert(self.elem_type == .bool);
        return self.bytes[i] != 0;
    }

    pub fn float(self: Array, i: usize) f64 {
        const sz = scalarSize(self.elem_type).?;
        var r = Reader{ .buf = self.bytes[i * sz ..] };
        return switch (r.scalar(self.elem_type) catch unreachable) {
            .float => |v| v,
            else => unreachable,
        };
    }

    /// Whether `int` reads every element.
    pub fn isInt(self: Array) bool {
        return switch (self.elem_type) {
            .u8, .i8, .u16, .i16, .u32, .i32, .u64, .i64 => true,
            else => false,
        };
    }

    /// Scalar element as i64 (token types, eos id lists, ...).
    pub fn int(self: Array, i: usize) i64 {
        const sz = scalarSize(self.elem_type).?;
        var r = Reader{ .buf = self.bytes[i * sz ..] };
        return (r.scalar(self.elem_type) catch unreachable).asInt().?;
    }
};

pub const StringIterator = struct {
    r: Reader,
    left: usize,

    pub fn next(self: *StringIterator) ?[]const u8 {
        if (self.left == 0) return null;
        self.left -= 1;
        return self.r.string() catch unreachable; // validated at parse
    }
};

pub const Value = union(enum) {
    uint: u64,
    int: i64,
    float: f64,
    bool: bool,
    string: []const u8,
    array: Array,

    pub fn asInt(self: Value) ?i64 {
        return switch (self) {
            .uint => |v| std.math.cast(i64, v),
            .int => |v| v,
            else => null,
        };
    }
};

pub const Tensor = struct {
    /// ggml order: dims[0] is the row length (in_features for a linear).
    dims: [4]u64,
    n_dims: u32,
    ty: GgmlType,
    data: []const u8,

    pub fn elems(self: Tensor) u64 {
        var n: u64 = 1;
        for (self.dims[0..self.n_dims]) |d| n *= d;
        return n;
    }
};

const Reader = struct {
    buf: []const u8,
    pos: usize = 0,

    fn take(self: *Reader, n: u64) Error![]const u8 {
        if (n > self.buf.len - self.pos) return error.Truncated;
        const s = self.buf[self.pos..][0..@intCast(n)];
        self.pos += s.len;
        return s;
    }

    fn num(self: *Reader, comptime T: type) Error!T {
        const bytes = try self.take(@sizeOf(T));
        return @bitCast(std.mem.readInt(@Int(.unsigned, @bitSizeOf(T)), bytes[0..@sizeOf(T)], .little));
    }

    fn string(self: *Reader) Error![]const u8 {
        return self.take(try self.num(u64));
    }

    fn scalar(self: *Reader, t: ValueType) Error!Value {
        return switch (t) {
            .u8 => .{ .uint = try self.num(u8) },
            .u16 => .{ .uint = try self.num(u16) },
            .u32 => .{ .uint = try self.num(u32) },
            .u64 => .{ .uint = try self.num(u64) },
            .i8 => .{ .int = try self.num(i8) },
            .i16 => .{ .int = try self.num(i16) },
            .i32 => .{ .int = try self.num(i32) },
            .i64 => .{ .int = try self.num(i64) },
            .f32 => .{ .float = try self.num(f32) },
            .f64 => .{ .float = try self.num(f64) },
            .bool => .{ .bool = (try self.num(u8)) != 0 },
            else => error.BadValueType,
        };
    }

    fn value(self: *Reader) Error!Value {
        const t: ValueType = @fromBackingInt(@intCast(try self.num(u32)));
        switch (t) {
            .string => return .{ .string = try self.string() },
            .array => {
                const et: ValueType = @fromBackingInt(@intCast(try self.num(u32)));
                const len = std.math.cast(usize, try self.num(u64)) orelse return error.Truncated;
                const start = self.pos;
                if (et == .string) {
                    for (0..len) |_| _ = try self.string();
                } else {
                    const sz = scalarSize(et) orelse return error.BadValueType;
                    _ = try self.take(std.math.mul(u64, sz, len) catch return error.Truncated);
                }
                return .{ .array = .{ .elem_type = et, .len = len, .bytes = self.buf[start..self.pos] } };
            },
            else => return self.scalar(t),
        }
    }
};

pub const File = struct {
    arena: std.heap.ArenaAllocator,
    /// The mmaps we own (open: one per shard), none for parse() over caller bytes.
    maps: []const Map = &.{},
    version: u32,
    kv: std.StringHashMapUnmanaged(Value) = .empty,
    tensors: std.StringArrayHashMapUnmanaged(Tensor) = .empty,

    /// A shard of a split GGUF (`<prefix>-00001-of-0000N.gguf`, split.count
    /// N > 1) opens its siblings too: one tensor table over every shard, the
    /// metadata of the first.
    pub fn open(allocator: std.mem.Allocator, path: []const u8) !File {
        const first = try mapFile(path);
        var f = parse(allocator, first) catch |err| {
            std.posix.munmap(first);
            return err;
        };
        f.maps = f.arena.allocator().dupe(Map, &.{first}) catch |err| {
            std.posix.munmap(first);
            f.deinit();
            return err;
        };
        errdefer f.deinit();
        try f.openSiblings(path);
        return f;
    }

    fn mapFile(path: []const u8) !Map {
        var pbuf: [std.fs.max_path_bytes]u8 = undefined;
        if (path.len >= pbuf.len) return error.NameTooLong;
        @memcpy(pbuf[0..path.len], path);
        pbuf[path.len] = 0;
        const fd = std.c.open(pbuf[0..path.len :0], .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.FileNotFound;
        defer _ = std.c.close(fd); // the mapping outlives the fd
        // lseek, not fstat: std.c.fstat is void on Linux in Zig 0.17.
        const size = std.c.lseek(fd, 0, std.c.SEEK.END);
        if (size < 0) return error.StatFailed;
        return std.posix.mmap(null, @intCast(size), .{ .READ = true }, .{ .TYPE = .PRIVATE }, fd, 0);
    }

    /// Shards are named like llama.cpp's `llama_split_path`; the others carry
    /// only the split.* keys, so their metadata is not merged.
    fn openSiblings(self: *File, path: []const u8) !void {
        const count = self.getInt("split.count") orelse return;
        if (count <= 1) return;
        if ((self.getInt("split.no") orelse return error.BadSplit) != 0) return error.SplitNotFirstShard;
        const n = std.math.cast(usize, count) orelse return error.BadSplit;
        const a = self.arena.allocator();
        const suffix = try std.fmt.allocPrint(a, "-00001-of-{d:0>5}.gguf", .{n});
        if (!std.mem.endsWith(u8, path, suffix)) return error.BadSplit;
        const maps = try a.alloc(Map, n);
        maps[0] = self.maps[0];
        for (1..n) |i| {
            const sibling = try std.fmt.allocPrint(a, "{s}-{d:0>5}-of-{d:0>5}.gguf", .{ path[0 .. path.len - suffix.len], i + 1, n });
            maps[i] = try mapFile(sibling);
            self.maps = maps[0 .. i + 1];
            var shard = try parse(self.arena.child_allocator, maps[i]);
            defer shard.deinit();
            if (shard.getInt("split.no") != @as(i64, @intCast(i)) or shard.getInt("split.count") != count) return error.BadSplit;
            var it = shard.tensors.iterator();
            while (it.next()) |e| {
                // Names and data are slices of the shard's map, which we own now.
                const gop = try self.tensors.getOrPut(a, e.key_ptr.*);
                if (gop.found_existing) return error.BadSplit;
                gop.value_ptr.* = e.value_ptr.*;
            }
        }
        if (self.getInt("split.tensors.count")) |total| if (total != self.tensors.count()) return error.BadSplit;
    }

    /// `bytes` must outlive the File.
    pub fn parse(allocator: std.mem.Allocator, bytes: []const u8) Error!File {
        var f = File{ .arena = std.heap.ArenaAllocator.init(allocator), .version = 0 };
        errdefer f.arena.deinit();
        const a = f.arena.allocator();

        var r = Reader{ .buf = bytes };
        if (!std.mem.eql(u8, try r.take(4), "GGUF")) return error.BadMagic;
        f.version = try r.num(u32);
        if (f.version < 2 or f.version > 3) return error.UnsupportedVersion;
        const n_tensors = try r.num(u64);
        const n_kv = try r.num(u64);

        for (0..n_kv) |_| {
            const key = try r.string();
            const gop = try f.kv.getOrPut(a, key);
            if (gop.found_existing) return error.DuplicateKey;
            gop.value_ptr.* = try r.value();
        }

        const Pending = struct { name: []const u8, t: Tensor, offset: u64 };
        const pending = try a.alloc(Pending, std.math.cast(usize, n_tensors) orelse return error.Truncated);
        for (pending) |*p| {
            p.name = try r.string();
            p.t = .{ .dims = @splat(1), .n_dims = try r.num(u32), .ty = undefined, .data = &.{} };
            if (p.t.n_dims > 4) return error.BadTensor;
            for (p.t.dims[0..p.t.n_dims]) |*d| d.* = try r.num(u64);
            p.t.ty = @fromBackingInt(@intCast(try r.num(u32)));
            p.offset = try r.num(u64);
        }

        const alignment: u64 = if (f.kv.get("general.alignment")) |v| @intCast(v.asInt() orelse 32) else 32;
        if (alignment == 0) return error.BadTensor;
        const data_start = (r.pos + alignment - 1) / alignment * alignment;

        for (pending) |p| {
            var t = p.t;
            // Unknown types stay in the directory with empty data so callers can
            // report exactly which type blocked the load.
            if (t.ty.supported()) {
                if (t.dims[0] % t.ty.blockElems() != 0) return error.BadTensor;
                const size = t.elems() / t.ty.blockElems() * t.ty.blockBytes();
                const start = data_start + p.offset;
                if (start > bytes.len or size > bytes.len - start) return error.Truncated;
                t.data = bytes[@intCast(start)..][0..@intCast(size)];
            }
            try f.tensors.put(a, p.name, t);
        }
        return f;
    }

    pub fn deinit(self: *File) void {
        for (self.maps) |m| std.posix.munmap(m);
        self.arena.deinit();
    }

    pub fn getString(self: *const File, key: []const u8) ?[]const u8 {
        const v = self.kv.get(key) orelse return null;
        return if (v == .string) v.string else null;
    }

    pub fn getInt(self: *const File, key: []const u8) ?i64 {
        return (self.kv.get(key) orelse return null).asInt();
    }

    pub fn getFloat(self: *const File, key: []const u8) ?f64 {
        const v = self.kv.get(key) orelse return null;
        return if (v == .float) v.float else null;
    }

    pub fn getBool(self: *const File, key: []const u8) ?bool {
        const v = self.kv.get(key) orelse return null;
        return if (v == .bool) v.bool else null;
    }

    pub fn getArray(self: *const File, key: []const u8) ?Array {
        const v = self.kv.get(key) orelse return null;
        return if (v == .array) v.array else null;
    }

    /// First tensor whose type we have no kernels for, null when all are servable.
    pub fn firstUnsupportedTensor(self: *const File) ?[]const u8 {
        var it = self.tensors.iterator();
        while (it.next()) |e| if (!e.value_ptr.ty.supported()) return e.key_ptr.*;
        return null;
    }
};

const TestWriter = struct {
    buf: std.ArrayList(u8) = .empty,
    a: std.mem.Allocator = std.testing.allocator,

    fn num(self: *TestWriter, v: anytype) !void {
        try self.buf.appendSlice(self.a, std.mem.asBytes(&v));
    }
    fn str(self: *TestWriter, s: []const u8) !void {
        try self.num(@as(u64, s.len));
        try self.buf.appendSlice(self.a, s);
    }
};

/// Two KVs (string + string array) and one Q8_0 tensor [32 x 2].
fn buildTestFile(w: *TestWriter) !void {
    try w.buf.appendSlice(w.a, "GGUF");
    try w.num(@as(u32, 3));
    try w.num(@as(u64, 1));
    try w.num(@as(u64, 2));
    try w.str("general.architecture");
    try w.num(@as(u32, 8));
    try w.str("qwen35");
    try w.str("tokenizer.ggml.tokens");
    try w.num(@as(u32, 9));
    try w.num(@as(u32, 8));
    try w.num(@as(u64, 2));
    try w.str("hello");
    try w.str("world");
    try w.str("blk.0.ffn_up.weight");
    try w.num(@as(u32, 2));
    try w.num(@as(u64, 32));
    try w.num(@as(u64, 2));
    try w.num(@as(u32, 8));
    try w.num(@as(u64, 0));
    while (w.buf.items.len % 32 != 0) try w.buf.append(w.a, 0);
    try w.buf.appendNTimes(w.a, 7, 2 * 34);
}

test "parse reads metadata, string arrays and aligned tensor data" {
    var w = TestWriter{};
    defer w.buf.deinit(w.a);
    try buildTestFile(&w);

    var f = try File.parse(std.testing.allocator, w.buf.items);
    defer f.deinit();
    try std.testing.expectEqualStrings("qwen35", f.getString("general.architecture").?);
    var it = f.getArray("tokenizer.ggml.tokens").?.strings();
    try std.testing.expectEqualStrings("hello", it.next().?);
    try std.testing.expectEqualStrings("world", it.next().?);
    try std.testing.expect(it.next() == null);

    const t = f.tensors.get("blk.0.ffn_up.weight").?;
    try std.testing.expectEqual(GgmlType.q8_0, t.ty);
    try std.testing.expectEqual(@as(u64, 64), t.elems());
    try std.testing.expectEqual(@as(usize, 68), t.data.len);
    try std.testing.expectEqual(@as(u8, 7), t.data[0]);
    try std.testing.expect(f.firstUnsupportedTensor() == null);
}

test "parse rejects a file cut short anywhere" {
    var w = TestWriter{};
    defer w.buf.deinit(w.a);
    try buildTestFile(&w);
    for (0..w.buf.items.len) |n| {
        if (File.parse(std.testing.allocator, w.buf.items[0..n])) |*f| {
            @constCast(f).deinit();
            return error.TestExpectedError;
        } else |_| {}
    }
}

const TestShard = struct {
    /// null: no split.no key.
    no: ?u16,
    count: u16,
    total: i32,
    /// Q8_0 [32 x 2] tensors, every data byte = `fill`.
    tensors: []const struct { name: []const u8, fill: u8 },
};

/// One shard of a split GGUF as llama.cpp writes it: the first shard carries
/// the model metadata, every shard the three split.* keys.
fn buildShard(w: *TestWriter, sh: TestShard) !void {
    try w.buf.appendSlice(w.a, "GGUF");
    try w.num(@as(u32, 3));
    try w.num(@as(u64, sh.tensors.len));
    const first = (sh.no orelse 0) == 0;
    try w.num(@as(u64, 2) + @intFromBool(first) + @intFromBool(sh.no != null));
    if (first) {
        try w.str("general.architecture");
        try w.num(@as(u32, 8));
        try w.str("qwen35");
    }
    if (sh.no) |no| {
        try w.str("split.no");
        try w.num(@as(u32, 2));
        try w.num(no);
    }
    try w.str("split.count");
    try w.num(@as(u32, 2));
    try w.num(sh.count);
    try w.str("split.tensors.count");
    try w.num(@as(u32, 5));
    try w.num(sh.total);
    for (sh.tensors, 0..) |t, i| {
        try w.str(t.name);
        try w.num(@as(u32, 2));
        try w.num(@as(u64, 32));
        try w.num(@as(u64, 2));
        try w.num(@as(u32, 8));
        try w.num(@as(u64, 96 * i));
    }
    for (sh.tensors) |t| {
        while (w.buf.items.len % 32 != 0) try w.buf.append(w.a, 0);
        try w.buf.appendNTimes(w.a, t.fill, 68);
    }
}

/// Writes `shards` as `<dir>/<names[i]>`, returns the directory's absolute path.
fn writeShards(dir: std.testing.TmpDir, names: []const []const u8, shards: []const TestShard, path_buf: []u8) ![]const u8 {
    const io = std.testing.io;
    for (names, shards) |name, sh| {
        var w = TestWriter{};
        defer w.buf.deinit(w.a);
        try buildShard(&w, sh);
        try dir.dir.writeFile(io, .{ .sub_path = name, .data = w.buf.items });
    }
    return path_buf[0..try dir.dir.realPath(io, path_buf)];
}

const shard_names = [_][]const u8{ "m-00001-of-00002.gguf", "m-00002-of-00002.gguf" };

test "a split GGUF opens from its first shard as one tensor table over every shard" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = try writeShards(tmp, &shard_names, &.{
        .{ .no = 0, .count = 2, .total = 3, .tensors = &.{ .{ .name = "token_embd.weight", .fill = 1 }, .{ .name = "blk.0.ffn_up.weight", .fill = 2 } } },
        .{ .no = 1, .count = 2, .total = 3, .tensors = &.{.{ .name = "per_layer_token_embd.weight", .fill = 3 }} },
    }, &buf);
    const path = try std.fmt.allocPrint(std.testing.allocator, "{s}/{s}", .{ root, shard_names[0] });
    defer std.testing.allocator.free(path);

    var f = try File.open(std.testing.allocator, path);
    defer f.deinit();
    try std.testing.expectEqualStrings("qwen35", f.getString("general.architecture").?);
    try std.testing.expectEqual(@as(usize, 3), f.tensors.count());
    for ([_][]const u8{ "token_embd.weight", "blk.0.ffn_up.weight", "per_layer_token_embd.weight" }, 1..) |name, fill| {
        const t = f.tensors.get(name).?;
        try std.testing.expectEqual(@as(usize, 68), t.data.len);
        for (t.data) |b| try std.testing.expectEqual(@as(u8, @intCast(fill)), b);
    }
}

test "a later shard opened directly is refused by name" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = try writeShards(tmp, &shard_names, &.{
        .{ .no = 0, .count = 2, .total = 2, .tensors = &.{.{ .name = "a.weight", .fill = 1 }} },
        .{ .no = 1, .count = 2, .total = 2, .tensors = &.{.{ .name = "b.weight", .fill = 2 }} },
    }, &buf);
    const path = try std.fmt.allocPrint(std.testing.allocator, "{s}/{s}", .{ root, shard_names[1] });
    defer std.testing.allocator.free(path);
    try std.testing.expectError(error.SplitNotFirstShard, File.open(std.testing.allocator, path));
}

test "a split set that does not add up is refused" {
    const Case = struct { names: []const []const u8, shards: []const TestShard, want: anyerror };
    const a1 = [_]TestShard{.{ .no = 0, .count = 2, .total = 2, .tensors = &.{.{ .name = "a.weight", .fill = 1 }} }};
    const cases = [_]Case{
        // the second shard is missing
        .{ .names = shard_names[0..1], .shards = &a1, .want = error.FileNotFound },
        // the sibling disagrees on the shard count
        .{ .names = &shard_names, .shards = &.{ a1[0], .{ .no = 1, .count = 3, .total = 2, .tensors = &.{.{ .name = "b.weight", .fill = 2 }} } }, .want = error.BadSplit },
        // the sibling claims to be the first shard
        .{ .names = &shard_names, .shards = &.{ a1[0], .{ .no = 0, .count = 2, .total = 2, .tensors = &.{.{ .name = "b.weight", .fill = 2 }} } }, .want = error.BadSplit },
        // fewer tensors than split.tensors.count
        .{ .names = &shard_names, .shards = &.{ .{ .no = 0, .count = 2, .total = 3, .tensors = &.{.{ .name = "a.weight", .fill = 1 }} }, .{ .no = 1, .count = 2, .total = 3, .tensors = &.{.{ .name = "b.weight", .fill = 2 }} } }, .want = error.BadSplit },
        // the same tensor in two shards
        .{ .names = &shard_names, .shards = &.{ a1[0], .{ .no = 1, .count = 2, .total = 2, .tensors = &.{.{ .name = "a.weight", .fill = 2 }} } }, .want = error.BadSplit },
        // split.count > 1 on a file not named like a shard
        .{ .names = &.{"m.gguf"}, .shards = &a1, .want = error.BadSplit },
        // split.count > 1 without split.no
        .{ .names = shard_names[0..1], .shards = &.{.{ .no = null, .count = 2, .total = 1, .tensors = &.{.{ .name = "a.weight", .fill = 1 }} }}, .want = error.BadSplit },
    };
    for (cases) |c| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const root = try writeShards(tmp, c.names, c.shards, &buf);
        const path = try std.fmt.allocPrint(std.testing.allocator, "{s}/{s}", .{ root, c.names[0] });
        defer std.testing.allocator.free(path);
        try std.testing.expectError(c.want, File.open(std.testing.allocator, path));
    }
}

test "real model file parses (set MLX_SERVE_GGUF_TEST_MODEL)" {
    const path = std.mem.span(std.c.getenv("MLX_SERVE_GGUF_TEST_MODEL") orelse return error.SkipZigTest);
    var f = try File.open(std.testing.allocator, path);
    defer f.deinit();
    try std.testing.expect(f.getString("general.architecture").?.len > 0);
    try std.testing.expect(f.firstUnsupportedTensor() == null);
    try std.testing.expect(f.getArray("tokenizer.ggml.tokens").?.len > 10_000);

    // Last block of the last tensor must be readable (catches offset/size math).
    const t = f.tensors.values()[f.tensors.count() - 1];
    var y: [256]f32 = undefined;
    if (t.ty.isQuantized()) {
        const bb = t.ty.blockBytes();
        try quants.dequantize(t.ty, t.data[t.data.len - bb ..], y[0..t.ty.blockElems()]);
        for (y[0..t.ty.blockElems()]) |v| try std.testing.expect(std.math.isFinite(v));
    }
}
