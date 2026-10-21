// Q2_0: one f16 scale per 64 weights, weight j in bits 2 (j % 4) of code byte
// j / 4, y = (q - 1) d. Unit u = weights [32u, 32u + 32) = code bytes [8u, 8u + 8).
template <typename O>
inline void gg_q2_0_codes(uint4 lo, uint4 hi, float d, thread O &o) {
    const uint4 sh = uint4(0, 2, 4, 6);
    for (uint i = 0; i < 4; i++) {
        o.emit(i, d * (float4((uint4(lo[i]) >> sh) & 3) - 1.0f));
        o.emit(i + 4, d * (float4((uint4(hi[i]) >> sh) & 3) - 1.0f));
    }
}

template <typename O>
inline void deq_q2_0(const device uint8_t *b, uint u, thread O &o) {
    const device uint8_t *qs = b + 2 + 8 * u;
    gg_q2_0_codes(GG_U4(qs), GG_U4(qs + 4), gg_half(b, 0), o);
}

// Split rows: the unit's 8 code bytes as one aligned load.
template <typename O>
inline void deq_q2_0_split(const device uint8_t *h, const device uint8_t *qs, uint u, thread O &o) {
    const uint2 w = *(const device uint2 *)(qs + 8 * u);
    gg_q2_0_codes(gg_bytes(w.x), gg_bytes(w.y), gg_half(h, 0), o);
}
