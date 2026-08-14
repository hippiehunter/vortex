// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Packed bitmaps that can be used to store boolean values.
//!
//! This module provides a wrapper on top of the `Buffer` type to store mutable and immutable
//! bitsets. The bitsets are stored in little-endian order, meaning that the least significant bit
//! of the first byte is the first bit in the bitset.
#[cfg(feature = "arrow")]
mod arrow;
mod buf;
mod buf_mut;
mod count_ones;
mod iter;
mod macros;
mod meta;
mod ops;
mod pack;
mod select;
mod view;

// `UnalignedBitChunk` and the arrow set-bit iterators built on it are deliberately not
// re-exported: they read bitmap words through `align_to::<u64>()` in native byte order, which
// scrambles LSB-first bitmaps on big-endian hosts. Use [`BitIndexIterator`] and
// [`BitSliceIterator`] from [`iter`] instead.
pub use arrow_buffer::bit_chunk_iterator::BitChunkIterator;
pub use arrow_buffer::bit_chunk_iterator::BitChunks;
pub use arrow_buffer::bit_iterator::BitIterator;
pub use buf::*;
pub use buf_mut::*;
pub use iter::*;
pub use meta::*;
pub use pack::*;
pub use view::*;

use crate::BufferMut;
use crate::ByteBufferMut;

/// Packs up to 64 boolean values into a little-endian `u64` word.
///
/// This is [`collect_bool_words`] for a single word: a full 64-bit word is materialized as a
/// `[bool; 64]` and packed with the baseline SIMD byte→bit instruction of the target; shorter
/// lengths fall back to the bit-at-a-time [`collect_bool_word_scalar`] loop.
#[inline]
pub fn collect_bool_word<F>(len: usize, f: F) -> u64
where
    F: FnMut(usize) -> bool,
{
    assert!(len <= 64, "cannot pack {len} bits into a u64 word");

    let mut word = [0u64; 1];
    collect_bool_words_inline(&mut word, len, f);
    word[0]
}

/// Pack `len` boolean values returned by `f` into the prefix of `words`, LSB-first,
/// 64 bits per `u64`. `words` must have capacity for at least `len.div_ceil(64)` entries.
///
/// `f` is invoked exactly once per index, in ascending order `0..len`.
///
/// Writes via `=` (not `|=`), so the destination need not be zero-initialised.
///
/// The word loop packs with the baseline SIMD kernel of the target (SSE2 on x86-64, NEON on
/// aarch64), which inlines fully into the caller together with the predicate and the
/// `[bool; 64]` materialization — wider kernels would sit behind a non-inlinable
/// `#[target_feature]` boundary that deoptimizes expensive predicates. See
/// [`BitBuffer::collect_bool`] for the performance note on
/// avoiding bounds checks in `f`.
///
/// Prefer this entry point for every predicate; only switch to
/// [`collect_bool_words_multiversioned`] after carefully checking that your specific `f`
/// meets its contract.
#[inline]
pub fn collect_bool_words<F>(words: &mut [u64], len: usize, f: F)
where
    F: FnMut(usize) -> bool,
{
    let num_words = len.div_ceil(64);
    assert!(
        words.len() >= num_words,
        "words slice has {} entries, need at least {num_words}",
        words.len(),
    );

    collect_bool_words_inline(words, len, f)
}

/// Read up to 8 bytes as a little-endian `u64`, zero-padding the high bytes when fewer than 8
/// bytes are supplied.
///
/// This preserves Vortex's least-significant-bit-first bitmap numbering on little- and big-endian
/// targets. For a full 8-byte slice it lowers to a single word load.
#[inline]
pub fn read_u64_le(bytes: &[u8]) -> u64 {
    debug_assert!(bytes.len() <= 8);
    let mut buf = [0u8; 8];
    buf[..bytes.len()].copy_from_slice(bytes);
    u64::from_le_bytes(buf)
}

/// Byte-reverse each bitmap word in place on big-endian hosts; a no-op on little-endian hosts.
///
/// Bit `i` of a bitmap lives at bit `i % 64` of word `i / 64`, while the bitmap's byte form
/// keeps it at bit `i % 8` of byte `i / 8`. The two agree only when each word's bytes are laid
/// out little-endian, so words must pass through this before being reinterpreted as bitmap
/// bytes. The read direction is already covered by [`read_u64_le`] and the LE-normalizing
/// chunk iterators.
#[inline]
pub fn bitmap_words_to_le(words: &mut [u64]) {
    if cfg!(target_endian = "big") {
        for w in words {
            *w = w.to_le();
        }
    }
}

/// Reinterpret a buffer of bitmap words (LSB-first, 64 bits per word) as the bitmap's byte
/// form, byte-reversing each word on big-endian hosts via [`bitmap_words_to_le`].
///
/// This is the only correct way to turn packed bitmap words into `BitBuffer` bytes; a plain
/// `into_byte_buffer()` on the word buffer produces a byte-swapped bitmap on big-endian hosts.
#[inline]
pub fn bitmap_words_into_bytes(mut words: BufferMut<u64>) -> ByteBufferMut {
    bitmap_words_to_le(words.as_mut_slice());
    words.into_byte_buffer()
}

/// Splice a packed word `w` (whose bits above the highest valid bit are zero) into
/// `words` at the given bit position.
///
/// The destination word at `bit_offset / 64` is OR'd, preserving any bits below
/// `bit_offset % 64`. When `w` has high bits that spill into the next word, those
/// bits are *assigned* (not OR'd) — so callers must ensure that next slot is zero
/// (e.g. via `BufferMut::zeroed`).
///
/// `words.len()` need only cover the slots `w` actually writes to: skipping the
/// spillover when its bits are all zero means a tail that fits entirely in the
/// leading word never touches `words[dest_word + 1]`.
#[inline]
pub fn splice_word_at_bit(words: &mut [u64], bit_offset: usize, word: u64) {
    let dest_word = bit_offset / 64;
    let bit_in_word = bit_offset % 64;
    words[dest_word] |= word << bit_in_word;
    if bit_in_word != 0 {
        let high = word >> (64 - bit_in_word);
        if high != 0 {
            words[dest_word + 1] = high;
        }
    }
}

/// Pack `len` boolean values returned by `f` into `words` starting at bit position
/// `bit_offset`, LSB-first.
///
/// Composes [`collect_bool_word`] (pack up to 64 bools into a u64) with
/// [`splice_word_at_bit`] (merge the packed word into the destination via shift-OR).
///
/// `words` must have at least `(bit_offset + len).div_ceil(64)` entries; see
/// [`splice_word_at_bit`] for zero-init requirements on words above the cursor.
#[inline]
pub fn pack_bools_into_words<F>(words: &mut [u64], bit_offset: usize, len: usize, mut f: F)
where
    F: FnMut(usize) -> bool,
{
    if len == 0 {
        return;
    }
    let num_words = (bit_offset + len).div_ceil(64);
    assert!(
        words.len() >= num_words,
        "words slice has {} entries, need at least {num_words}",
        words.len(),
    );

    let mut done = 0;
    while len - done >= 64 {
        let word = collect_bool_word(64, |bit| f(done + bit));
        splice_word_at_bit(words, bit_offset + done, word);
        done += 64;
    }
    let tail = len - done;
    if tail > 0 {
        let word = collect_bool_word(tail, |bit| f(done + bit));
        splice_word_at_bit(words, bit_offset + done, word);
    }
}

/// Get the bit value at `index` out of `buf`.
///
/// # Panics
///
/// Panics if `index` is not between 0 and length of `buf * 8`.
#[inline(always)]
pub fn get_bit(buf: &[u8], index: usize) -> bool {
    buf[index / 8] & (1 << (index % 8)) != 0
}

/// Get the bit value at `index` out of `buf` without bounds checking.
///
/// # Safety
///
/// `index` must be between 0 and length of `buf * 8`.
#[inline(always)]
pub unsafe fn get_bit_unchecked(buf: *const u8, index: usize) -> bool {
    (unsafe { *buf.add(index / 8) } & (1 << (index % 8))) != 0
}

/// Set the bit value at `index` in `buf` without bounds checking.
///
/// # Safety
///
/// `index` must be between 0 and length of `buf * 8`.
#[inline(always)]
pub unsafe fn set_bit_unchecked(buf: *mut u8, index: usize) {
    unsafe { *buf.add(index / 8) |= 1 << (index % 8) };
}

/// Unset the bit value at `index` in `buf` without bounds checking.
///
/// # Safety
///
/// `index` must be between 0 and length of `buf * 8`.
#[inline(always)]
pub unsafe fn unset_bit_unchecked(buf: *mut u8, index: usize) {
    unsafe { *buf.add(index / 8) &= !(1 << (index % 8)) };
}

#[cfg(test)]
mod tests {
    use super::collect_bool_word;
    use super::pack_bools_into_words;
    use super::read_u64_le;

    #[test]
    fn collect_bool_word_packs_lsb_first() {
        let word = collect_bool_word(5, |idx| idx.is_multiple_of(2));
        assert_eq!(word, 0b10101);
    }

    #[test]
    fn collect_bool_word_empty() {
        assert_eq!(collect_bool_word(0, |_| true), 0);
    }

    #[test]
    fn read_u64_le_zero_pads_tail() {
        assert_eq!(read_u64_le(&[0x34, 0x12]), 0x1234);
        assert_eq!(read_u64_le(&[0xff; 8]), u64::MAX);
    }

    #[test]
    #[should_panic(expected = "cannot pack 65 bits into a u64 word")]
    fn collect_bool_word_rejects_too_many_bits() {
        let _ = collect_bool_word(65, |_| true);
    }

    fn pack(bit_offset: usize, len: usize, f: impl Fn(usize) -> bool) -> Vec<bool> {
        let num_words = (bit_offset + len).div_ceil(64);
        let mut words = vec![0u64; num_words];
        pack_bools_into_words(&mut words, bit_offset, len, &f);
        (0..bit_offset + len)
            .map(|i| (words[i / 64] >> (i % 64)) & 1 == 1)
            .collect()
    }

    #[test]
    fn pack_bools_aligned_multi_word_with_tail() {
        let bits = pack(0, 130, |i| i.is_multiple_of(3));
        for i in 0..130 {
            assert_eq!(bits[i], i.is_multiple_of(3), "bit {i}");
        }
    }

    #[test]
    fn pack_bools_unaligned_crossing_words() {
        let bits = pack(40, 200, |i| i.is_multiple_of(7));
        assert!(bits[..40].iter().all(|&b| !b));
        for i in 0..200 {
            assert_eq!(bits[40 + i], i.is_multiple_of(7), "bit {}", 40 + i);
        }
    }

    #[test]
    fn pack_bools_preserves_low_bits_of_leading_word() {
        let mut words = vec![0u64; 2];
        words[0] = 0b11111;
        pack_bools_into_words(&mut words, 5, 70, |_| true);
        for i in 0..5 {
            assert_eq!((words[0] >> i) & 1, 1, "preserved bit {i}");
        }
        for i in 5..75 {
            assert_eq!((words[i / 64] >> (i % 64)) & 1, 1, "extended bit {i}");
        }
    }
}
