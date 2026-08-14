// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Portable set-bit iterators.
//!
//! These replace `arrow_buffer::bit_iterator::{BitIndexIterator, BitSliceIterator}`, whose
//! shared `UnalignedBitChunk` walks bitmap words through `align_to::<u64>()` and reads them
//! native-endian — byte-swapped relative to the LSB-first bitmap contract on big-endian
//! hosts. The implementations here build on [`BitChunks`], whose word loads are explicitly
//! little-endian on every target.

use crate::bit::BitChunkIterator;
use crate::bit::BitChunks;

/// Iterator over the indices of set bits, in ascending order.
#[derive(Debug)]
pub struct BitIndexIterator<'a> {
    words: BitChunkIterator<'a>,
    remainder: Option<u64>,
    current: u64,
    base: usize,
    next_base: usize,
}

impl<'a> BitIndexIterator<'a> {
    /// Create a new iterator over `len` bits of `buffer`, starting at bit `offset`.
    pub fn new(buffer: &'a [u8], offset: usize, len: usize) -> Self {
        let chunks = BitChunks::new(buffer, offset, len);
        Self {
            remainder: Some(chunks.remainder_bits()),
            words: chunks.iter(),
            current: 0,
            base: 0,
            next_base: 0,
        }
    }

    #[inline]
    fn next_word(&mut self) -> Option<u64> {
        self.words.next().or_else(|| self.remainder.take())
    }
}

impl Iterator for BitIndexIterator<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        loop {
            if self.current != 0 {
                let bit = self.current.trailing_zeros() as usize;
                self.current &= self.current - 1;
                return Some(self.base + bit);
            }
            self.current = self.next_word()?;
            self.base = self.next_base;
            self.next_base += 64;
        }
    }
}

/// Iterator over `(start, end)` ranges of consecutive set bits, in ascending order.
#[derive(Debug)]
pub struct BitSliceIterator<'a> {
    words: BitChunkIterator<'a>,
    remainder: Option<u64>,
    len: usize,
    word: u64,
    /// Bit cursor within `word`; 64 means the word is fully consumed.
    bit: usize,
    /// Absolute bit index of bit 0 of `word`.
    base: usize,
    next_base: usize,
    run_start: Option<usize>,
}

impl<'a> BitSliceIterator<'a> {
    /// Create a new iterator over `len` bits of `buffer`, starting at bit `offset`.
    pub fn new(buffer: &'a [u8], offset: usize, len: usize) -> Self {
        let chunks = BitChunks::new(buffer, offset, len);
        Self {
            remainder: Some(chunks.remainder_bits()),
            words: chunks.iter(),
            len,
            word: 0,
            bit: 64,
            base: 0,
            next_base: 0,
            run_start: None,
        }
    }

    #[inline]
    fn next_word(&mut self) -> Option<u64> {
        self.words.next().or_else(|| self.remainder.take())
    }
}

impl Iterator for BitSliceIterator<'_> {
    type Item = (usize, usize);

    #[inline]
    fn next(&mut self) -> Option<(usize, usize)> {
        loop {
            if self.bit == 64 {
                let Some(word) = self.next_word() else {
                    // The words always cover `len` bits, so an open run can only reach here
                    // when the final word ends in set bits.
                    return self.run_start.take().map(|start| (start, self.len));
                };
                self.word = word;
                self.bit = 0;
                self.base = self.next_base;
                self.next_base += 64;
            }
            let remaining = self.word >> self.bit;
            match self.run_start {
                None => {
                    if remaining == 0 {
                        self.bit = 64;
                    } else {
                        let zeros = remaining.trailing_zeros() as usize;
                        self.run_start = Some(self.base + self.bit + zeros);
                        self.bit += zeros;
                    }
                }
                Some(start) => {
                    let ones = remaining.trailing_ones() as usize;
                    if self.bit + ones == 64 {
                        // The run continues into the next word.
                        self.bit = 64;
                    } else {
                        self.bit += ones;
                        self.run_start = None;
                        return Some((start, self.base + self.bit));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::bit::BitBuffer;

    #[rstest]
    #[case(0, 0)]
    #[case(0, 5)]
    #[case(0, 64)]
    #[case(0, 65)]
    #[case(0, 200)]
    #[case(0, 1000)]
    #[case(3, 200)]
    #[case(7, 130)]
    #[case(13, 64)]
    fn test_iterators_match_scalar_walk(#[case] offset: usize, #[case] len: usize) {
        let base = BitBuffer::collect_bool(offset + len, |i| i % 5 == 0 || i % 7 == 0);
        let view = BitBuffer::new_with_offset(base.inner().clone(), len, offset);

        let expected_indices: Vec<usize> = (0..len).filter(|i| view.value(*i)).collect();
        let indices: Vec<usize> =
            BitIndexIterator::new(view.inner().as_slice(), view.offset(), len).collect();
        assert_eq!(indices, expected_indices);

        let mut expected_slices: Vec<(usize, usize)> = Vec::new();
        for &i in &expected_indices {
            match expected_slices.last_mut() {
                Some((_, end)) if *end == i => *end = i + 1,
                _ => expected_slices.push((i, i + 1)),
            }
        }
        let slices: Vec<(usize, usize)> =
            BitSliceIterator::new(view.inner().as_slice(), view.offset(), len).collect();
        assert_eq!(slices, expected_slices);
    }

    #[test]
    fn test_slice_iterator_all_set_and_chunk_aligned_tail() {
        let buf = BitBuffer::collect_bool(128, |_| true);
        let slices: Vec<_> = BitSliceIterator::new(buf.inner().as_slice(), 0, 128).collect();
        assert_eq!(slices, vec![(0, 128)]);
    }
}
