// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Portable reads of Arrow view-array elements.

use arrow_array::GenericByteViewArray;
use arrow_array::types::ByteViewType;

/// Read element `i` of an Arrow view array through the documented u128 value convention.
///
/// `GenericByteViewArray::value` slices the stored u128's raw memory bytes for inlined views
/// while reading the length from its *value*; the two agree only on little-endian hosts, so
/// arrow's accessor returns scrambled inline bytes on big-endian targets. This helper reads
/// the value convention directly and is correct on any host.
#[allow(clippy::cast_possible_truncation)]
pub fn arrow_view_value<T: ByteViewType>(array: &GenericByteViewArray<T>, i: usize) -> Vec<u8> {
    let v = array.views()[i];
    let len = v as u32 as usize;
    if len <= 12 {
        v.to_le_bytes()[4..4 + len].to_vec()
    } else {
        let buffer_index = (v >> 64) as u32 as usize;
        let offset = (v >> 96) as u32 as usize;
        array.data_buffers()[buffer_index][offset..offset + len].to_vec()
    }
}
