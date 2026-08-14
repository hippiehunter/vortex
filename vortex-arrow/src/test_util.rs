// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::GenericByteViewArray;
use arrow_array::types::ByteViewType;

/// Read element `i` of an arrow view array through the documented u128 value convention.
///
/// `GenericByteViewArray::value` slices the stored u128's raw memory bytes for inlined views
/// while reading the length from its *value*, which only agree on little-endian hosts. This
/// helper stays portable, so tests validate the emitted views on any host instead of arrow's
/// accessor behavior.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn arrow_view_value<T: ByteViewType>(
    array: &GenericByteViewArray<T>,
    i: usize,
) -> Vec<u8> {
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
