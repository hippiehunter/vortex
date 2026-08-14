// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cross-endian serde conformance.
//!
//! [`golden_array`] is serialized to one checked-in blob per byte order (regenerated with the
//! ignored [`generate_golden`] test). Each `decode_*` test decodes both blobs, so on a
//! little-endian host the big-endian blob exercises the cross-endian buffer swap in
//! `SerializedArray::decode`, and under big-endian emulation the little-endian blob does.

use std::path::PathBuf;

use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DecimalDType;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::registry::ReadContext;

/// One array per buffer-owning encoding with endian-sensitive contents, plus nullability so
/// validity bitmaps ride along.
fn golden_array() -> VortexResult<ArrayRef> {
    let ints = PrimitiveArray::new(
        buffer![1i32, -2, 3, i32::MAX, i32::MIN, 0, 42, -42],
        Validity::from_iter([true, true, false, true, true, false, true, true]),
    );
    let longs = PrimitiveArray::from_iter([
        1u64,
        0xDEAD_BEEF_CAFE_F00D,
        3,
        u64::MAX,
        0,
        0x0102_0304_0506_0708,
        7,
        8,
    ]);
    let floats = PrimitiveArray::from_iter([
        1.5f64,
        -2.25,
        0.0,
        f64::MAX,
        f64::MIN_POSITIVE,
        -0.0,
        6.0,
        7.5,
    ]);
    let decimals = DecimalArray::new(
        buffer![
            1i128,
            -1,
            i128::MAX,
            i128::MIN,
            0,
            0x0102_0304_0506_0708_090A_0B0C_0D0E_0F10,
            -99,
            100
        ],
        DecimalDType::new(38, 2),
        Validity::NonNullable,
    );
    let bools = BoolArray::from_iter([true, false, true, true, false, false, true, false]);
    let strings = VarBinViewArray::from_iter_nullable_str([
        Some("short"),
        Some("a considerably longer string that needs a reference view"),
        None,
        Some(""),
        Some("exactly12byt"),
        Some("thirteen-byte"),
        Some("x"),
        Some("another long value stored out of line in the data buffer"),
    ]);

    Ok(StructArray::from_fields(&[
        ("ints", ints.into_array()),
        ("longs", longs.into_array()),
        ("floats", floats.into_array()),
        ("decimals", decimals.into_array()),
        ("bools", bools.into_array()),
        ("strings", strings.into_array()),
    ])?
    .into_array())
}

fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn serialize_golden(array: &ArrayRef, array_ctx: &ArrayContext) -> VortexResult<Vec<u8>> {
    let session = array_session();
    let buffers = array.serialize(array_ctx, &session, &SerializeOptions::default())?;
    let mut bytes = Vec::new();
    for buf in buffers {
        bytes.extend_from_slice(buf.as_ref());
    }
    Ok(bytes)
}

/// Regenerates the golden blob for this host's byte order. Run natively for the little-endian
/// blob and under big-endian emulation (e.g. `cross test --target s390x-unknown-linux-gnu`)
/// for the big-endian one.
#[test]
#[ignore = "regenerates the golden blob for this host's byte order"]
fn generate_golden() -> VortexResult<()> {
    let name = if cfg!(target_endian = "big") {
        "golden.be.vxarr"
    } else {
        "golden.le.vxarr"
    };
    let array = golden_array()?;
    let bytes = serialize_golden(&array, &ArrayContext::empty())?;
    let path = golden_path(name);
    std::fs::create_dir_all(path.parent().vortex_expect("golden path has a parent"))?;
    std::fs::write(path, &bytes)?;
    Ok(())
}

fn decode_golden(name: &str) -> VortexResult<()> {
    let session = array_session();
    let mut ctx = session.create_execution_ctx();
    let expected = golden_array()?;

    // Serializing the expected array interns encodings into the context in traversal order,
    // reproducing the id table the golden blob was generated with.
    let array_ctx = ArrayContext::empty();
    serialize_golden(&expected, &array_ctx)?;

    let bytes = std::fs::read(golden_path(name))?;
    let parts = SerializedArray::try_from(ByteBuffer::from(bytes))?;
    let decoded = parts.decode(
        expected.dtype(),
        expected.len(),
        &ReadContext::new(array_ctx.to_ids()),
        &session,
    )?;

    assert_arrays_eq!(decoded, expected, &mut ctx);
    Ok(())
}

#[test]
fn decode_little_endian_golden() -> VortexResult<()> {
    decode_golden("golden.le.vxarr")
}

#[test]
fn decode_big_endian_golden() -> VortexResult<()> {
    decode_golden("golden.be.vxarr")
}
