// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cross-endian file conformance.
//!
//! One golden `.vortex` file is checked in per byte order (regenerated with the ignored
//! [`generate_golden`] test, run natively for little-endian and under big-endian emulation
//! for big-endian). Decoding both files on both host byte orders exercises the per-array
//! endianness swap through the whole file stack — including compressed encodings chosen by
//! the default write strategy, and the `vortex.big_endian` tripwire layout wrapping the root
//! of the big-endian file.

#![expect(clippy::tests_outside_test_module)]

use std::path::PathBuf;
use std::sync::LazyLock;

use futures::StreamExt;
use futures::pin_mut;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::session::RuntimeSession;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

mod common;

use common::enable_all_registered_array_encodings;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    vortex_file::register_default_encodings(&session);
    enable_all_registered_array_encodings(&session);
    session
});

const ROWS: usize = 1024;

/// Compressible, mixed-type data so the default write strategy picks real encodings
/// (bitpacking, dictionaries, FSST, ALP) whose buffers all cross the endianness boundary.
fn golden_table() -> VortexResult<ArrayRef> {
    let ints = PrimitiveArray::from_iter((0..ROWS as i64).map(|i| (i % 100) * 7)).into_array();
    let small = PrimitiveArray::new(
        vortex_buffer::Buffer::from_iter((0..ROWS as u32).map(|i| i % 16)),
        Validity::from_iter((0..ROWS).map(|i| i % 11 != 0)),
    )
    .into_array();
    let floats = PrimitiveArray::from_iter((0..ROWS).map(|i| (i % 250) as f64 * 0.25)).into_array();
    let bools = BoolArray::from_iter((0..ROWS).map(|i| i % 3 == 0)).into_array();
    let words = [
        "alpha",
        "beta",
        "gamma",
        "a considerably longer repeated value",
        "delta",
    ];
    let strings =
        VarBinViewArray::from_iter_str((0..ROWS).map(|i| words[i % words.len()])).into_array();

    Ok(StructArray::new(
        FieldNames::from(["ints", "small", "floats", "bools", "strings"]),
        vec![ints, small, floats, bools, strings],
        ROWS,
        Validity::NonNullable,
    )
    .into_array())
}

fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

async fn write_golden() -> VortexResult<Vec<u8>> {
    let mut bytes = Vec::new();
    SESSION
        .write_options()
        .write(&mut bytes, golden_table()?.to_array_stream())
        .await?;
    Ok(bytes)
}

/// Regenerates the golden file for this host's byte order. Run natively for the little-endian
/// file and under big-endian emulation for the big-endian one.
#[tokio::test]
#[ignore = "regenerates the golden file for this host's byte order"]
async fn generate_golden() -> VortexResult<()> {
    let name = if cfg!(target_endian = "big") {
        "golden.be.vortex"
    } else {
        "golden.le.vortex"
    };
    let path = golden_path(name);
    std::fs::create_dir_all(path.parent().vortex_expect("golden path has a parent"))?;
    std::fs::write(path, write_golden().await?)?;
    Ok(())
}

async fn decode_golden(name: &str) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let expected = golden_table()?;

    let bytes = ByteBuffer::from(std::fs::read(golden_path(name))?);
    let vxf = SESSION.open_options().open_buffer(bytes)?;

    let stream = vxf.scan()?.into_stream()?;
    pin_mut!(stream);
    let mut chunks: Vec<ArrayRef> = Vec::new();
    while let Some(chunk) = stream.next().await {
        chunks.push(chunk?);
    }
    assert_eq!(
        chunks.len(),
        1,
        "golden table should scan as a single chunk"
    );
    let decoded = chunks
        .pop()
        .vortex_expect("one chunk")
        .execute::<StructArray>(&mut ctx)?;

    assert_arrays_eq!(decoded, expected, &mut ctx);
    Ok(())
}

#[tokio::test]
async fn decode_little_endian_golden_file() -> VortexResult<()> {
    decode_golden("golden.le.vortex").await
}

#[tokio::test]
async fn decode_big_endian_golden_file() -> VortexResult<()> {
    decode_golden("golden.be.vortex").await
}
