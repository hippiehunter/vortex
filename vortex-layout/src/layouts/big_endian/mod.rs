// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tripwire layout for files whose array data buffers are big-endian.
//!
//! The layout is pure delegation: it has exactly one child with the same dtype and row count,
//! and its reader is the child's reader. Its value is entirely in its ID — readers that
//! predate the per-array endianness tag fail loudly with
//! `Cannot read unknown layout encoding 'vortex.big_endian'` instead of silently misreading
//! big-endian buffers as little-endian values. The actual byte-order handling happens per
//! serialized array via the `endianness` field on the array flatbuffer; this wrapper carries
//! no data of its own.

use std::sync::Arc;

use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::Layout;
use crate::LayoutChildType;
use crate::LayoutDeserializeArgs;
use crate::LayoutId;
use crate::LayoutParts;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::VTable;
use crate::children::OwnedLayoutChildren;
use crate::segments::SegmentSource;

/// Big-endian tripwire layout vtable.
#[derive(Clone, Debug)]
pub struct BigEndian;

/// A transparent layout marking a file whose array data buffers are big-endian.
pub type BigEndianLayout = Layout<BigEndian>;

impl VTable for BigEndian {
    type LayoutData = ();
    type Metadata = EmptyMetadata;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex.big_endian");
        *ID
    }

    fn metadata(_layout: &Layout<Self>) -> Self::Metadata {
        EmptyMetadata
    }

    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        _metadata: &EmptyMetadata,
    ) -> VortexResult<Self::LayoutData> {
        if args.children.nchildren() != 1 {
            vortex_bail!(
                "BigEndian layout must have exactly one child, got {}",
                args.children.nchildren()
            );
        }
        if !args.segment_ids.is_empty() {
            vortex_bail!("BigEndian layout must not own segments");
        }
        Ok(())
    }

    fn child_dtype(layout: &Layout<Self>, slot: usize) -> VortexResult<DType> {
        if slot != 0 {
            vortex_bail!("BigEndian layout has no child {slot}");
        }
        Ok(layout.dtype().clone())
    }

    fn child_type(_layout: &Layout<Self>, _slot: usize) -> LayoutChildType {
        LayoutChildType::Transparent("data".into())
    }

    fn new_reader(
        layout: &Layout<Self>,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        layout
            .slot(0)?
            .vortex_expect("BigEndian layout always has one child")
            .new_reader(name, segment_source, session, ctx)
    }
}

impl Layout<BigEndian> {
    /// Wrap `child` as the root of a big-endian file.
    pub fn wrap(child: LayoutRef) -> Self {
        LayoutParts::new(
            BigEndian,
            child.dtype().clone(),
            child.row_count(),
            Vec::new(),
            OwnedLayoutChildren::layout_children(vec![child]),
            (),
        )
        .into_typed()
    }
}
