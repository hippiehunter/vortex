// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Flat `u8` transition table DFA for contains matching (`LIKE '%needle%'`).
//!
//! Short needles (at most 127 bytes) fold the "previous code was ESCAPE"
//! condition into the `u8` state space, so every compressed byte is a single
//! dependent table lookup with no escape branch. Longer needles retain the
//! compact escape-sentinel strategy, which keeps the public needle limit.
//!
//! ## Construction (needle = `"aba"`, symbols = `[0:"ab", 1:"ba"]`)
//!
//! ### Step 1: KMP (Knuth–Morris–Pratt) byte-level transition table
//!
//! See: <https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm>
//!
//! Build a `(state × byte) → state` table using the KMP failure function.
//! States 0..2 track match progress, state 3 is accept (sticky).
//!
//! ```text
//!         Input byte
//! State   'a'    'b'    other
//! ─────   ────   ────   ─────
//!   0      1      0      0      ← want 'a'
//!   1      1      2      0      ← matched "a", want 'b' (KMP: 'a'→stay at 1)
//!   2      3✓     0      0      ← matched "ab", want 'a'
//!   3✓     3✓     3✓     3✓     ← accept (sticky)
//! ```
//!
//! ### Step 2: Symbol-level transitions
//!
//! For each `(state, symbol)` pair, simulate feeding the symbol's bytes
//! through the byte table:
//!
//! ```text
//! Symbol 0 = "ab" (2 bytes):
//!   state 0 + 'a' → 1, + 'b' → 2  ⟹ sym_trans[0][0] = 2
//!   state 1 + 'a' → 1, + 'b' → 2  ⟹ sym_trans[1][0] = 2
//!   state 2 + 'a' → 3✓             ⟹ sym_trans[2][0] = 3✓ (accept)
//!
//! Symbol 1 = "ba" (2 bytes):
//!   state 0 + 'b' → 0, + 'a' → 1  ⟹ sym_trans[0][1] = 1
//!   state 1 + 'b' → 2, + 'a' → 3✓ ⟹ sym_trans[1][1] = 3✓ (accept)
//!   state 2 + 'b' → 0, + 'a' → 1  ⟹ sym_trans[2][1] = 1
//! ```
//!
//! ### Step 3: Fused 256-wide table with escape sentinel
//!
//! Merge symbol transitions into a 256-wide table. Code bytes 0–1 use symbol
//! transitions, code 255 (ESCAPE_CODE) maps to the sentinel (4), and
//! unused code bytes default to 0:
//!
//! ```text
//!              Code byte
//! State   0("ab") 1("ba") 2..254  255(ESC)
//! ─────   ─────── ─────── ──────  ────────
//!   0       2       1       0       4(S)
//!   1       2       3✓      0       4(S)
//!   2       3✓      1       0       4(S)
//!   3✓      3✓      3✓      3✓      3✓
//! ```
//!
//! When the scanner sees sentinel (4), it reads the next byte and looks it
//! up in the byte-level escape table (from step 1).
use fsst::Symbol;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::build_fused_table;
use super::build_symbol_transitions;
use super::kmp_byte_transitions;

/// Flat `u8` transition table DFA for contains matching.
///
/// The backend is selected solely from the pattern length; both execute the
/// same byte-level KMP automaton over decoded FSST symbols.
pub(crate) struct FlatContainsDfa {
    transitions: ContainsTransitions,
    accept_state: u8,
}

enum ContainsTransitions {
    /// Short needles have enough `u8` state space to encode "the previous
    /// code was ESCAPE" directly. Every compressed byte is then one dependent
    /// table lookup with no unpredictable escape branch.
    EscapeFolded(Vec<u8>),
    /// Long needles retain the compact sentinel representation so the public
    /// 254-byte limit is unchanged.
    Sentinel {
        symbols: Vec<u8>,
        escaped_bytes: Vec<u8>,
        sentinel: u8,
    },
}

impl FlatContainsDfa {
    /// Maximum needle length: need accept + sentinel to fit in u8.
    pub(crate) const MAX_NEEDLE_LEN: usize = u8::MAX as usize - 1;

    pub(crate) fn new(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        needle: &[u8],
    ) -> VortexResult<Self> {
        if needle.len() > Self::MAX_NEEDLE_LEN {
            vortex_bail!(
                "needle length {} exceeds maximum {} for flat contains DFA",
                needle.len(),
                Self::MAX_NEEDLE_LEN
            );
        }

        let accept_state = u8::try_from(needle.len())
            .vortex_expect("FlatContainsDfa: accept state must fit into u8");
        let n_states = accept_state + 1;
        let byte_table = kmp_byte_transitions(needle);
        let sym_trans =
            build_symbol_transitions(symbols, symbol_lengths, &byte_table, n_states, accept_state);

        let transitions = if needle.len() <= 127 {
            let pending_base = n_states;
            let total_states = usize::from(n_states) + usize::from(accept_state);
            let mut folded = build_fused_table(
                &sym_trans,
                symbols.len(),
                n_states,
                |state| {
                    if state == accept_state {
                        accept_state
                    } else {
                        pending_base + state
                    }
                },
                0,
            );
            folded.resize(total_states * 256, 0);
            for state in 0..accept_state {
                let source = usize::from(state) * 256;
                let target = usize::from(pending_base + state) * 256;
                folded[target..target + 256].copy_from_slice(&byte_table[source..source + 256]);
            }
            ContainsTransitions::EscapeFolded(folded)
        } else {
            let sentinel = n_states;
            ContainsTransitions::Sentinel {
                symbols: build_fused_table(&sym_trans, symbols.len(), n_states, |_| sentinel, 0),
                escaped_bytes: byte_table,
                sentinel,
            }
        };

        Ok(Self {
            transitions,
            accept_state,
        })
    }

    pub(crate) fn matches(&self, codes: &[u8]) -> bool {
        match &self.transitions {
            ContainsTransitions::EscapeFolded(transitions) => {
                let mut state = 0u8;
                for &code in codes {
                    state = transitions[usize::from(state) * 256 + usize::from(code)];
                    if state == self.accept_state {
                        return true;
                    }
                }
                false
            }
            ContainsTransitions::Sentinel {
                symbols,
                escaped_bytes,
                sentinel,
            } => self.matches_sentinel(codes, symbols, escaped_bytes, *sentinel),
        }
    }

    fn matches_sentinel(
        &self,
        codes: &[u8],
        transitions: &[u8],
        escape_transitions: &[u8],
        sentinel: u8,
    ) -> bool {
        let mut state = 0u8;
        let mut pos = 0;
        while pos < codes.len() {
            let code = codes[pos];
            pos += 1;
            let next = transitions[usize::from(state) * 256 + usize::from(code)];
            if next == sentinel {
                if pos >= codes.len() {
                    return false;
                }
                let b = codes[pos];
                pos += 1;
                state = escape_transitions[usize::from(state) * 256 + usize::from(b)];
            } else {
                state = next;
            }
            if state == self.accept_state {
                return true;
            }
        }
        false
    }
}
