//! #86 serve: the OpenAI `stop` strings — the tool-call tail-hold, generalized.
//!
//! | lives here | lives in `bin/serve.rs` |
//! |---|---|
//! | `StopStrings`, the hold/match policy | `parse_chat`'s `stop` field, `send_emits` / `send_split`, `chat_generate`'s stop break and finish mapping |
//! | the filter's unit tests | the pipeline tests that pin the think-split interaction |
//!
//! - The mechanism is the one `toolcall.rs` built for `<tool_call>` (#29 A7): a decoded
//!   piece may END inside a marker, so the tail that could still become one is HELD
//!   BACK and `toolcall::find_marker` is the scan both run. No logic lives twice.
//! - What this filter holds is a tail that could still become ANY stop string of the
//!   request; what it lets through can never be part of one. So NO HALF of a stop
//!   string ever reaches the wire — the OpenAI contract's one hard rule for `stop`.
//!
//! Match policy (documented, pinned by tests):
//!
//! | question | answer | why |
//! |---|---|---|
//! | which stop wins? | the one at the EARLIEST byte position | llama-server stops at the earliest stop occurrence in the text |
//! | two stops at the SAME position (one contains the other)? | the LONGEST | the list is sorted longest-first and `find_marker` keeps the first slice entry on ties. The WIRE BYTES are identical either way — generation ends at the match and the tail is swallowed — so this only names which string the log line reports |
//! | what is emitted? | every content byte BEFORE the match, nothing of it or after | "sequences end generation before the sequence is emitted" |
//! | `finish_reason`? | `stop` (the caller sets it), unless a CLOSED tool call says `tool_calls` first | the same precedence EOS already has |
//! | a held prefix that never completes? | it is TEXT, and `flush()` releases it | no byte of the answer is lost to the hold — `ThinkFilter`'s rule |
//!
//! Channel contract (the caller's, pinned by `bin/serve.rs` tests):
//!
//! - This filter runs LAST on the CONTENT channel, AFTER the think split: it sees
//!   `Split.content`, never `Split.reasoning`. A stop string inside a think block is
//!   reasoning, not the answer, and does not stop the answer.
//! - Tool-call fragments (`Emit::Call`, `Emit::Args`) never pass through here; after a
//!   hit the caller swallows them too — everything generated after the stop point is cut.
//!
//! Byte accounting:
//!
//! - `dropped()` counts every content byte swallowed: the matched stop, the tail after
//!   it in the same piece, and every byte fed to `push` after the hit. `serve` prints
//!   the total in one stderr line per request, so the number is exact.
//! - `matched_at()` is the content byte the match landed on; `emitted + held + (on a
//!   hit) dropped-so-far` always account for every byte fed.

use crate::toolcall::find_marker;

/// #86: the stop-string filter of ONE request. Empty is INERT — `push` returns every
/// piece verbatim, nothing is held, `hit` is never true — so a request without `stop`
/// streams the bytes every release before #86 streamed, pinned by test.
///
/// - pure: no engine, no socket; the tests drive it at every split point directly.
#[derive(Debug, Clone, Default)]
pub struct StopStrings {
    /// the request's stop strings, LONGEST FIRST (the tie policy of the module doc);
    /// empty entries were dropped at parse — an empty string matches at byte 0 of
    /// everything and `find_marker` holds on `m.len() - 1`
    stops: Vec<String>,
    /// a tail that might still become the start of a stop string, held back
    held: String,
    /// the stop string that ended this answer; `None` until one matched
    matched: Option<String>,
    /// the content byte the match landed on (the length of everything emitted before it)
    matched_at: usize,
    /// content bytes that left as content (the offset `matched_at` names)
    emitted: usize,
    /// content bytes swallowed: the stop, the tail after it, and everything fed after
    dropped: usize,
}

impl StopStrings {
    /// the filter of a request whose `stop` list is `stops`, in the body's own order:
    /// empties drop, the rest sort LONGEST FIRST, and an empty list is the inert filter
    pub fn new(stops: &[String]) -> Self {
        let mut stops: Vec<String> = stops.iter().filter(|s| !s.is_empty()).cloned().collect();
        stops.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        StopStrings { stops, ..StopStrings::default() }
    }

    /// is there a stop list at all? An inert filter is a passthrough, and the caller's
    /// hot path may skip it entirely
    pub fn active(&self) -> bool {
        !self.stops.is_empty()
    }

    /// did a stop string match? Once it has, every later byte is swallowed and the
    /// generation loop must stop with `finish_reason` `stop`
    pub fn hit(&self) -> bool {
        self.matched.is_some()
    }

    /// the stop string that ended this answer, for the log line
    pub fn matched(&self) -> Option<&str> {
        self.matched.as_deref()
    }

    /// the content byte the match landed on: the length of everything emitted before it
    pub fn matched_at(&self) -> usize {
        self.matched_at
    }

    /// content bytes swallowed over the whole request (the stop, the tail after it,
    /// everything fed after the hit) — the number `serve`'s one stderr line carries
    pub fn dropped(&self) -> usize {
        self.dropped
    }

    /// - one CONTENT piece in, the emit-safe prefix out (possibly empty).
    /// - after a hit: empty, always — everything after the stop point is cut.
    /// - a complete stop in the buffer: everything before it leaves, the stop and the
    ///   tail after it are swallowed and counted, `hit` turns true.
    /// - no complete stop: the bytes that can never be part of one leave, a tail that
    ///   is a proper prefix of some stop is HELD — the tool-call hold, on arbitrary
    ///   strings, across any number of pieces.
    pub fn push(&mut self, piece: &str) -> String {
        if self.matched.is_some() {
            self.dropped += piece.len();
            return String::new();
        }
        if self.stops.is_empty() {
            return piece.to_string();
        }
        let mut buf = std::mem::take(&mut self.held);
        buf.push_str(piece);
        let refs: Vec<&str> = self.stops.iter().map(|s| s.as_str()).collect();
        match find_marker(&buf, &refs) {
            (Some((at, i)), _) => {
                let out = buf[..at].to_string();
                self.matched = Some(self.stops[i].clone());
                self.matched_at = self.emitted + at;
                self.dropped += buf.len() - at;
                self.emitted += at;
                out
            }
            (None, safe) => {
                let out = buf[..safe].to_string();
                self.held = buf[safe..].to_string();
                self.emitted += out.len();
                out
            }
        }
    }

    /// - end of generation: a held prefix never completed a stop, so it is TEXT and it
    ///   leaves here — no byte of the answer is lost to the hold. After a hit the
    ///   answer already ended; the flush is empty.
    pub fn flush(&mut self) -> String {
        if self.matched.is_some() {
            return String::new();
        }
        let out = std::mem::take(&mut self.held);
        self.emitted += out.len();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// drive the filter over `pieces`; returns (every emission, in order, then the
    /// flush; whether a stop hit; the matched stop)
    fn drive(stops: &[&str], pieces: &[&str]) -> (Vec<String>, bool, Option<String>) {
        let owned: Vec<String> = stops.iter().map(|s| s.to_string()).collect();
        let mut f = StopStrings::new(&owned);
        let mut out = Vec::new();
        for p in pieces {
            let e = f.push(p);
            if !e.is_empty() {
                out.push(e);
            }
            if f.hit() {
                break;
            }
        }
        let tail = f.flush();
        if !tail.is_empty() {
            out.push(tail);
        }
        (out, f.hit(), f.matched().map(|s| s.to_string()))
    }

    /// cut a string into pieces of at most `n` bytes, never inside a character
    /// (and never an EMPTY piece: when `n` is smaller than the character at `i`,
    /// the piece is that whole character — otherwise `j` walks back onto `i`,
    /// no progress is made, and the piece list grows without bound. That exact
    /// helper bug OOM-killed two test runs and, with them, the host session on
    /// 2026-09-21; this comment is the tombstone.)
    fn cut(s: &str, n: usize) -> Vec<&str> {
        let mut out = Vec::new();
        let mut i = 0usize;
        while i < s.len() {
            let mut j = (i + n).min(s.len());
            while !s.is_char_boundary(j) {
                j -= 1;
            }
            if j == i {
                let ch = s[i..].chars().next().expect("i is a char boundary");
                j = i + ch.len_utf8();
            }
            out.push(&s[i..j]);
            i = j;
        }
        out
    }

    /// the wire contract, the whole of it: whatever the piece split, the emissions
    /// concatenate to exactly the text before the earliest stop, and no emission ever
    /// CONTAINS a stop (not even partially reassembled across frames — the client
    /// concatenates frames, so the concatenation is what must stay clean)
    fn check(stops: &[&str], text: &str, n: usize) {
        let expect_hit = stops.iter().any(|s| !s.is_empty() && text.contains(s));
        let (out, hit, matched) = drive(stops, &cut(text, n));
        assert_eq!(hit, expect_hit, "stops {stops:?} text {text:?} cut {n}");
        let joined = out.concat();
        if expect_hit {
            // the earliest stop of the list, at its earliest position in the text
            let mut at = usize::MAX;
            for s in stops.iter().filter(|s| !s.is_empty()) {
                if let Some(p) = text.find(s) {
                    at = at.min(p);
                }
            }
            assert_eq!(joined, text[..at], "stops {stops:?} text {text:?} cut {n}");
            assert!(matched.is_some(), "the match is named");
            assert!(
                stops.contains(&matched.as_deref().unwrap()),
                "the named stop {matched:?} is one of the request's"
            );
            assert!(joined.len() <= at);
        } else {
            assert_eq!(joined, text, "no stop: no byte is lost, cut {n}");
        }
        // no half of a stop ever reached the wire, in either case
        for s in stops.iter().filter(|s| !s.is_empty()) {
            assert!(!joined.contains(s), "stop {s:?} leaked at cut {n}: {joined:?}");
        }
    }

    #[test]
    fn a_stop_string_ends_the_answer_and_never_reaches_the_wire() {
        let (out, hit, m) = drive(&["END"], &["the answer is ", "42.", " END and more text"]);
        assert!(hit);
        assert_eq!(m.as_deref(), Some("END"));
        assert_eq!(out.concat(), "the answer is 42. ");
        // frame for frame: the first two pieces pass whole, the third gives up the one
        // byte before the stop and nothing of the stop or after it
        assert_eq!(out, vec!["the answer is ".to_string(), "42.".to_string(), " ".to_string()]);
    }

    #[test]
    fn a_stop_string_split_across_pieces_is_held_not_emitted() {
        // "END" arrives as "E", "ND": the "E" and the "ND" are held, never emitted
        let mut f = StopStrings::new(&["END".to_string()]);
        assert_eq!(f.push("answer E"), "answer ");
        assert_eq!(f.push("N"), "");
        assert_eq!(f.push("D"), "");
        assert!(f.hit(), "the third piece completed the stop");
        assert_eq!(f.matched_at(), 7, "the match landed at content byte 7");
        assert_eq!(f.dropped(), 3, "only the stop was swallowed so far");
        assert_eq!(f.push(" and everything after"), "");
        assert_eq!(f.dropped(), 3 + " and everything after".len());
        assert_eq!(f.flush(), "", "after a hit the flush is empty");
        // and the split is irrelevant: byte-by-byte and whole give the same wire bytes
        let text = "one two END three";
        for n in [1usize, 2, 3, 5, 100] {
            let (out, hit, _) = drive(&["END"], &cut(text, n));
            assert!(hit, "cut {n}");
            assert_eq!(out.concat(), "one two ", "cut {n}");
        }
    }

    #[test]
    fn when_one_stop_contains_another_the_longest_is_named() {
        // both match at byte 2; the list is sorted longest-first, so ENDMARK is named.
        // The WIRE BYTES are the same either way - generation ends at the match.
        let (out, hit, m) = drive(&["END", "ENDMARK"], &["xx ENDMARK rest"]);
        assert!(hit);
        assert_eq!(m.as_deref(), Some("ENDMARK"));
        assert_eq!(out.concat(), "xx ");
        // the contained one alone stops at the same content bytes
        let (out2, hit2, m2) = drive(&["END"], &["xx ENDMARK rest"]);
        assert!(hit2);
        assert_eq!(m2.as_deref(), Some("END"));
        assert_eq!(out2.concat(), "xx ", "the wire bytes do not depend on the tie");
    }

    #[test]
    fn the_earliest_stop_position_wins_across_different_stops() {
        // "abc" sits at 1, "bcd" at 2: the earliest ends the answer
        let (out, hit, m) = drive(&["bcd", "abc"], &["xabc"]);
        assert!(hit);
        assert_eq!(m.as_deref(), Some("abc"));
        assert_eq!(out.concat(), "x");
        let (out, hit, m) = drive(&["bcd", "abc"], &["xbcd"]);
        assert!(hit);
        assert_eq!(m.as_deref(), Some("bcd"));
        assert_eq!(out.concat(), "x");
    }

    #[test]
    fn a_stop_string_at_byte_zero_empties_the_answer() {
        // the empty-content finish: nothing is emitted, the answer still ends `stop`
        let (out, hit, m) = drive(&["STOP"], &["STOP immediately"]);
        assert!(hit);
        assert_eq!(m.as_deref(), Some("STOP"));
        assert!(out.is_empty(), "not one byte left: {out:?}");
        // and the same when the first byte alone cannot decide yet
        let mut f = StopStrings::new(&["STOP".to_string()]);
        assert_eq!(f.push("ST"), "");
        assert!(!f.hit(), "a held prefix is not a match");
        assert_eq!(f.push("OP!"), "");
        assert!(f.hit());
        assert_eq!(f.matched_at(), 0);
    }

    #[test]
    fn an_absent_stop_list_is_a_byte_identical_passthrough() {
        // the default of every release before #86: no filter runs, no byte is held
        let mut f = StopStrings::new(&[]);
        assert!(!f.active());
        for piece in ["the answer", " is 42.", " <partial<tool_call"] {
            assert_eq!(f.push(piece), piece, "verbatim, piece for piece");
        }
        assert!(!f.hit());
        assert_eq!(f.flush(), "", "nothing was held");
        assert_eq!(f.dropped(), 0);
    }

    #[test]
    fn a_partial_prefix_that_never_completes_leaves_at_flush() {
        // a tail that could have become a stop but never did is TEXT, not a loss
        let mut f = StopStrings::new(&["</stop>".to_string()]);
        assert_eq!(f.push("done, almost </sto"), "done, almost ");
        assert_eq!(f.push("p"), ""); // still inside a possible "</stop>"
        assert!(!f.hit());
        assert_eq!(f.flush(), "</stop", "the held prefix leaves at flush");
        assert_eq!(f.dropped(), 0, "nothing was swallowed");
    }

    #[test]
    fn empty_entries_are_dropped_and_duplicates_are_harmless() {
        // an empty stop string would match at byte 0 of everything and would break
        // find_marker's hold arithmetic; parse drops it, and so does the filter
        let mut f = StopStrings::new(&[
            "".to_string(),
            "END".to_string(),
            "END".to_string(),
        ]);
        assert!(f.active());
        assert_eq!(f.push("kept END"), "kept ");
        assert!(f.hit());
    }

    #[test]
    fn non_ascii_stop_strings_hold_on_character_boundaries() {
        // the hold is byte-exact but never splits a character: the emitted prefix is
        // always a valid &str, whatever the stop's bytes
        for (stops, text) in [
            (vec!["→"], "a → b"),
            (vec!["é"], "café é régime"),
            (vec!["🦆!"], "duck 🦆! gone"),
            (vec!["end", "→"], "→ end"),
        ] {
            for n in [1usize, 2, 3, 4, 8, 100] {
                check(&stops, text, n);
            }
        }
    }

    /// the sweep: every text x every stop set x every piece size gives the wire
    /// contract of `check` — earliest stop, no leak, no loss. Nested, overlapping,
    /// repeated and adjacency-hazard shapes included.
    #[test]
    fn every_split_gives_the_earliest_stop_and_never_leaks_or_loses_a_byte() {
        let cases: Vec<(&[&str], &str)> = vec![
            (&["END"], "END"),
            (&["END"], "no end here, only End"),
            (&["END"], "E N D E N D"),
            (&["END", "EN"], "x EN y END z"),
            (&["ab", "abc", "abcd"], "abcd"),
            (&["ab", "abc", "abcd"], "zz abc abcd"),
            (&["</s>", "<s>"], "a <s> b </s> c"),
            (&["one", "two", "three"], "zero one two three"),
            (&["llama"], "llama llam ll"),
        ];
        for (stops, text) in &cases {
            for n in [1usize, 2, 3, 5, 7, 11, 1000] {
                check(stops, text, n);
            }
        }
    }
}
