//! The response budget: how much of a tool result the agent sees, and what
//! happens when a result is too large.
//!
//! Sizes are counted in characters of what reaches the agent: every text
//! block plus the serialized structured content, without the
//! `io.phirewall/release` key (the Phirewall gateway strips it before the
//! agent sees the result). The hard limit (`PULSAR_MCP_MAX_RESPONSE_BYTES`,
//! default 45,000) stays below Claude Code's 50,000-character spill line;
//! tools that can shape their own output render to [`render_budget`], 5%
//! under the 40,000-character target, so the gateway can annotate in place.
//!
//! A result over the hard limit is never replaced by an error when a
//! smaller form exists: [`degrade`] trims the largest arrays and strings,
//! says what it trimmed in `truncation`, and keeps every scalar (the ids a
//! follow-up call needs). Only when even that cannot fit does the call
//! return a short (< [`MAX_ERROR_CHARS`]) error, which names any committed
//! ids so a side effect is never hidden.

use rmcp::model::{CallToolResult, ContentBlock};
use serde_json::{json, Map, Value};

use crate::envelope::ToolError;
use crate::phirewall::KEY as SUMMARY_KEY;
use crate::release;

/// Default hard limit, in characters.
pub const DEFAULT_HARD_LIMIT: usize = 45_000;

/// Error envelopes stay under this many characters.
pub const MAX_ERROR_CHARS: usize = 5_000;

/// Below this budget no useful degraded form exists.
const MIN_DEGRADED_CHARS: usize = 1_000;

/// The target for a hard limit: 40,000 for the default 45,000.
pub fn target(hard: usize) -> usize {
    hard / 9 * 8
}

/// What a budget-aware tool renders to: 5% under the target, leaving room
/// for the gateway's in-place replacements and its `_phirewall` note.
pub fn render_budget(hard: Option<usize>) -> usize {
    target(hard.unwrap_or(DEFAULT_HARD_LIMIT)) / 20 * 19
}

fn chars(s: &str) -> usize {
    s.chars().count()
}

/// Characters of `value` serialized compactly.
pub fn json_chars(value: &Value) -> usize {
    serde_json::to_string(value).map_or(0, |s| chars(&s))
}

/// Characters of structured content the agent sees (without the release key).
pub fn structured_chars(value: &Value) -> usize {
    match value {
        Value::Object(map) if map.contains_key(release::KEY) => {
            let mut visible = map.clone();
            visible.remove(release::KEY);
            json_chars(&Value::Object(visible))
        }
        other => json_chars(other),
    }
}

fn text_chars(result: &CallToolResult) -> usize {
    result
        .content
        .iter()
        .map(|b| match b {
            ContentBlock::Text(t) => chars(&t.text),
            _ => 0,
        })
        .sum()
}

/// Agent-visible characters of a result: text plus structured content,
/// excluding [`release::KEY`].
pub fn visible_chars(result: &CallToolResult) -> usize {
    text_chars(result)
        + result
            .structured_content
            .as_ref()
            .map_or(0, structured_chars)
}

/// Agent-visible characters of a `(payload, text)` pair as `ok_result`
/// would send it.
pub fn pair_chars(payload: &Value, text: &str) -> usize {
    structured_chars(payload) + chars(text)
}

/// Cut `text` to at most `max` characters at a line boundary.
pub fn cut_text(text: &str, max: usize) -> &str {
    if chars(text) <= max {
        return text;
    }
    let end = text.char_indices().nth(max).map_or(text.len(), |(i, _)| i);
    let head = &text[..end];
    match head.rfind('\n') {
        Some(i) if i > end / 2 => &head[..i],
        _ => head,
    }
}

/// Fit a result into `hard` characters (see the module docs). `tool` names
/// the tool in the notice and in the error.
pub fn degrade(tool: &str, result: CallToolResult, hard: usize) -> CallToolResult {
    if result.is_error == Some(true) {
        return bound_error(result);
    }
    let size = visible_chars(&result);
    if size <= hard {
        return result;
    }
    let budget = render_budget(Some(hard));
    match shrink(tool, &result, size, budget) {
        Some(smaller) => smaller,
        None => too_large(tool, &result, size, hard),
    }
}

/// One trimmed location in a degraded payload.
struct Trim {
    path: String,
    shown: usize,
    total: usize,
}

fn shrink(
    tool: &str,
    result: &CallToolResult,
    size: usize,
    budget: usize,
) -> Option<CallToolResult> {
    if budget < MIN_DEGRADED_CHARS {
        return None;
    }
    let original = result.structured_content.clone();
    let text: String = result
        .content
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    // A JSON text view is the payload again: re-render it from the trimmed
    // payload instead of cutting it mid-value.
    let json_text = original
        .as_ref()
        .is_some_and(|p| serde_json::from_str::<Value>(&text).ok().as_ref() == Some(p));
    let next = format!(
        "Re-call {tool} with a narrower request (detail='summary', lower max_* or preview limits, or a filter), or use a drill-down tool for the omitted part."
    );
    let notice_room = 600;
    let mut trims: Vec<Trim> = vec![];
    let payload = match original {
        Some(mut payload) => {
            let share = if json_text {
                (budget - notice_room) / 2
            } else {
                (budget - notice_room) * 3 / 5
            };
            trim_to(&mut payload, share, &mut trims);
            if structured_chars(&payload) > share {
                return None;
            }
            Some(payload)
        }
        None => None,
    };
    let truncation = json!({
        "truncated": true,
        "by": "response_budget",
        "original_chars": size,
        "budget_chars": budget,
        "trimmed": trims.iter().map(|t| json!({"path": t.path, "shown": t.shown, "total": t.total})).collect::<Vec<_>>(),
        "next": next,
    });
    let payload = payload.map(|mut p| {
        if let Value::Object(map) = &mut p {
            map.insert("truncation".into(), truncation.clone());
        }
        p
    });
    let struct_size = payload.as_ref().map_or(0, structured_chars);
    let notice = format!(
        "[truncated: this result was {size} characters, over the {budget}-character budget; {} part(s) trimmed. {next}]",
        trims.len().max(1)
    );
    let room = budget.checked_sub(struct_size + chars(&notice) + 2)?;
    let body = if json_text {
        let compact = serde_json::to_string(payload.as_ref()?).ok()?;
        if chars(&compact) > room {
            return None;
        }
        compact
    } else {
        cut_text(&text, room).to_owned()
    };
    let text = format!("{notice}\n\n{body}");
    let mut out = CallToolResult::success(vec![ContentBlock::text(text)]);
    out.structured_content = payload;
    Some(out)
}

/// Trim `value` (largest arrays and strings first) until it fits `max`
/// characters, as far as trimming can; returns how many places it cut.
pub(crate) fn trim_value(value: &mut Value, max: usize) -> usize {
    let mut trims = vec![];
    trim_to(value, max, &mut trims);
    trims.len()
}

/// Halve the largest array (or cut the longest string) until `payload`
/// fits `max`. Scalars and object keys are kept, so required fields and ids
/// survive. The gateway's keys are left alone.
fn trim_to(payload: &mut Value, max: usize, trims: &mut Vec<Trim>) {
    for _ in 0..200 {
        if structured_chars(payload) <= max {
            return;
        }
        let Some((path, len)) = largest(payload, "") else {
            // The gateway's summary is a validated contract: dropped whole,
            // never cut, and only when nothing else is left to trim.
            match payload.as_object_mut().map(|m| m.remove(SUMMARY_KEY)) {
                Some(Some(_)) => continue,
                _ => return,
            }
        };
        let Some(target) = payload.pointer_mut(&path) else {
            return;
        };
        let total = match target {
            Value::Array(items) => {
                let keep = len / 2;
                items.truncate(keep);
                len
            }
            Value::String(s) => {
                let keep = chars(s) / 2;
                let cut: String = s.chars().take(keep).collect();
                *s = format!("{cut}…");
                len
            }
            _ => return,
        };
        match trims.iter_mut().find(|t| t.path == path) {
            Some(t) => t.shown = shown(payload.pointer(&path)),
            None => trims.push(Trim {
                shown: shown(payload.pointer(&path)),
                path,
                total,
            }),
        }
    }
}

fn shown(v: Option<&Value>) -> usize {
    match v {
        Some(Value::Array(a)) => a.len(),
        Some(Value::String(s)) => chars(s),
        _ => 0,
    }
}

/// The JSON pointer and length of the trimmable value (a non-empty array
/// or a string over 200 characters) with the largest serialized size.
fn largest(v: &Value, path: &str) -> Option<(String, usize)> {
    let mut best: Option<(usize, String, usize)> = None;
    let mut consider = |size: usize, p: String, len: usize| {
        if best.as_ref().is_none_or(|b| size > b.0) {
            best = Some((size, p, len));
        }
    };
    fn walk(v: &Value, path: &str, f: &mut dyn FnMut(usize, String, usize)) {
        match v {
            Value::Array(items) => {
                if !items.is_empty() {
                    f(json_chars(v), path.to_owned(), items.len());
                }
                for (i, item) in items.iter().enumerate() {
                    walk(item, &format!("{path}/{i}"), f);
                }
            }
            Value::String(s) if chars(s) > 200 => f(chars(s), path.to_owned(), chars(s)),
            Value::Object(map) => {
                for (k, child) in map {
                    if path.is_empty()
                        && (k == release::KEY || k == SUMMARY_KEY || k == "truncation")
                    {
                        continue;
                    }
                    let key = k.replace('~', "~0").replace('/', "~1");
                    walk(child, &format!("{path}/{key}"), f);
                }
            }
            _ => {}
        }
    }
    walk(v, path, &mut consider);
    best.map(|(_, p, len)| (p, len))
}

/// Top-level ids of a payload: what a call committed and what a follow-up
/// call needs.
fn committed_ids(payload: Option<&Value>) -> Map<String, Value> {
    payload
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
        .filter(|(k, v)| (k.ends_with("_id") || k.as_str() == "output_path") && v.is_string())
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

/// `RESPONSE_TOO_LARGE`, only when no degraded form fits. The call itself
/// succeeded, so the error names what it committed.
fn too_large(tool: &str, result: &CallToolResult, size: usize, hard: usize) -> CallToolResult {
    let tool: &'static str = Box::leak(tool.to_owned().into_boxed_str());
    let ids = committed_ids(result.structured_content.as_ref());
    let committed = if ids.is_empty() {
        String::new()
    } else {
        let list: Vec<String> = ids.iter().map(|(k, v)| format!("{k}={v}")).collect();
        format!(
            " The call itself succeeded and its effects stand ({}); do not repeat it to recover them.",
            list.join(", ")
        )
    };
    let live = if ids.contains_key("cluster_assignment_id") {
        " The session's live cluster labels now use that cluster_assignment_id."
    } else {
        ""
    };
    let mut error = ToolError::new(
        tool,
        "RESPONSE_TOO_LARGE",
        format!("The result is {size} characters; even a trimmed form does not fit the {hard}-character limit.{committed}{live}"),
        format!("Call {tool} with a narrower request (detail='summary', lower max_* or preview limits), or raise PULSAR_MCP_MAX_RESPONSE_BYTES."),
    )
    .detail("size_chars", json!(size))
    .detail("limit_chars", json!(hard));
    if !ids.is_empty() {
        error = error.detail("committed", Value::Object(ids));
    }
    bound_error(error.into_result())
}

/// Keep an error envelope under [`MAX_ERROR_CHARS`]: trim its details
/// first, then its reason. The code and the action always stay.
pub fn bound_error(mut result: CallToolResult) -> CallToolResult {
    if visible_chars(&result) <= MAX_ERROR_CHARS {
        return result;
    }
    let Some(mut envelope) = result.structured_content.take() else {
        return result;
    };
    // Text and structured content carry the envelope twice.
    let half = MAX_ERROR_CHARS / 2 - 50;
    if let Some(details) = envelope.get_mut("details") {
        let mut trims = vec![];
        trim_to(details, half / 3, &mut trims);
        if json_chars(details) > half / 3 {
            *details = json!({"trimmed": true});
        }
    }
    for key in ["metrics", "reason", "agent_action"] {
        if json_chars(&envelope) <= half {
            break;
        }
        match envelope.get_mut(key) {
            Some(Value::String(s)) => *s = cut_text(s, 800).to_owned() + "…",
            Some(v @ Value::Object(_)) => *v = json!({}),
            _ => {}
        }
    }
    let text = serde_json::to_string(&envelope).unwrap_or_default();
    let mut out = CallToolResult::error(vec![ContentBlock::text(text)]);
    out.structured_content = Some(envelope);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envelope::ok_result;

    #[test]
    fn the_release_block_is_not_counted() {
        let small = ok_result(json!({"a": 1}), Some("x".into()));
        let with = ok_result(
            json!({"a": 1, release::KEY: {"rows": "x".repeat(10_000)}}),
            Some("x".into()),
        );
        assert_eq!(visible_chars(&small), visible_chars(&with));
    }

    #[test]
    fn the_default_budget_is_under_the_spill_line() {
        assert_eq!(target(DEFAULT_HARD_LIMIT), 40_000);
        assert_eq!(render_budget(None), 38_000);
        const { assert!(DEFAULT_HARD_LIMIT < 50_000) };
    }

    #[test]
    fn an_oversized_result_is_trimmed_not_failed_and_keeps_its_ids() {
        let rows: Vec<Value> = (0..5_000).map(|i| json!({"row": i, "v": "abc"})).collect();
        let payload = json!({"status": "ok", "run_id": "run_1", "rows": rows});
        let result = ok_result(payload, None);
        let out = degrade("t", result, 45_000);
        assert_eq!(out.is_error, Some(false));
        assert!(visible_chars(&out) <= render_budget(Some(45_000)));
        let s = out.structured_content.unwrap();
        assert_eq!(s["run_id"], "run_1");
        assert_eq!(s["truncation"]["truncated"], true);
        assert_eq!(s["truncation"]["trimmed"][0]["path"], "/rows");
        assert_eq!(s["truncation"]["trimmed"][0]["total"], 5_000);
        let text = match &out.content[0] {
            ContentBlock::Text(t) => t.text.clone(),
            _ => unreachable!(),
        };
        assert!(text.starts_with("[truncated:"), "{}", &text[..80]);
    }

    #[test]
    fn the_gateway_summary_is_never_cut() {
        let rows: Vec<Value> = (0..5_000).map(|i| json!({"row": i})).collect();
        let summary = json!({"version": 1, "grouping": {"groups": vec![json!({"n": 1}); 50]}});
        let payload = json!({"rows": rows, SUMMARY_KEY: summary.clone()});
        let out = degrade("t", ok_result(payload, None), 45_000);
        assert_eq!(out.structured_content.unwrap()[SUMMARY_KEY], summary);
    }

    #[test]
    fn markdown_is_cut_at_a_line_with_a_notice() {
        let text = (0..10_000)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let out = degrade("t", ok_result(json!({"plan_id": "p"}), Some(text)), 20_000);
        assert_eq!(out.is_error, Some(false));
        assert!(visible_chars(&out) <= render_budget(Some(20_000)));
    }

    #[test]
    fn with_no_room_the_error_is_short_and_names_what_was_committed() {
        let payload = json!({"cluster_assignment_id": "ca_1", "rows": vec![1; 1000]});
        let out = degrade("generate_cluster_dossier", ok_result(payload, None), 200);
        assert_eq!(out.is_error, Some(true));
        assert!(visible_chars(&out) < MAX_ERROR_CHARS);
        let s = out.structured_content.unwrap();
        assert_eq!(s["error_code"], "RESPONSE_TOO_LARGE");
        assert_eq!(s["details"]["committed"]["cluster_assignment_id"], "ca_1");
        assert!(s["reason"]
            .as_str()
            .unwrap()
            .contains("live cluster labels"));
    }

    #[test]
    fn a_large_error_is_bounded() {
        let error = ToolError::new("t", "X", "r".repeat(20_000), "do this")
            .detail("diag", json!(vec!["d".repeat(50); 400]));
        let out = degrade("t", error.into_result(), 45_000);
        assert!(
            visible_chars(&out) < MAX_ERROR_CHARS,
            "{}",
            visible_chars(&out)
        );
        let s = out.structured_content.unwrap();
        assert_eq!(s["error_code"], "X");
        assert_eq!(s["agent_action"], "do this");
    }
}
