//! Split-manifest reading and append-only result splicing for the benchmark
//! harness `--split` mode.
//!
//! A split manifest is the JSON that `examples/make-split.rs` writes: a frozen
//! scenario-stratified train/test split plus a growing `results` array. This
//! module reads the fields the harness needs to run a set (the pool location
//! and the per-scenario stem lists) and appends one result entry to the
//! `results` array WITHOUT reserializing the rest of the file, so every frozen
//! field stays byte-identical across the append. The splice is textual: it
//! locates the `results` array by structure and rewrites only its interior.
//!
//! The reader is a small self-contained JSON parser (no serde dependency in
//! the kit), sufficient for the manifest schema the tool produces.

/// The parsed manifest fields the harness needs to run a set.
pub struct Manifest {
    pub name: String,
    pub pool: String,
    train: Vec<(String, Vec<String>)>,
    test: Vec<(String, Vec<String>)>,
}

impl Manifest {
    /// The stems named for `scenario` in the given set (`"train"` or
    /// `"test"`), empty when the manifest lists none.
    pub fn stems(&self, set: &str, scenario: &str) -> &[String] {
        let map = if set == "test" {
            &self.test
        } else {
            &self.train
        };
        map.iter()
            .find(|(k, _)| k == scenario)
            .map(|(_, v)| v.as_slice())
            .unwrap_or(&[])
    }
}

/// Parses a manifest, extracting the pool, the name, and the train/test stem
/// lists per scenario.
pub fn parse(text: &str) -> Result<Manifest, String> {
    let value = json::parse(text)?;
    let obj = value
        .as_object()
        .ok_or("manifest root is not a JSON object")?;
    let name = obj_str(obj, "name")?;
    let pool = obj_str(obj, "pool")?;
    let train = obj_set(obj, "train")?;
    let test = obj_set(obj, "test")?;
    Ok(Manifest {
        name,
        pool,
        train,
        test,
    })
}

fn obj_str(obj: &[(String, json::Json)], key: &str) -> Result<String, String> {
    obj.iter()
        .find(|(k, _)| k == key)
        .and_then(|(_, v)| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| format!("manifest is missing a string \"{key}\""))
}

/// Reads a `train`/`test` object into (scenario, stems) pairs.
fn obj_set(obj: &[(String, json::Json)], key: &str) -> Result<Vec<(String, Vec<String>)>, String> {
    let set = obj
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v)
        .ok_or_else(|| format!("manifest is missing the \"{key}\" object"))?
        .as_object()
        .ok_or_else(|| format!("manifest \"{key}\" is not an object"))?;
    let mut out = Vec::new();
    for (scenario, list) in set {
        let arr = list
            .as_array()
            .ok_or_else(|| format!("manifest \"{key}\".\"{scenario}\" is not an array"))?;
        let mut stems = Vec::with_capacity(arr.len());
        for item in arr {
            let stem = item
                .as_str()
                .ok_or_else(|| format!("manifest \"{key}\".\"{scenario}\" holds a non-string"))?;
            stems.push(stem.to_string());
        }
        out.push((scenario.clone(), stems));
    }
    Ok(out)
}

/// Appends `entry_json` (one rendered result object, a single line) to the
/// manifest's `results` array and returns the updated manifest text. Only the
/// interior of the `results` array is rewritten; every other byte, including
/// all frozen split fields, is preserved exactly.
pub fn append_result(text: &str, entry_json: &str) -> Result<String, String> {
    let (open, close) = results_span(text)?;
    let bytes = text.as_bytes();
    if bytes.get(open) != Some(&b'[') || bytes.get(close - 1) != Some(&b']') {
        return Err("manifest \"results\" is not an array".to_string());
    }
    let head = &text[..open];
    let tail = &text[close..];
    let inner = &text[open + 1..close - 1];
    let new_inner = if inner.trim().is_empty() {
        format!("\n    {entry_json}\n  ")
    } else {
        format!("{},\n    {entry_json}\n  ", inner.trim_end())
    };
    Ok(format!("{head}[{new_inner}]{tail}"))
}

/// Byte span `[start, end)` of the top-level `results` member's value, found
/// structurally so string contents and nesting cannot confuse it.
fn results_span(text: &str) -> Result<(usize, usize), String> {
    let b = text.as_bytes();
    let mut i = skip_ws(b, 0);
    if b.get(i) != Some(&b'{') {
        return Err("manifest root is not an object".to_string());
    }
    i += 1;
    loop {
        i = skip_ws(b, i);
        match b.get(i) {
            Some(b'}') | None => return Err("manifest has no \"results\" member".to_string()),
            Some(b'"') => {}
            _ => return Err("expected a member name in manifest".to_string()),
        }
        let key_start = i;
        let key_end = skip_string(b, i)?;
        let key = &text[key_start + 1..key_end - 1];
        i = skip_ws(b, key_end);
        if b.get(i) != Some(&b':') {
            return Err("expected ':' after a manifest member name".to_string());
        }
        i = skip_ws(b, i + 1);
        let value_start = i;
        let value_end = skip_value(b, i)?;
        if key == "results" {
            return Ok((value_start, value_end));
        }
        i = skip_ws(b, value_end);
        match b.get(i) {
            Some(b',') => i += 1,
            _ => return Err("manifest has no \"results\" member".to_string()),
        }
    }
}

fn skip_ws(b: &[u8], mut i: usize) -> usize {
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') {
        i += 1;
    }
    i
}

/// Advances past a string starting at `b[i] == '"'`, returning the index just
/// past the closing quote. UTF-8 continuation bytes never equal a structural
/// ASCII byte, so byte scanning is safe.
fn skip_string(b: &[u8], i: usize) -> Result<usize, String> {
    debug_assert_eq!(b.get(i), Some(&b'"'));
    let mut i = i + 1;
    while i < b.len() {
        match b[i] {
            b'\\' => i += 2,
            b'"' => return Ok(i + 1),
            _ => i += 1,
        }
    }
    Err("unterminated string in manifest".to_string())
}

/// Advances past one JSON value, returning its exclusive end index.
fn skip_value(b: &[u8], i: usize) -> Result<usize, String> {
    let i = skip_ws(b, i);
    match b.get(i) {
        Some(b'"') => skip_string(b, i),
        Some(b'{') | Some(b'[') => skip_container(b, i),
        Some(_) => {
            let mut j = i;
            while j < b.len() && !matches!(b[j], b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r')
            {
                j += 1;
            }
            Ok(j)
        }
        None => Err("unexpected end of manifest".to_string()),
    }
}

/// Advances past a `{...}` or `[...]` container, returning the index just past
/// the matching close. Strings are skipped so brackets inside them do not
/// count; nested containers of the same bracket type are matched by depth, and
/// the other bracket type is always balanced within.
fn skip_container(b: &[u8], i: usize) -> Result<usize, String> {
    let open = b[i];
    let close = if open == b'{' { b'}' } else { b']' };
    let mut depth = 0i32;
    let mut i = i;
    while i < b.len() {
        match b[i] {
            b'"' => {
                i = skip_string(b, i)?;
                continue;
            }
            c if c == open => depth += 1,
            c if c == close => {
                depth -= 1;
                if depth == 0 {
                    return Ok(i + 1);
                }
            }
            _ => {}
        }
        i += 1;
    }
    Err("unbalanced container in manifest".to_string())
}

// ---------------------------------------------------------------------------
// A minimal JSON value parser, enough for the manifest schema.
// ---------------------------------------------------------------------------

mod json {
    /// A parsed JSON value. The manifest reader only consumes strings,
    /// arrays, and objects; the scalar payloads are parsed for completeness
    /// (the manifest holds numbers and nulls) but never read, so the unread
    /// fields are allowed.
    #[allow(dead_code)]
    pub enum Json {
        Null,
        Bool(bool),
        Num(f64),
        Str(String),
        Arr(Vec<Json>),
        Obj(Vec<(String, Json)>),
    }

    impl Json {
        pub fn as_str(&self) -> Option<&str> {
            match self {
                Json::Str(s) => Some(s),
                _ => None,
            }
        }
        pub fn as_array(&self) -> Option<&[Json]> {
            match self {
                Json::Arr(v) => Some(v),
                _ => None,
            }
        }
        pub fn as_object(&self) -> Option<&[(String, Json)]> {
            match self {
                Json::Obj(v) => Some(v),
                _ => None,
            }
        }
    }

    pub fn parse(text: &str) -> Result<Json, String> {
        let bytes = text.as_bytes();
        let mut p = Parser { b: bytes, i: 0 };
        p.skip_ws();
        let value = p.value()?;
        p.skip_ws();
        if p.i != bytes.len() {
            return Err("trailing content after JSON value".to_string());
        }
        Ok(value)
    }

    struct Parser<'a> {
        b: &'a [u8],
        i: usize,
    }

    impl Parser<'_> {
        fn skip_ws(&mut self) {
            while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
                self.i += 1;
            }
        }

        fn value(&mut self) -> Result<Json, String> {
            self.skip_ws();
            match self.b.get(self.i) {
                Some(b'{') => self.object(),
                Some(b'[') => self.array(),
                Some(b'"') => Ok(Json::Str(self.string()?)),
                Some(b't') => self.literal("true", Json::Bool(true)),
                Some(b'f') => self.literal("false", Json::Bool(false)),
                Some(b'n') => self.literal("null", Json::Null),
                Some(_) => self.number(),
                None => Err("unexpected end of JSON".to_string()),
            }
        }

        fn literal(&mut self, word: &str, value: Json) -> Result<Json, String> {
            if self.b[self.i..].starts_with(word.as_bytes()) {
                self.i += word.len();
                Ok(value)
            } else {
                Err(format!("expected '{word}' in JSON"))
            }
        }

        fn number(&mut self) -> Result<Json, String> {
            let start = self.i;
            while self.i < self.b.len()
                && matches!(
                    self.b[self.i],
                    b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E'
                )
            {
                self.i += 1;
            }
            let raw = std::str::from_utf8(&self.b[start..self.i])
                .map_err(|_| "invalid number in JSON".to_string())?;
            raw.parse::<f64>()
                .map(Json::Num)
                .map_err(|_| format!("invalid number '{raw}' in JSON"))
        }

        fn string(&mut self) -> Result<String, String> {
            debug_assert_eq!(self.b.get(self.i), Some(&b'"'));
            self.i += 1;
            let mut out = String::new();
            while self.i < self.b.len() {
                let c = self.b[self.i];
                match c {
                    b'"' => {
                        self.i += 1;
                        return Ok(out);
                    }
                    b'\\' => {
                        self.i += 1;
                        let esc = *self.b.get(self.i).ok_or("truncated escape in JSON")?;
                        match esc {
                            b'"' => out.push('"'),
                            b'\\' => out.push('\\'),
                            b'/' => out.push('/'),
                            b'n' => out.push('\n'),
                            b'r' => out.push('\r'),
                            b't' => out.push('\t'),
                            b'b' => out.push('\u{0008}'),
                            b'f' => out.push('\u{000C}'),
                            b'u' => {
                                let hex = self
                                    .b
                                    .get(self.i + 1..self.i + 5)
                                    .ok_or("truncated \\u escape in JSON")?;
                                let code = u32::from_str_radix(
                                    std::str::from_utf8(hex)
                                        .map_err(|_| "invalid \\u escape in JSON".to_string())?,
                                    16,
                                )
                                .map_err(|_| "invalid \\u escape in JSON".to_string())?;
                                out.push(char::from_u32(code).unwrap_or('\u{FFFD}'));
                                self.i += 4;
                            }
                            other => {
                                return Err(format!("unknown escape '\\{}' in JSON", other as char))
                            }
                        }
                        self.i += 1;
                    }
                    _ => {
                        // Copy one UTF-8 code point verbatim.
                        let rest = &self.b[self.i..];
                        let width = utf8_width(rest[0]);
                        let slice = rest.get(..width).ok_or("truncated UTF-8 in JSON string")?;
                        out.push_str(
                            std::str::from_utf8(slice)
                                .map_err(|_| "invalid UTF-8 in JSON string".to_string())?,
                        );
                        self.i += width;
                    }
                }
            }
            Err("unterminated string in JSON".to_string())
        }

        fn array(&mut self) -> Result<Json, String> {
            self.i += 1; // '['
            let mut out = Vec::new();
            self.skip_ws();
            if self.b.get(self.i) == Some(&b']') {
                self.i += 1;
                return Ok(Json::Arr(out));
            }
            loop {
                out.push(self.value()?);
                self.skip_ws();
                match self.b.get(self.i) {
                    Some(b',') => {
                        self.i += 1;
                        self.skip_ws();
                    }
                    Some(b']') => {
                        self.i += 1;
                        return Ok(Json::Arr(out));
                    }
                    _ => return Err("expected ',' or ']' in JSON array".to_string()),
                }
            }
        }

        fn object(&mut self) -> Result<Json, String> {
            self.i += 1; // '{'
            let mut out = Vec::new();
            self.skip_ws();
            if self.b.get(self.i) == Some(&b'}') {
                self.i += 1;
                return Ok(Json::Obj(out));
            }
            loop {
                self.skip_ws();
                if self.b.get(self.i) != Some(&b'"') {
                    return Err("expected a string key in JSON object".to_string());
                }
                let key = self.string()?;
                self.skip_ws();
                if self.b.get(self.i) != Some(&b':') {
                    return Err("expected ':' in JSON object".to_string());
                }
                self.i += 1;
                let value = self.value()?;
                out.push((key, value));
                self.skip_ws();
                match self.b.get(self.i) {
                    Some(b',') => {
                        self.i += 1;
                    }
                    Some(b'}') => {
                        self.i += 1;
                        return Ok(Json::Obj(out));
                    }
                    _ => return Err("expected ',' or '}' in JSON object".to_string()),
                }
            }
        }
    }

    /// Byte length of the UTF-8 code point a lead byte begins.
    fn utf8_width(lead: u8) -> usize {
        if lead < 0x80 {
            1
        } else if lead >> 5 == 0b110 {
            2
        } else if lead >> 4 == 0b1110 {
            3
        } else if lead >> 3 == 0b11110 {
            4
        } else {
            1
        }
    }
}
