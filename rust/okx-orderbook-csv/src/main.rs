use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Result};
use chrono::{DateTime, TimeZone, Utc};
use clap::{ArgAction, Args, Parser, Subcommand};
use ordered_float::OrderedFloat;
use serde::Deserialize;
use serde_json::Value;

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Convert(args) => run_convert(args),
        Command::Rebuild(args) => run_rebuild(args),
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "okx-orderbook-csv",
    version,
    about = "Utilities for OKX Level 2 orderbook dumps"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Convert newline-delimited JSON dumps (.data) into a flat CSV
    Convert(ConvertArgs),
    /// Rebuild snapshots at a fixed frequency directly from the CSV rows
    Rebuild(RebuildArgs),
}

#[derive(Args, Debug, Clone)]
struct ConvertArgs {
    /// Input .data file path
    #[arg(short = 'i', long = "input")]
    input: PathBuf,

    /// Output .csv file path (defaults to input with .csv extension)
    #[arg(short = 'o', long = "output")]
    output: Option<PathBuf>,

    /// Max number of lines (NDJSON rows) to read from input
    #[arg(long = "nrows")]
    nrows: Option<usize>,

    /// Do not append a human-readable UTC timestamp column
    #[arg(long = "no-timestamp", action = ArgAction::SetTrue)]
    no_timestamp: bool,
}

#[derive(Args, Debug, Clone)]
struct RebuildArgs {
    /// Input CSV produced by `convert`
    #[arg(short = 'i', long = "input")]
    input: PathBuf,

    /// Output CSV path (defaults to `<input>.snapshots.csv`)
    #[arg(short = 'o', long = "output")]
    output: Option<PathBuf>,

    /// Snapshot frequency in milliseconds
    #[arg(long = "freq-ms", default_value_t = 100)]
    freq_ms: u64,

    /// Limit number of price levels per side
    #[arg(long = "depth")]
    depth: Option<usize>,

    /// Only process the first N minutes from the earliest timestamp
    #[arg(long = "max-duration-minutes")]
    max_duration_minutes: Option<f64>,

    /// Only process the first N milliseconds from the earliest timestamp
    #[arg(long = "max-duration-ms")]
    max_duration_ms: Option<u64>,
}

impl RebuildArgs {
    fn duration_limit_ms(&self) -> Option<i64> {
        if let Some(ms) = self.max_duration_ms {
            if ms > 0 {
                return Some(ms as i64);
            }
        }
        self.max_duration_minutes.and_then(|mins| {
            if mins > 0.0 {
                Some((mins * 60_000.0) as i64)
            } else {
                None
            }
        })
    }
}

fn run_convert(args: ConvertArgs) -> Result<()> {
    if !args.input.exists() {
        bail!("Input file not found: {}", args.input.display());
    }
    let output = args
        .output
        .unwrap_or_else(|| derive_output_path(&args.input));

    let infile = File::open(&args.input)?;
    let reader = BufReader::new(infile);

    let mut wtr = csv::Writer::from_path(&output)?;

    let write_timestamp = !args.no_timestamp;
    if write_timestamp {
        wtr.write_record([
            "instrument",
            "action",
            "side",
            "price",
            "size",
            "count",
            "ts",
            "timestamp",
        ])?;
    } else {
        wtr.write_record([
            "instrument",
            "action",
            "side",
            "price",
            "size",
            "count",
            "ts",
        ])?;
    }

    let mut line_no: usize = 0;
    for line_res in reader.lines() {
        line_no += 1;
        if let Some(n) = args.nrows {
            if line_no > n {
                break;
            }
        }

        let line = match line_res {
            Ok(s) => s.trim().to_string(),
            Err(e) => {
                eprintln!("Failed to read line {}: {}", line_no, e);
                continue;
            }
        };
        if line.is_empty() {
            continue;
        }

        let v: Value = serde_json::from_str(&line)
            .map_err(|e| anyhow!("Failed to parse JSON on line {}: {}", line_no, e))?;

        let ts_opt = v.get("ts").and_then(parse_i64);
        let inst = v
            .get("instId")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string());
        let action = v
            .get("action")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string());

        let mut wrote_any = false;
        for (side_key, side_label) in [("bids", "bid"), ("asks", "ask")] {
            if let Some(levels) = v.get(side_key).and_then(|lvl| lvl.as_array()) {
                for level in levels {
                    if let Some(list) = level.as_array() {
                        if list.len() < 2 {
                            continue;
                        }
                        let price = parse_f64(&list[0]);
                        let size = parse_f64(&list[1]);
                        let count = if list.len() > 2 {
                            parse_i64(&list[2])
                        } else {
                            None
                        };

                        let ts_str = ts_opt.map(|x| x.to_string()).unwrap_or_default();
                        if write_timestamp {
                            let timestamp = ts_opt.map(ms_to_rfc3339_utc).unwrap_or_default();
                            wtr.write_record(&[
                                inst.as_deref().unwrap_or(""),
                                action.as_deref().unwrap_or(""),
                                side_label,
                                &price.map(|x| x.to_string()).unwrap_or_default(),
                                &size.map(|x| x.to_string()).unwrap_or_default(),
                                &count.map(|x| x.to_string()).unwrap_or_default(),
                                &ts_str,
                                &timestamp,
                            ])?;
                        } else {
                            wtr.write_record(&[
                                inst.as_deref().unwrap_or(""),
                                action.as_deref().unwrap_or(""),
                                side_label,
                                &price.map(|x| x.to_string()).unwrap_or_default(),
                                &size.map(|x| x.to_string()).unwrap_or_default(),
                                &count.map(|x| x.to_string()).unwrap_or_default(),
                                &ts_str,
                            ])?;
                        }
                        wrote_any = true;
                    }
                }
            }
        }

        if !wrote_any {
            let ts_str = ts_opt.map(|x| x.to_string()).unwrap_or_default();
            if write_timestamp {
                let timestamp = ts_opt.map(ms_to_rfc3339_utc).unwrap_or_default();
                wtr.write_record(&[
                    inst.as_deref().unwrap_or(""),
                    action.as_deref().unwrap_or(""),
                    "",
                    "",
                    "",
                    "",
                    &ts_str,
                    &timestamp,
                ])?;
            } else {
                wtr.write_record(&[
                    inst.as_deref().unwrap_or(""),
                    action.as_deref().unwrap_or(""),
                    "",
                    "",
                    "",
                    "",
                    &ts_str,
                ])?;
            }
        }
    }

    wtr.flush()?;
    eprintln!("Wrote CSV: {}", output.display());
    Ok(())
}

fn run_rebuild(args: RebuildArgs) -> Result<()> {
    let duration_limit = args.duration_limit_ms();
    let RebuildArgs {
        input,
        output,
        freq_ms,
        depth,
        max_duration_minutes: _,
        max_duration_ms: _,
    } = args;

    if !input.exists() {
        bail!("Input file not found: {}", input.display());
    }
    let output = output.unwrap_or_else(|| derive_snapshot_output_path(&input));

    let mut reader = csv::Reader::from_path(&input)?;
    let mut writer = csv::Writer::from_path(&output)?;
    writer.write_record([
        "instrument",
        "action",
        "side",
        "price",
        "size",
        "count",
        "ts",
        "timestamp",
    ])?;

    let freq_ms = freq_ms.max(1);

    let mut absolute_limit_ts: Option<i64> = None;
    let mut earliest_ts: Option<i64> = None;
    let mut states: HashMap<String, InstrumentState> = HashMap::new();

    for row in reader.deserialize::<CsvSourceRow>() {
        let raw = row?;
        let ts = match raw.ts {
            Some(value) => value,
            None => continue,
        };
        let instrument = raw.instrument.unwrap_or_default();
        let side = raw.side.as_deref().and_then(Side::from_str);
        let event = InstrumentEvent {
            ts,
            action: raw.action,
            side,
            price: raw.price,
            size: raw.size,
        };

        if earliest_ts.is_none() {
            earliest_ts = Some(ts);
            if let Some(limit) = duration_limit {
                absolute_limit_ts = Some(ts + limit);
            }
        }

        if let Some(limit_ts) = absolute_limit_ts {
            if ts > limit_ts {
                break;
            }
        }

        let state = states
            .entry(instrument.clone())
            .or_insert_with(InstrumentState::new);
        state.push_event(
            &instrument,
            event,
            freq_ms,
            depth,
            absolute_limit_ts,
            &mut writer,
        )?;
    }

    for (instrument, state) in states.iter_mut() {
        state.flush_pending(instrument, freq_ms, depth, absolute_limit_ts, &mut writer)?;
    }

    writer.flush()?;
    eprintln!("Rebuilt snapshots CSV: {}", output.display());
    Ok(())
}

#[derive(Debug, Deserialize)]
struct CsvSourceRow {
    instrument: Option<String>,
    action: Option<String>,
    side: Option<String>,
    price: Option<f64>,
    size: Option<f64>,
    ts: Option<i64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    Bid,
    Ask,
}

impl Side {
    fn from_str(input: &str) -> Option<Self> {
        match input.to_ascii_lowercase().as_str() {
            "bid" => Some(Self::Bid),
            "ask" => Some(Self::Ask),
            _ => None,
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Side::Bid => "bid",
            Side::Ask => "ask",
        }
    }
}

struct InstrumentEvent {
    ts: i64,
    action: Option<String>,
    side: Option<Side>,
    price: Option<f64>,
    size: Option<f64>,
}

#[derive(Default)]
struct InstrumentState {
    bids: BTreeMap<OrderedFloat<f64>, f64>,
    asks: BTreeMap<OrderedFloat<f64>, f64>,
    have_snapshot: bool,
    next_emit_ts: Option<i64>,
    pending_ts: Option<i64>,
    pending_action: Option<String>,
    pending_rows: Vec<PendingRow>,
}

impl InstrumentState {
    fn new() -> Self {
        Self::default()
    }

    fn push_event(
        &mut self,
        instrument: &str,
        event: InstrumentEvent,
        freq_ms: u64,
        depth: Option<usize>,
        limit_ts: Option<i64>,
        writer: &mut csv::Writer<File>,
    ) -> Result<()> {
        if let Some(current_ts) = self.pending_ts {
            if event.ts != current_ts {
                self.consume_pending(instrument, freq_ms, depth, limit_ts, writer)?;
            }
        }

        if self.pending_ts.is_none() {
            self.pending_ts = Some(event.ts);
        }
        if self.pending_action.is_none() {
            self.pending_action = event.action.clone();
        }
        self.pending_rows.push(PendingRow {
            side: event.side,
            price: event.price,
            size: event.size,
        });

        // Maintain most recent action label in case earlier rows were empty
        if let Some(action) = event.action {
            self.pending_action = Some(action);
        }

        Ok(())
    }

    fn flush_pending(
        &mut self,
        instrument: &str,
        freq_ms: u64,
        depth: Option<usize>,
        limit_ts: Option<i64>,
        writer: &mut csv::Writer<File>,
    ) -> Result<()> {
        if self.pending_ts.is_none() || self.pending_rows.is_empty() {
            return Ok(());
        }
        self.consume_pending(instrument, freq_ms, depth, limit_ts, writer)
    }

    fn consume_pending(
        &mut self,
        instrument: &str,
        freq_ms: u64,
        depth: Option<usize>,
        limit_ts: Option<i64>,
        writer: &mut csv::Writer<File>,
    ) -> Result<()> {
        if self.pending_ts.is_none() {
            return Ok(());
        }
        let ts = self.pending_ts.take().unwrap();
        let action = self.pending_action.take();
        let rows = std::mem::take(&mut self.pending_rows);
        self.apply_chunk(
            instrument, ts, action, rows, freq_ms, depth, limit_ts, writer,
        )?;
        Ok(())
    }

    fn apply_chunk(
        &mut self,
        instrument: &str,
        chunk_ts: i64,
        action: Option<String>,
        rows: Vec<PendingRow>,
        freq_ms: u64,
        depth: Option<usize>,
        limit_ts: Option<i64>,
        writer: &mut csv::Writer<File>,
    ) -> Result<()> {
        if let Some(action_label) = action.as_deref() {
            match action_label {
                "snapshot" => {
                    self.bids.clear();
                    self.asks.clear();
                    self.have_snapshot = true;
                    for row in &rows {
                        let (Some(side), Some(price), Some(size)) = (row.side, row.price, row.size)
                        else {
                            continue;
                        };
                        if !price.is_finite() || !size.is_finite() {
                            continue;
                        }
                        let key = OrderedFloat(price);
                        match side {
                            Side::Bid => {
                                self.bids.insert(key, size);
                            }
                            Side::Ask => {
                                self.asks.insert(key, size);
                            }
                        }
                    }
                    self.next_emit_ts = Some(match self.next_emit_ts {
                        Some(existing) => existing.max(chunk_ts),
                        None => chunk_ts,
                    });
                }
                "update" => {
                    if !self.have_snapshot {
                        return Ok(());
                    }
                    for row in &rows {
                        let Some(side) = row.side else {
                            continue;
                        };
                        let Some(price) = row.price else {
                            continue;
                        };
                        let key = OrderedFloat(price);
                        match (side, row.size) {
                            (Side::Bid, Some(size)) if size > 0.0 && size.is_finite() => {
                                self.bids.insert(key, size);
                            }
                            (Side::Bid, _) => {
                                self.bids.remove(&key);
                            }
                            (Side::Ask, Some(size)) if size > 0.0 && size.is_finite() => {
                                self.asks.insert(key, size);
                            }
                            (Side::Ask, _) => {
                                self.asks.remove(&key);
                            }
                        }
                    }
                }
                _ => return Ok(()),
            }
        }

        if !self.have_snapshot {
            return Ok(());
        }
        if self.next_emit_ts.is_none() {
            self.next_emit_ts = Some(chunk_ts);
        }

        while let Some(emit_ts) = self.next_emit_ts {
            if emit_ts > chunk_ts {
                break;
            }
            if let Some(limit) = limit_ts {
                if emit_ts > limit {
                    break;
                }
            }
            self.emit_snapshot(instrument, emit_ts, depth, writer)?;
            self.next_emit_ts = Some(emit_ts + freq_ms as i64);
        }

        Ok(())
    }

    fn emit_snapshot(
        &self,
        instrument: &str,
        ts: i64,
        depth: Option<usize>,
        writer: &mut csv::Writer<File>,
    ) -> Result<()> {
        let timestamp = ms_to_rfc3339_utc(ts);
        self.write_side(
            instrument,
            ts,
            &timestamp,
            Side::Bid,
            self.bids.iter().rev(),
            depth,
            writer,
        )?;
        self.write_side(
            instrument,
            ts,
            &timestamp,
            Side::Ask,
            self.asks.iter(),
            depth,
            writer,
        )?;
        Ok(())
    }

    fn write_side<'a, I>(
        &self,
        instrument: &str,
        ts: i64,
        timestamp: &str,
        side: Side,
        iter: I,
        depth: Option<usize>,
        writer: &mut csv::Writer<File>,
    ) -> Result<()>
    where
        I: Iterator<Item = (&'a OrderedFloat<f64>, &'a f64)>,
    {
        let mut written = 0usize;
        for (price, size) in iter {
            if let Some(max_depth) = depth {
                if written >= max_depth {
                    break;
                }
            }
            writer.write_record(&[
                instrument,
                "snapshot",
                side.label(),
                &price.into_inner().to_string(),
                &size.to_string(),
                "",
                &ts.to_string(),
                timestamp,
            ])?;
            written += 1;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct PendingRow {
    side: Option<Side>,
    price: Option<f64>,
    size: Option<f64>,
}

fn parse_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Null => None,
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

fn parse_i64(v: &Value) -> Option<i64> {
    match v {
        Value::Null => None,
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        Value::String(s) => s.parse::<i64>().ok(),
        _ => None,
    }
}

fn ms_to_rfc3339_utc(ms: i64) -> String {
    let secs = ms / 1000;
    let sub_ms = (ms % 1000) as u32;
    let nanos = sub_ms * 1_000_000;
    let dt: DateTime<Utc> = Utc
        .timestamp_opt(secs, nanos)
        .single()
        .unwrap_or_else(|| Utc.timestamp_opt(0, 0).single().unwrap());
    dt.to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

fn derive_output_path(input: &Path) -> PathBuf {
    let mut out = input.to_path_buf();
    out.set_extension("csv");
    out
}

fn derive_snapshot_output_path(input: &Path) -> PathBuf {
    let mut out = input.to_path_buf();
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| format!("{}.snapshots", s))
        .unwrap_or_else(|| "snapshots".to_string());
    out.set_file_name(stem);
    out.set_extension("csv");
    out
}
