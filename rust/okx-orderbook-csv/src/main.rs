use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Cursor, Read};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Result};
use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use clap::{ArgAction, Args, Parser, Subcommand};
use flate2::read::MultiGzDecoder;
use ordered_float::OrderedFloat;
use serde::Deserialize;
use serde_json::Value;
use tempfile::TempDir;
use zip::read::ZipArchive;

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Convert(args) => run_convert(args),
        Command::Rebuild(args) => run_rebuild(args),
        Command::Features(args) => run_features(args),
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
    /// Emit fixed-interval features from L2 `.tar.gz` and trade `.zip` archives
    Features(FeatureArgs),
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

#[derive(Args, Debug, Clone)]
struct FeatureArgs {
    /// Directory containing L2 `.tar.gz` dumps (one per day)
    #[arg(long = "l2-dir")]
    l2_dir: PathBuf,

    /// Directory containing trade `.zip` files (one per day)
    #[arg(long = "trade-dir")]
    trade_dir: PathBuf,

    /// Output CSV path (defaults to `<l2-dir>/features.csv`)
    #[arg(short = 'o', long = "output")]
    output: Option<PathBuf>,

    /// Snapshot frequency in milliseconds
    #[arg(long = "freq-ms", default_value_t = 1000)]
    freq_ms: u64,

    /// Number of best price levels per side to emit
    #[arg(long = "depth", default_value_t = 5)]
    depth: usize,

    /// Optional instrument filter (process all instruments when omitted)
    #[arg(long = "instrument")]
    instrument: Option<String>,

    /// Inclusive start date in `YYYY-MM-DD` (UTC)
    #[arg(long = "start-date")]
    start_date: String,

    /// Inclusive end date in `YYYY-MM-DD` (UTC)
    #[arg(long = "end-date")]
    end_date: String,
}

struct FeatureInstrumentState {
    bids: BTreeMap<OrderedFloat<f64>, f64>,
    asks: BTreeMap<OrderedFloat<f64>, f64>,
    trade_buy_volume: f64,
    trade_sell_volume: f64,
    trade_vwap_num: f64,
    trade_vwap_den: f64,
    next_emit_ts: Option<i64>,
}

impl FeatureInstrumentState {
    fn new() -> Self {
        Self {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            trade_buy_volume: 0.0,
            trade_sell_volume: 0.0,
            trade_vwap_num: 0.0,
            trade_vwap_den: 0.0,
            next_emit_ts: None,
        }
    }

    fn handle_orderbook_event(
        &mut self,
        instrument: &str,
        event: OrderbookEvent,
        freq_ms: u64,
        depth: usize,
        emit_output: bool,
        writer: &mut csv::Writer<File>,
    ) -> Result<()> {
        self.emit_due_snapshots(instrument, event.ts, freq_ms, depth, emit_output, writer)?;

        match event.action.as_deref() {
            Some("snapshot") => self.apply_snapshot(&event.bids, &event.asks),
            _ => {
                self.apply_updates(Side::Bid, &event.bids);
                self.apply_updates(Side::Ask, &event.asks);
            }
        }

        Ok(())
    }

    fn handle_trade_event(
        &mut self,
        instrument: &str,
        trade: &TradeEvent,
        freq_ms: u64,
        depth: usize,
        emit_output: bool,
        writer: &mut csv::Writer<File>,
    ) -> Result<()> {
        self.emit_due_snapshots(instrument, trade.ts, freq_ms, depth, emit_output, writer)?;
        self.record_trade(trade);
        Ok(())
    }

    fn apply_snapshot(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)]) {
        self.bids.clear();
        self.asks.clear();
        for &(price, size) in bids {
            if !price.is_finite() || !size.is_finite() || size <= 0.0 {
                continue;
            }
            self.bids.insert(OrderedFloat(price), size);
        }
        for &(price, size) in asks {
            if !price.is_finite() || !size.is_finite() || size <= 0.0 {
                continue;
            }
            self.asks.insert(OrderedFloat(price), size);
        }
    }

    fn apply_updates(&mut self, side: Side, updates: &[(f64, f64)]) {
        if updates.is_empty() {
            return;
        }
        let book = match side {
            Side::Bid => &mut self.bids,
            Side::Ask => &mut self.asks,
        };
        for &(price, size) in updates {
            if !price.is_finite() || !size.is_finite() {
                continue;
            }
            let key = OrderedFloat(price);
            if size > 0.0 {
                book.insert(key, size);
            } else {
                book.remove(&key);
            }
        }
    }

    fn record_trade(&mut self, trade: &TradeEvent) {
        if !trade.price.is_finite() || !trade.size.is_finite() || trade.size <= 0.0 {
            return;
        }
        match trade.side {
            TradeSide::Buy => {
                self.trade_buy_volume += trade.size;
            }
            TradeSide::Sell => {
                self.trade_sell_volume += trade.size;
            }
        }
        self.trade_vwap_num += trade.price * trade.size;
        self.trade_vwap_den += trade.size;
    }

    fn emit_due_snapshots(
        &mut self,
        instrument: &str,
        limit_ts: i64,
        freq_ms: u64,
        depth: usize,
        emit_output: bool,
        writer: &mut csv::Writer<File>,
    ) -> Result<()> {
        let freq = freq_ms.min(i64::MAX as u64) as i64;
        if self.next_emit_ts.is_none() {
            if limit_ts < 0 {
                return Ok(());
            }
            let aligned = align_down(limit_ts, freq);
            self.next_emit_ts = Some(aligned);
        }
        while let Some(next_ts) = self.next_emit_ts {
            if next_ts > limit_ts {
                break;
            }
            if emit_output {
                self.write_snapshot(instrument, next_ts, depth, writer)?;
            }
            self.trade_buy_volume = 0.0;
            self.trade_sell_volume = 0.0;
            self.trade_vwap_num = 0.0;
            self.trade_vwap_den = 0.0;
            self.next_emit_ts = Some(next_ts.saturating_add(freq));
        }
        Ok(())
    }

    fn write_snapshot(
        &self,
        instrument: &str,
        ts: i64,
        depth: usize,
        writer: &mut csv::Writer<File>,
    ) -> Result<()> {
        let timestamp = ms_to_rfc3339_utc(ts);
        let mut record = Vec::with_capacity(3 + depth * 2 + 3);
        record.push(instrument.to_string());
        record.push(ts.to_string());
        record.push(timestamp);
        self.push_sizes(Side::Bid, depth, &mut record);
        self.push_sizes(Side::Ask, depth, &mut record);
        let vwap = if self.trade_vwap_den > 0.0 {
            self.trade_vwap_num / self.trade_vwap_den
        } else {
            self.mid_price().unwrap_or(0.0)
        };
        record.push(vwap.to_string());
        record.push(self.trade_buy_volume.to_string());
        record.push(self.trade_sell_volume.to_string());
        writer.write_record(record)?;
        Ok(())
    }

    fn push_sizes(&self, side: Side, depth: usize, out: &mut Vec<String>) {
        let mut written = 0usize;
        match side {
            Side::Bid => {
                for (_, size) in self.bids.iter().rev() {
                    if written >= depth {
                        break;
                    }
                    out.push((*size).max(0.0).to_string());
                    written += 1;
                }
            }
            Side::Ask => {
                for (_, size) in self.asks.iter() {
                    if written >= depth {
                        break;
                    }
                    out.push((*size).max(0.0).to_string());
                    written += 1;
                }
            }
        }
        while written < depth {
            out.push("0".to_string());
            written += 1;
        }
    }

    fn mid_price(&self) -> Option<f64> {
        let bid = self.bids.iter().rev().next().map(|(p, _)| p.into_inner());
        let ask = self.asks.iter().next().map(|(p, _)| p.into_inner());
        match (bid, ask) {
            (Some(b), Some(a)) => Some((a + b) * 0.5),
            (Some(b), None) => Some(b),
            (None, Some(a)) => Some(a),
            (None, None) => None,
        }
    }

    fn set_next_emit_ts(&mut self, ts: i64) {
        self.next_emit_ts = Some(match self.next_emit_ts {
            Some(existing) => existing.max(ts),
            None => ts,
        });
    }
}

struct OrderbookEvent {
    instrument: String,
    ts: i64,
    action: Option<String>,
    bids: Vec<(f64, f64)>,
    asks: Vec<(f64, f64)>,
}

struct TradeEvent {
    instrument: String,
    ts: i64,
    side: TradeSide,
    price: f64,
    size: f64,
}

#[derive(Clone, Copy)]
enum TradeSide {
    Buy,
    Sell,
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

fn run_features(args: FeatureArgs) -> Result<()> {
    let FeatureArgs {
        l2_dir,
        trade_dir,
        output,
        freq_ms,
        depth,
        instrument,
        start_date,
        end_date,
    } = args;

    if !l2_dir.exists() || !l2_dir.is_dir() {
        bail!(
            "L2 directory not found or not a directory: {}",
            l2_dir.display()
        );
    }
    if !trade_dir.exists() || !trade_dir.is_dir() {
        bail!(
            "Trade directory not found or not a directory: {}",
            trade_dir.display()
        );
    }

    let start_date = parse_ymd(&start_date)?;
    let end_date = parse_ymd(&end_date)?;
    if end_date < start_date {
        bail!("end-date must be >= start-date");
    }

    let l2_archives = collect_archives(&l2_dir, ArchiveKind::Orderbook)?;
    let trade_archives = collect_archives(&trade_dir, ArchiveKind::Trade)?;
    if l2_archives.is_empty() {
        bail!("No `.tar.gz` files discovered in {}", l2_dir.display());
    }
    if trade_archives.is_empty() {
        bail!("No `.zip` files discovered in {}", trade_dir.display());
    }

    let output_path = output.unwrap_or_else(|| derive_features_output_path(&l2_dir));
    let mut writer = csv::Writer::from_path(&output_path)?;

    let mut header = vec![
        "instrument".to_string(),
        "ts".to_string(),
        "timestamp".to_string(),
    ];
    for i in 0..depth {
        header.push(format!("bid_size_{}", i + 1));
    }
    for i in 0..depth {
        header.push(format!("ask_size_{}", i + 1));
    }
    header.push("vwap".to_string());
    header.push("buy_volume".to_string());
    header.push("sell_volume".to_string());
    writer.write_record(header)?;

    let freq_ms = freq_ms.max(1);
    let instrument_filter = instrument.as_deref();
    let mut states: HashMap<String, FeatureInstrumentState> = HashMap::new();

    let start_ms = date_to_ms(start_date)?;
    if let Some(prev_day) = start_date.pred_opt() {
        if let Some(prev_trade_path) = trade_archives.get(&prev_day) {
            eprintln!(
                "Seeding trade state from previous day {} ({})",
                prev_trade_path.display(),
                prev_day
            );
            seed_previous_day_trades(
                prev_trade_path,
                start_ms,
                freq_ms,
                instrument_filter,
                &mut states,
            )?;
        } else {
            eprintln!(
                "Warning: trade archive missing for previous day {}; first VWAP interval may start empty",
                prev_day
            );
        }
    }

    let mut day = start_date;
    while day <= end_date {
        let l2_path = l2_archives
            .get(&day)
            .ok_or_else(|| anyhow!("Missing L2 archive for {}", day))?;
        let trade_path = trade_archives
            .get(&day)
            .ok_or_else(|| anyhow!("Missing trade archive for {}", day))?;
        eprintln!("Processing {}", day);
        process_day(
            l2_path,
            Some(trade_path),
            &mut states,
            freq_ms,
            depth,
            instrument_filter,
            true,
            &mut writer,
        )?;
        day = day
            .succ_opt()
            .ok_or_else(|| anyhow!("Date overflow while iterating range"))?;
    }

    writer.flush()?;
    eprintln!(
        "Wrote feature CSV for {} .. {}: {}",
        start_date,
        end_date,
        output_path.display()
    );
    Ok(())
}

fn process_day(
    l2_path: &Path,
    trade_path: Option<&Path>,
    states: &mut HashMap<String, FeatureInstrumentState>,
    freq_ms: u64,
    depth: usize,
    instrument_filter: Option<&str>,
    emit_output: bool,
    writer: &mut csv::Writer<File>,
) -> Result<()> {
    let extracted = extract_orderbook_file(l2_path)?;
    let l2_file = File::open(&extracted.path)?;
    let mut orderbook_stream = OrderbookStream::new(l2_file);
    let mut trade_stream = if let Some(path) = trade_path {
        Some(TradeStream::new(read_trade_zip(path)?)?)
    } else {
        None
    };

    let mut next_orderbook = orderbook_stream.next_event(instrument_filter)?;
    let mut next_trade = match trade_stream.as_mut() {
        Some(stream) => stream.next_event(instrument_filter)?,
        None => None,
    };
    loop {
        let take_trade = match (&next_orderbook, &next_trade) {
            (Some(ob), Some(tr)) => tr.ts <= ob.ts,
            (None, Some(_)) => true,
            (Some(_), None) => false,
            (None, None) => break,
        };

        if take_trade {
            let trade = next_trade.take().unwrap();
            let state = states
                .entry(trade.instrument.clone())
                .or_insert_with(FeatureInstrumentState::new);
            state.handle_trade_event(
                &trade.instrument,
                &trade,
                freq_ms,
                depth,
                emit_output,
                writer,
            )?;
            next_trade = match trade_stream.as_mut() {
                Some(stream) => stream.next_event(instrument_filter)?,
                None => None,
            };
        } else if let Some(event) = next_orderbook.take() {
            let instrument = event.instrument.clone();
            let state = states
                .entry(instrument.clone())
                .or_insert_with(FeatureInstrumentState::new);
            state.handle_orderbook_event(
                &instrument,
                event,
                freq_ms,
                depth,
                emit_output,
                writer,
            )?;
            next_orderbook = orderbook_stream.next_event(instrument_filter)?;
        }
    }

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

struct OrderbookStream {
    lines: std::io::Lines<BufReader<File>>,
}

impl OrderbookStream {
    fn new(file: File) -> Self {
        Self {
            lines: BufReader::new(file).lines(),
        }
    }

    fn next_event(&mut self, instrument_filter: Option<&str>) -> Result<Option<OrderbookEvent>> {
        while let Some(line_res) = self.lines.next() {
            let line = match line_res {
                Ok(s) => s,
                Err(err) => {
                    eprintln!("Failed to read L2 line: {}", err);
                    continue;
                }
            };
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let payload: Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(err) => {
                    eprintln!("Failed to parse L2 JSON: {}", err);
                    continue;
                }
            };
            let ts = match payload.get("ts").and_then(parse_i64) {
                Some(value) => value,
                None => continue,
            };
            let instrument = match payload
                .get("instId")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
            {
                Some(val) => val,
                None => continue,
            };
            if let Some(filter) = instrument_filter {
                if instrument != filter {
                    continue;
                }
            }
            let action = payload
                .get("action")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let bids = extract_levels(payload.get("bids"));
            let asks = extract_levels(payload.get("asks"));
            return Ok(Some(OrderbookEvent {
                instrument,
                ts,
                action,
                bids,
                asks,
            }));
        }
        Ok(None)
    }
}

struct TradeStream {
    iter: csv::StringRecordsIntoIter<Cursor<Vec<u8>>>,
}

impl TradeStream {
    fn new(data: Vec<u8>) -> Result<Self> {
        let cursor = Cursor::new(data);
        let reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(cursor);
        Ok(Self {
            iter: reader.into_records(),
        })
    }

    fn next_event(&mut self, instrument_filter: Option<&str>) -> Result<Option<TradeEvent>> {
        while let Some(record_res) = self.iter.next() {
            let record = record_res?;
            let instrument = record.get(0).unwrap_or_default().to_string();
            if let Some(filter) = instrument_filter {
                if instrument != filter {
                    continue;
                }
            }
            let ts = match record.get(5).and_then(|v| v.parse::<i64>().ok()) {
                Some(value) => value,
                None => continue,
            };
            let price = match record.get(3).and_then(|v| v.parse::<f64>().ok()) {
                Some(value) => value,
                None => continue,
            };
            let size = match record.get(4).and_then(|v| v.parse::<f64>().ok()) {
                Some(value) => value,
                None => continue,
            };
            let side = match record.get(2).map(|s| s.to_ascii_lowercase()) {
                Some(ref s) if s == "buy" => TradeSide::Buy,
                Some(ref s) if s == "sell" => TradeSide::Sell,
                _ => continue,
            };
            return Ok(Some(TradeEvent {
                instrument,
                ts,
                side,
                price,
                size,
            }));
        }
        Ok(None)
    }
}

struct ExtractedDataFile {
    #[allow(dead_code)]
    tempdir: TempDir,
    path: PathBuf,
}

fn extract_orderbook_file(path: &Path) -> Result<ExtractedDataFile> {
    let file = File::open(path)?;
    let decoder = MultiGzDecoder::new(file);
    let mut archive = tar::Archive::new(decoder);
    let tempdir = TempDir::new()?;
    for entry_res in archive.entries()? {
        let mut entry = entry_res?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let entry_path = match entry.path() {
            Ok(p) => p.into_owned(),
            Err(err) => {
                eprintln!(
                    "Skipping entry inside {} due to invalid path: {}",
                    path.display(),
                    err
                );
                continue;
            }
        };
        let is_data = entry_path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("data"))
            .unwrap_or(false);
        if !is_data {
            continue;
        }
        let filename = entry_path
            .file_name()
            .map(|name| name.to_owned())
            .unwrap_or_else(|| "orderbook.data".into());
        let out_path = tempdir.path().join(filename);
        entry.unpack(&out_path)?;
        return Ok(ExtractedDataFile {
            tempdir,
            path: out_path,
        });
    }
    bail!("No `.data` file found inside {}", path.display());
}

fn read_trade_zip(path: &Path) -> Result<Vec<u8>> {
    let file = File::open(path)?;
    let mut archive = ZipArchive::new(file)?;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        if !entry.name().ends_with(".csv") {
            continue;
        }
        let mut buffer = Vec::new();
        entry.read_to_end(&mut buffer)?;
        return Ok(buffer);
    }
    bail!("No CSV payload found inside {}", path.display());
}

fn seed_previous_day_trades(
    trade_path: &Path,
    next_boundary_ms: i64,
    freq_ms: u64,
    instrument_filter: Option<&str>,
    states: &mut HashMap<String, FeatureInstrumentState>,
) -> Result<()> {
    let data = read_trade_zip(trade_path)?;
    let interval_start = next_boundary_ms.saturating_sub(freq_ms.min(i64::MAX as u64) as i64);
    let cursor = Cursor::new(data);
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(cursor);
    for record_res in reader.records() {
        let record = record_res?;
        let instrument = match record.get(0) {
            Some(val) if !val.is_empty() => val.to_string(),
            _ => continue,
        };
        if let Some(filter) = instrument_filter {
            if instrument != filter {
                continue;
            }
        }
        let ts = match record.get(5).and_then(|v| v.parse::<i64>().ok()) {
            Some(value) => value,
            None => continue,
        };
        if ts < interval_start || ts >= next_boundary_ms {
            continue;
        }
        let price = match record.get(3).and_then(|v| v.parse::<f64>().ok()) {
            Some(value) => value,
            None => continue,
        };
        let size = match record.get(4).and_then(|v| v.parse::<f64>().ok()) {
            Some(value) => value,
            None => continue,
        };
        let side = match record.get(2).map(|s| s.to_ascii_lowercase()) {
            Some(ref s) if s == "buy" => TradeSide::Buy,
            Some(ref s) if s == "sell" => TradeSide::Sell,
            _ => continue,
        };
        let trade = TradeEvent {
            instrument: instrument.clone(),
            ts,
            side,
            price,
            size,
        };
        let state = states
            .entry(instrument.clone())
            .or_insert_with(FeatureInstrumentState::new);
        state.record_trade(&trade);
        state.set_next_emit_ts(next_boundary_ms);
    }
    Ok(())
}

enum ArchiveKind {
    Orderbook,
    Trade,
}

fn collect_archives(dir: &Path, kind: ArchiveKind) -> Result<BTreeMap<NaiveDate, PathBuf>> {
    let mut map = BTreeMap::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let matches = match kind {
            ArchiveKind::Orderbook => is_tar_gz(&path),
            ArchiveKind::Trade => path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("zip"))
                .unwrap_or(false),
        };
        if !matches {
            continue;
        }
        if let Some(date) = extract_date_from_filename(&path) {
            map.insert(date, path);
        }
    }
    Ok(map)
}

fn date_to_ms(date: NaiveDate) -> Result<i64> {
    let dt = date
        .and_hms_opt(0, 0, 0)
        .ok_or_else(|| anyhow!("Invalid date: {}", date))?;
    let utc: DateTime<Utc> = Utc.from_utc_datetime(&dt);
    Ok(utc.timestamp_millis())
}

fn parse_ymd(input: &str) -> Result<NaiveDate> {
    NaiveDate::parse_from_str(input, "%Y-%m-%d")
        .map_err(|err| anyhow!("Invalid date '{}': {}", input, err))
}

fn extract_date_from_filename(path: &Path) -> Option<NaiveDate> {
    let name = path.file_name()?.to_str()?;
    if name.len() < 10 {
        return None;
    }
    let bytes = name.as_bytes();
    for idx in 0..=bytes.len().saturating_sub(10) {
        if bytes[idx + 4] == b'-' && bytes[idx + 7] == b'-' {
            let candidate = &name[idx..idx + 10];
            if let Ok(date) = NaiveDate::parse_from_str(candidate, "%Y-%m-%d") {
                return Some(date);
            }
        }
    }
    None
}

fn is_tar_gz(path: &Path) -> bool {
    let Some(extension) = path.extension().and_then(|ext| ext.to_str()) else {
        return false;
    };
    if extension.eq_ignore_ascii_case("gz") {
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            return name.ends_with(".tar.gz");
        }
    }
    false
}

fn derive_features_output_path(dir: &Path) -> PathBuf {
    if dir.is_file() {
        dir.parent()
            .unwrap_or_else(|| Path::new("."))
            .join("features.csv")
    } else {
        dir.join("features.csv")
    }
}

fn extract_levels(value: Option<&Value>) -> Vec<(f64, f64)> {
    let mut levels = Vec::new();
    let Some(array) = value.and_then(|v| v.as_array()) else {
        return levels;
    };
    for entry in array {
        if let Some(items) = entry.as_array() {
            if items.len() < 2 {
                continue;
            }
            let Some(price) = parse_f64(&items[0]) else {
                continue;
            };
            let Some(size) = parse_f64(&items[1]) else {
                continue;
            };
            levels.push((price, size));
        }
    }
    levels
}

fn align_down(ts: i64, freq: i64) -> i64 {
    if freq <= 0 {
        return ts;
    }
    ts - (ts % freq)
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
