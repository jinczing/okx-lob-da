use std::collections::VecDeque;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Cursor, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use chrono::{Duration as ChronoDuration, NaiveDate};
use clap::Parser;
use humantime::parse_duration;
use ndarray::ArrayView1;
use ndarray_npy::NpzWriter;
use ndarray_npy::{WritableElement, WriteDataError};
use py_literal::Value as PyValue;
use serde::Deserialize;

const NS_PER_MS: i64 = 1_000_000;
const DEPTH_EVENT: u64 = 1;
const TRADE_EVENT: u64 = 2;
const EXCH_EVENT: u64 = 1 << 31;
const LOCAL_EVENT: u64 = 1 << 30;
const BUY_EVENT: u64 = 1 << 29;
const SELL_EVENT: u64 = 1 << 28;

#[derive(Parser, Debug)]
#[command(
    name = "okx-backtest",
    about = "Convert OKX L2 + trades into hftbacktest NPZ feeds"
)]
struct Cli {
    #[arg(long, value_name = "YYYY-MM-DD", help = "First UTC day to include")]
    start: NaiveDate,
    #[arg(long, value_name = "YYYY-MM-DD", help = "Last UTC day to include")]
    end: NaiveDate,
    #[arg(
        long,
        default_value = "btcusdt_l2",
        value_name = "DIR",
        help = "Directory containing BTC-USDT-L2orderbook-*.tar.gz"
    )]
    l2_dir: PathBuf,
    #[arg(
        long,
        default_value = "btcusdt_trade",
        value_name = "DIR",
        help = "Directory containing BTC-USDT-trades-*.zip"
    )]
    trade_dir: PathBuf,
    #[arg(
        long,
        default_value = "backtests",
        value_name = "DIR",
        help = "Output directory for NPZ feeds"
    )]
    output_dir: PathBuf,
    #[arg(long, default_value_t = 1, value_name = "DAYS", help = "Days per NPZ file")]
    days_per_file: usize,
    #[arg(long, default_value_t = 50, value_name = "LEVELS", help = "Depth per side to emit")]
    depth: usize,
    #[arg(long, default_value = "BTC-USDT", value_name = "SYMBOL", help = "Instrument prefix in file names")]
    instrument: String,
    #[arg(
        long,
        default_value = "0ns",
        value_name = "DURATION",
        value_parser = parse_latency_ns,
        help = "Offset added to local_ts (e.g. 500us, 2ms)"
    )]
    local_latency_ns: i64,
    #[arg(
        long,
        default_value_t = false,
        help = "Skip trade prints; emit only L2 depth events"
    )]
    skip_trades: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Write uncompressed NPZ (default is deflate-compressed)"
    )]
    uncompressed: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct HftEvent {
    ev: u64,
    exch_ts: i64,
    local_ts: i64,
    px: f64,
    qty: f64,
    order_id: u64,
    ival: i64,
    fval: f64,
}

// SAFETY: `HftEvent` is a POD type with no padding (8-byte fields only) and uses
// little-endian field descriptors to match the hftbacktest dtype.
unsafe impl WritableElement for HftEvent {
    fn type_descriptor() -> PyValue {
        PyValue::List(vec![
            PyValue::Tuple(vec![PyValue::String("ev".into()), PyValue::String("<u8".into())]),
            PyValue::Tuple(vec![PyValue::String("exch_ts".into()), PyValue::String("<i8".into())]),
            PyValue::Tuple(vec![PyValue::String("local_ts".into()), PyValue::String("<i8".into())]),
            PyValue::Tuple(vec![PyValue::String("px".into()), PyValue::String("<f8".into())]),
            PyValue::Tuple(vec![PyValue::String("qty".into()), PyValue::String("<f8".into())]),
            PyValue::Tuple(vec![PyValue::String("order_id".into()), PyValue::String("<u8".into())]),
            PyValue::Tuple(vec![PyValue::String("ival".into()), PyValue::String("<i8".into())]),
            PyValue::Tuple(vec![PyValue::String("fval".into()), PyValue::String("<f8".into())]),
        ])
    }

    fn write<W: io::Write>(&self, mut writer: W) -> Result<(), WriteDataError> {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                (self as *const HftEvent).cast::<u8>(),
                std::mem::size_of::<HftEvent>(),
            )
        };
        writer.write_all(bytes)?;
        Ok(())
    }

    fn write_slice<W: io::Write>(slice: &[Self], mut writer: W) -> Result<(), WriteDataError> {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                slice.as_ptr().cast::<u8>(),
                slice.len() * std::mem::size_of::<HftEvent>(),
            )
        };
        writer.write_all(bytes)?;
        Ok(())
    }
}

const _: [(); 64] = [(); std::mem::size_of::<HftEvent>()];

fn main() -> Result<()> {
    let cli = Cli::parse();
    run(cli)
}

fn run(cli: Cli) -> Result<()> {
    if cli.end < cli.start {
        bail!("end date must not be earlier than start date");
    }
    if cli.depth == 0 {
        bail!("depth must be greater than zero");
    }
    if cli.days_per_file == 0 {
        bail!("days_per_file must be greater than zero");
    }

    fs::create_dir_all(&cli.output_dir)
        .with_context(|| format!("unable to create {}", cli.output_dir.display()))?;

    let l2_dates = dates_inclusive(cli.start, cli.end);
    let trade_dates = build_trade_dates(cli.start, cli.end);
    let lower_ts_ms = date_to_timestamp(cli.start)?;
    let upper_ts_ms = date_to_timestamp(cli.end + ChronoDuration::days(1))?;

    let mut l2_stream = L2Stream::new(
        cli.l2_dir.clone(),
        l2_dates.clone(),
        cli.depth,
        cli.instrument.clone(),
    )?;
    let mut trade_stream = if cli.skip_trades {
        None
    } else {
        Some(TradeStream::new(
            cli.trade_dir.clone(),
            trade_dates,
            lower_ts_ms,
            upper_ts_ms,
            cli.instrument.clone(),
        )?)
    };

    for chunk in l2_dates.chunks(cli.days_per_file) {
        if chunk.is_empty() {
            continue;
        }
        let chunk_start = *chunk.first().unwrap();
        let chunk_end = *chunk.last().unwrap();
        let chunk_upper_ts = date_to_timestamp(chunk_end + ChronoDuration::days(1))?;

        let mut events = Vec::new();
        drain_events(
            chunk_upper_ts,
            cli.local_latency_ns,
            &mut l2_stream,
            trade_stream.as_mut(),
            &mut events,
        )?;

        if events.is_empty() {
            continue;
        }

        let fname = format!(
            "{}-{}-{}.npz",
            cli.instrument.replace('-', "").replace('/', "").to_lowercase(),
            chunk_start.format("%Y-%m-%d"),
            chunk_end.format("%Y-%m-%d"),
        );
        let out_path = cli.output_dir.join(fname);
        write_npz(&out_path, &events, !cli.uncompressed)?;
        println!(
            "wrote {} events to {}",
            events.len(),
            out_path.display()
        );
    }

    Ok(())
}

fn drain_events(
    upper_ts_ms: u64,
    local_latency_ns: i64,
    l2: &mut L2Stream,
    mut trades: Option<&mut TradeStream>,
    out: &mut Vec<HftEvent>,
) -> Result<()> {
    loop {
        let l2_ts = l2.peek_ts()?;
        let trade_ts = match trades.as_deref_mut() {
            Some(ts) => ts.peek_ts()?,
            None => None,
        };

        let next = match (l2_ts, trade_ts) {
            (None, None) => break,
            (Some(ts), None) => NextSource::L2(ts),
            (None, Some(ts)) => NextSource::Trade(ts),
            (Some(l2_ts), Some(tr_ts)) => {
                if l2_ts <= tr_ts {
                    NextSource::L2(l2_ts)
                } else {
                    NextSource::Trade(tr_ts)
                }
            }
        };

        match next {
            NextSource::L2(ts) => {
                if ts >= upper_ts_ms {
                    break;
                }
                if let Some(event) = l2.next()? {
                    out.push(event.into_hft(local_latency_ns)?);
                }
            }
            NextSource::Trade(ts) => {
                if ts >= upper_ts_ms {
                    break;
                }
                if let Some(stream) = trades.as_deref_mut() {
                    if let Some(event) = stream.next()? {
                        out.push(event.into_hft(local_latency_ns)?);
                    }
                }
            }
        }
    }
    Ok(())
}

fn write_npz(path: &Path, events: &[HftEvent], compressed: bool) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("unable to create {}", parent.display()))?;
    }
    let file = File::create(path)
        .with_context(|| format!("unable to create output file {}", path.display()))?;
    let mut npz = if compressed {
        NpzWriter::new_compressed(file)
    } else {
        NpzWriter::new(file)
    };
    let view = ArrayView1::from(events);
    npz.add_array("data", &view)?;
    npz.finish()?;
    Ok(())
}

fn parse_latency_ns(src: &str) -> std::result::Result<i64, String> {
    let duration = parse_duration(src).map_err(|err| err.to_string())?;
    i64::try_from(duration.as_nanos())
        .map_err(|_| "latency is too large to fit in i64 nanoseconds".to_string())
}

fn ms_to_ns(ts_ms: u64) -> Result<i64> {
    let ns = ts_ms
        .checked_mul(NS_PER_MS as u64)
        .context("timestamp overflow when converting to ns")?;
    i64::try_from(ns).context("nanosecond timestamp does not fit in i64")
}

#[derive(Clone, Copy, Debug)]
struct DepthEvent {
    ts_ms: u64,
    side: Side,
    price: f64,
    size: f64,
}

impl DepthEvent {
    fn into_hft(self, local_latency_ns: i64) -> Result<HftEvent> {
        let exch_ts = ms_to_ns(self.ts_ms)?;
        Ok(HftEvent {
            ev: EXCH_EVENT
                | LOCAL_EVENT
                | DEPTH_EVENT
                | match self.side {
                    Side::Bid => BUY_EVENT,
                    Side::Ask => SELL_EVENT,
                },
            exch_ts,
            local_ts: exch_ts.saturating_add(local_latency_ns),
            px: self.price,
            qty: self.size,
            order_id: 0,
            ival: 0,
            fval: 0.0,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct TradeEvent {
    ts_ms: u64,
    side: TradeSide,
    price: f64,
    size: f64,
}

impl TradeEvent {
    fn into_hft(self, local_latency_ns: i64) -> Result<HftEvent> {
        let exch_ts = ms_to_ns(self.ts_ms)?;
        Ok(HftEvent {
            ev: EXCH_EVENT
                | LOCAL_EVENT
                | TRADE_EVENT
                | match self.side {
                    TradeSide::Buy => BUY_EVENT,
                    TradeSide::Sell => SELL_EVENT,
                },
            exch_ts,
            local_ts: exch_ts.saturating_add(local_latency_ns),
            px: self.price,
            qty: self.size,
            order_id: 0,
            ival: 0,
            fval: 0.0,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    Bid,
    Ask,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TradeSide {
    Buy,
    Sell,
}

struct L2Stream {
    base_dir: PathBuf,
    instrument: String,
    dates: Vec<NaiveDate>,
    idx: usize,
    depth: usize,
    queue: VecDeque<DepthEvent>,
    current_day: Option<L2DayReader>,
}

impl L2Stream {
    fn new(base_dir: PathBuf, dates: Vec<NaiveDate>, depth: usize, instrument: String) -> Result<Self> {
        Ok(Self {
            base_dir,
            instrument,
            dates,
            idx: 0,
            depth,
            queue: VecDeque::new(),
            current_day: None,
        })
    }

    fn peek_ts(&mut self) -> Result<Option<u64>> {
        self.fill_queue()?;
        Ok(self.queue.front().map(|ev| ev.ts_ms))
    }

    fn next(&mut self) -> Result<Option<DepthEvent>> {
        self.fill_queue()?;
        Ok(self.queue.pop_front())
    }

    fn fill_queue(&mut self) -> Result<()> {
        if !self.queue.is_empty() {
            return Ok(());
        }
        loop {
            if let Some(reader) = self.current_day.as_mut() {
                match reader.next_message(self.depth)? {
                    Some(events) => {
                        self.queue.extend(events);
                        return Ok(());
                    }
                    None => {
                        self.current_day = None;
                        continue;
                    }
                }
            }

            if self.idx >= self.dates.len() {
                return Ok(());
            }
            let date = self.dates[self.idx];
            self.idx += 1;
            let reader = L2DayReader::open(&self.base_dir, &self.instrument, date)?;
            self.current_day = Some(reader);
        }
    }
}

struct L2DayReader {
    lines: io::Lines<BufReader<Cursor<Vec<u8>>>>,
}

impl L2DayReader {
    fn open(base_dir: &Path, instrument: &str, date: NaiveDate) -> Result<Self> {
        let file_name = l2_file_name(instrument, date);
        let path = base_dir.join(&file_name);
        let file = File::open(&path)
            .with_context(|| format!("unable to open order book file {}", path.display()))?;
        let gz = flate2::read::GzDecoder::new(file);
        let mut archive = tar::Archive::new(gz);
        let mut entries = archive.entries()?;
        let mut entry = entries
            .next()
            .ok_or_else(|| anyhow::anyhow!("empty tar archive in {}", file_name))??;
        let mut data = Vec::new();
        entry.read_to_end(&mut data)?;
        let reader = BufReader::new(Cursor::new(data));
        Ok(Self {
            lines: reader.lines(),
        })
    }

    fn next_message(&mut self, depth: usize) -> Result<Option<Vec<DepthEvent>>> {
        while let Some(line) = self.lines.next() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            match serde_json::from_str::<RawL2Message>(trimmed) {
                Ok(raw) => match raw.into_events(depth) {
                    Ok(events) => return Ok(Some(events)),
                    Err(err) => {
                        eprintln!("skipping malformed L2 message: {err}");
                        continue;
                    }
                },
                Err(err) => {
                    eprintln!("skipping invalid JSON line: {err}");
                    continue;
                }
            }
        }
        Ok(None)
    }
}

#[derive(Debug, Deserialize)]
struct RawL2Message {
    #[serde(rename = "instId")]
    _inst_id: String,
    action: String,
    ts: String,
    #[serde(default)]
    asks: Vec<RawLevel>,
    #[serde(default)]
    bids: Vec<RawLevel>,
}

#[derive(Debug, Deserialize)]
struct RawLevel(
    #[serde(deserialize_with = "de_f64_from_str")] f64,
    #[serde(deserialize_with = "de_f64_from_str")] f64,
    #[serde(default)]
    #[allow(dead_code)]
    serde_json::Value,
);

fn de_f64_from_str<'de, D>(deserializer: D) -> std::result::Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    let value: String = Deserialize::deserialize(deserializer)?;
    value
        .parse::<f64>()
        .map_err(|_| D::Error::custom("expected numeric string"))
}

impl RawL2Message {
    fn into_events(self, depth: usize) -> Result<Vec<DepthEvent>> {
        let action = match self.action.as_str() {
            "snapshot" | "update" => self.action,
            other => bail!("unsupported action {other}"),
        };
        let ts_ms = self.ts.parse::<u64>()?;
        let mut events = Vec::new();

        let take_depth = |levels: Vec<RawLevel>, side: Side, depth: usize| -> Vec<DepthEvent> {
            levels
                .into_iter()
                .take(depth)
                .map(|RawLevel(price, size, _)| DepthEvent {
                    ts_ms,
                    side,
                    price,
                    size,
                })
                .collect()
        };

        events.extend(take_depth(self.asks, Side::Ask, depth));
        events.extend(take_depth(self.bids, Side::Bid, depth));

        if action == "snapshot" && events.is_empty() {
            bail!("snapshot message contained no levels at ts {ts_ms}");
        }

        Ok(events)
    }
}

struct TradeStream {
    base_dir: PathBuf,
    instrument: String,
    dates: Vec<NaiveDate>,
    idx: usize,
    iter: Option<csv::DeserializeRecordsIntoIter<Cursor<Vec<u8>>, RawTradeRecord>>,
    next_trade: Option<TradeEvent>,
    lower_ts_ms: u64,
    upper_ts_ms: u64,
}

impl TradeStream {
    fn new(
        base_dir: PathBuf,
        dates: Vec<NaiveDate>,
        lower_ts_ms: u64,
        upper_ts_ms: u64,
        instrument: String,
    ) -> Result<Self> {
        Ok(Self {
            base_dir,
            instrument,
            dates,
            idx: 0,
            iter: None,
            next_trade: None,
            lower_ts_ms,
            upper_ts_ms,
        })
    }

    fn peek_ts(&mut self) -> Result<Option<u64>> {
        self.load_next_trade()?;
        Ok(self.next_trade.as_ref().map(|t| t.ts_ms))
    }

    fn next(&mut self) -> Result<Option<TradeEvent>> {
        self.load_next_trade()?;
        Ok(self.next_trade.take())
    }

    fn load_next_trade(&mut self) -> Result<()> {
        if self.next_trade.is_some() {
            return Ok(());
        }
        loop {
            if let Some(iter) = self.iter.as_mut() {
                if let Some(record) = iter.next() {
                    let row: RawTradeRecord = record?;
                    let trade = row.into_trade();
                    if trade.ts_ms < self.lower_ts_ms {
                        continue;
                    }
                    if trade.ts_ms >= self.upper_ts_ms {
                        self.iter = None;
                        continue;
                    }
                    self.next_trade = Some(trade);
                    return Ok(());
                } else {
                    self.iter = None;
                }
            }

            if self.idx >= self.dates.len() {
                return Ok(());
            }
            let date = self.dates[self.idx];
            self.idx += 1;
            self.iter = Some(load_trade_iter(&self.base_dir, &self.instrument, date)?);
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawTradeRecord {
    side: String,
    price: f64,
    size: f64,
    #[serde(rename = "created_time")]
    created_time: u64,
}

impl RawTradeRecord {
    fn into_trade(self) -> TradeEvent {
        let side = if self.side.eq_ignore_ascii_case("sell") {
            TradeSide::Sell
        } else {
            TradeSide::Buy
        };
        TradeEvent {
            ts_ms: self.created_time,
            side,
            price: self.price,
            size: self.size,
        }
    }
}

fn load_trade_iter(
    base_dir: &Path,
    instrument: &str,
    date: NaiveDate,
) -> Result<csv::DeserializeRecordsIntoIter<Cursor<Vec<u8>>, RawTradeRecord>> {
    let file_name = trade_file_name(instrument, date);
    let path = base_dir.join(&file_name);
    let file = File::open(&path)
        .with_context(|| format!("unable to open trade file {}", path.display()))?;
    let mut archive = zip::ZipArchive::new(file)?;
    if archive.len() == 0 {
        bail!("trade archive {} contains no entries", file_name);
    }
    let mut entry = archive.by_index(0)?;
    let mut buf = Vec::new();
    entry.read_to_end(&mut buf)?;
    let reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(Cursor::new(buf));
    Ok(reader.into_deserialize())
}

#[derive(Clone, Copy, Debug)]
enum NextSource {
    L2(u64),
    Trade(u64),
}

fn l2_file_name(instrument: &str, date: NaiveDate) -> String {
    format!(
        "{}-L2orderbook-400lv-{}.tar.gz",
        instrument,
        date.format("%Y-%m-%d")
    )
}

fn trade_file_name(instrument: &str, date: NaiveDate) -> String {
    format!("{}-trades-{}.zip", instrument, date.format("%Y-%m-%d"))
}

fn dates_inclusive(start: NaiveDate, end: NaiveDate) -> Vec<NaiveDate> {
    let mut cursor = start;
    let mut dates = Vec::new();
    while cursor <= end {
        dates.push(cursor);
        cursor = cursor.succ_opt().unwrap();
    }
    dates
}

fn build_trade_dates(start: NaiveDate, end: NaiveDate) -> Vec<NaiveDate> {
    let mut dates = dates_inclusive(start, end);
    dates.push(end + ChronoDuration::days(1));
    dates
}

fn date_to_timestamp(date: NaiveDate) -> Result<u64> {
    let dt = date
        .and_hms_opt(0, 0, 0)
        .ok_or_else(|| anyhow::anyhow!("cannot build midnight for {date}"))?;
    Ok(dt.and_utc().timestamp_millis() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latency_parser_accepts_units() {
        assert_eq!(parse_latency_ns("5ms").unwrap(), 5_000_000);
        assert_eq!(parse_latency_ns("250us").unwrap(), 250_000);
    }
}
