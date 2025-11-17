use std::collections::{BTreeMap, VecDeque};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Cursor, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use chrono::{Duration as ChronoDuration, NaiveDate, Utc};
use clap::Parser;
use humantime::parse_duration;
use ordered_float::OrderedFloat;
use serde::Deserialize;

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
    if cli.freq_ms == 0 {
        bail!("sampling frequency must be positive");
    }

    let start_ts = date_to_timestamp(cli.start)?;
    let end_ts = date_to_timestamp(cli.end + ChronoDuration::days(1))?;

    let sampling_days = dates_inclusive(cli.start, cli.end);
    let trade_dates = build_trade_dates(&cli.trade_dir, cli.start, cli.end);

    let mut l2_stream = L2Stream::new(cli.l2_dir.clone(), sampling_days.clone())
        .context("unable to prepare order book stream")?;
    let mut trade_engine = TradeEngine::new(cli.trade_dir.clone(), trade_dates, cli.freq_ms)
        .context("unable to prepare trade stream")?;
    let mut order_book = OrderBook::new();
    let mut writer = FeatureWriter::new(cli.output.clone(), cli.depth)?;

    let mut ts = start_ts;
    while ts < end_ts {
        l2_stream
            .advance_to(ts, &mut order_book)
            .with_context(|| format!("failed while applying L2 events up to {}", ts))?;
        trade_engine
            .advance_to(ts)
            .with_context(|| format!("failed while aggregating trades up to {}", ts))?;

        let (asks, bids) = order_book.snapshot_sizes(cli.depth);
        writer.write_row(
            ts,
            &asks,
            &bids,
            trade_engine.vwap(),
            trade_engine.buy_volume(),
            trade_engine.sell_volume(),
        )?;

        ts += cli.freq_ms;
    }

    writer.finish()?;
    Ok(())
}

#[derive(Parser, Debug)]
#[command(
    name = "okx-features",
    about = "Aggregate depth and trade features for OKX data"
)]
struct Cli {
    #[arg(long, value_name = "YYYY-MM-DD", help = "First day (UTC) to include")]
    start: NaiveDate,
    #[arg(long, value_name = "YYYY-MM-DD", help = "Last day (UTC) to include")]
    end: NaiveDate,
    #[arg(
        long = "freq",
        value_name = "DURATION",
        value_parser = parse_frequency_ms,
        help = "Sampling frequency, e.g. \"1s\" or \"500ms\""
    )]
    freq_ms: u64,
    #[arg(
        long,
        value_name = "LEVELS",
        help = "Number of book levels to export per side"
    )]
    depth: usize,
    #[arg(long, default_value = "btcusdt_l2", value_name = "DIR")]
    l2_dir: PathBuf,
    #[arg(long, default_value = "btcusdt_trade", value_name = "DIR")]
    trade_dir: PathBuf,
    #[arg(long, default_value = "features.csv", value_name = "FILE")]
    output: PathBuf,
}

fn parse_frequency_ms(src: &str) -> std::result::Result<u64, String> {
    let duration = parse_duration(src).map_err(|err| err.to_string())?;
    let millis = duration.as_millis();
    if millis == 0 {
        return Err("duration must be positive".into());
    }
    Ok(millis as u64)
}

fn date_to_timestamp(date: NaiveDate) -> Result<u64> {
    let dt = date
        .and_hms_opt(0, 0, 0)
        .ok_or_else(|| anyhow::anyhow!("cannot build midnight for {date}"))?;
    Ok(dt.and_utc().timestamp_millis() as u64)
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

fn build_trade_dates(trade_dir: &Path, start: NaiveDate, end: NaiveDate) -> Vec<NaiveDate> {
    let mut dates = Vec::new();
    if let Some(prev) = start.pred_opt() {
        let prev_path = trade_dir.join(trade_file_name(prev));
        if prev_path.exists() {
            dates.push(prev);
        }
    }
    dates.extend(dates_inclusive(start, end));
    dates
}

struct FeatureWriter {
    writer: csv::Writer<File>,
    depth: usize,
}

impl FeatureWriter {
    fn new(path: PathBuf, depth: usize) -> Result<Self> {
        let mut writer = csv::WriterBuilder::new()
            .has_headers(true)
            .from_path(&path)
            .with_context(|| format!("unable to create {}", path.display()))?;
        let mut header = vec![
            "timestamp".to_string(),
            "iso_time".to_string(),
            "vwap".to_string(),
            "buy_volume".to_string(),
            "sell_volume".to_string(),
        ];
        for i in 0..depth {
            header.push(format!("ask_size_{}", i + 1));
        }
        for i in 0..depth {
            header.push(format!("bid_size_{}", i + 1));
        }
        writer.write_record(header)?;
        Ok(Self { writer, depth })
    }

    fn write_row(
        &mut self,
        ts: u64,
        asks: &[f64],
        bids: &[f64],
        vwap: f64,
        buy_volume: f64,
        sell_volume: f64,
    ) -> Result<()> {
        let mut record = Vec::with_capacity(5 + self.depth * 2);
        record.push(ts.to_string());
        record.push(format_timestamp(ts));
        record.push(format_float(vwap));
        record.push(format_float(buy_volume));
        record.push(format_float(sell_volume));
        for &value in asks {
            record.push(format_float(value));
        }
        for &value in bids {
            record.push(format_float(value));
        }
        self.writer.write_record(record)?;
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

fn format_timestamp(ts: u64) -> String {
    if let Some(dt) = chrono::DateTime::<Utc>::from_timestamp_millis(ts as i64) {
        dt.to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
    } else {
        "unknown".to_string()
    }
}

fn format_float(value: f64) -> String {
    if value.abs() < 1e-12 {
        "0".to_string()
    } else {
        format!("{value:.8}")
    }
}

struct OrderBook {
    asks: BTreeMap<OrderedFloat<f64>, f64>,
    bids: BTreeMap<OrderedFloat<f64>, f64>,
}

impl OrderBook {
    fn new() -> Self {
        Self {
            asks: BTreeMap::new(),
            bids: BTreeMap::new(),
        }
    }

    fn apply(&mut self, event: L2Event) {
        match event.action {
            L2Action::Snapshot => {
                self.asks.clear();
                self.bids.clear();
                for level in event.asks {
                    self.insert_ask(level);
                }
                for level in event.bids {
                    self.insert_bid(level);
                }
            }
            L2Action::Update => {
                for level in event.asks {
                    self.insert_ask(level);
                }
                for level in event.bids {
                    self.insert_bid(level);
                }
            }
        }
    }

    fn insert_ask(&mut self, level: LevelUpdate) {
        let key = OrderedFloat(level.price);
        if level.size <= 0.0 {
            self.asks.remove(&key);
        } else {
            self.asks.insert(key, level.size);
        }
    }

    fn insert_bid(&mut self, level: LevelUpdate) {
        let key = OrderedFloat(level.price);
        if level.size <= 0.0 {
            self.bids.remove(&key);
        } else {
            self.bids.insert(key, level.size);
        }
    }

    fn snapshot_sizes(&self, depth: usize) -> (Vec<f64>, Vec<f64>) {
        let mut asks = Vec::with_capacity(depth);
        let mut bids = Vec::with_capacity(depth);
        for (_, size) in self.asks.iter().take(depth) {
            asks.push(*size);
        }
        while asks.len() < depth {
            asks.push(0.0);
        }
        for (_, size) in self.bids.iter().rev().take(depth) {
            bids.push(*size);
        }
        while bids.len() < depth {
            bids.push(0.0);
        }
        (asks, bids)
    }
}

struct L2Stream {
    base_dir: PathBuf,
    dates: Vec<NaiveDate>,
    day_idx: usize,
    current_day: Option<L2DayBuffer>,
    next_event: Option<L2Event>,
}

impl L2Stream {
    fn new(base_dir: PathBuf, dates: Vec<NaiveDate>) -> Result<Self> {
        Ok(Self {
            base_dir,
            dates,
            day_idx: 0,
            current_day: None,
            next_event: None,
        })
    }

    fn advance_to(&mut self, ts: u64, book: &mut OrderBook) -> Result<()> {
        while let Some(event) = self.pop_until(ts)? {
            book.apply(event);
        }
        Ok(())
    }

    fn pop_until(&mut self, ts: u64) -> Result<Option<L2Event>> {
        self.load_next_event()?;
        if let Some(event) = &self.next_event {
            if event.ts <= ts {
                return Ok(self.next_event.take());
            }
        }
        Ok(None)
    }

    fn load_next_event(&mut self) -> Result<()> {
        if self.next_event.is_some() {
            return Ok(());
        }
        loop {
            if let Some(day) = self.current_day.as_mut() {
                if let Some(event) = day.read_event()? {
                    self.next_event = Some(event);
                    return Ok(());
                } else {
                    self.current_day = None;
                    continue;
                }
            }

            if self.day_idx >= self.dates.len() {
                return Ok(());
            }
            let date = self.dates[self.day_idx];
            self.day_idx += 1;
            let buffer = L2DayBuffer::load(&self.base_dir, date)?;
            self.current_day = Some(buffer);
        }
    }
}

struct L2DayBuffer {
    lines: io::Lines<BufReader<Cursor<Vec<u8>>>>,
}

impl L2DayBuffer {
    fn load(base_dir: &Path, date: NaiveDate) -> Result<Self> {
        let file_name = l2_file_name(date);
        let path = base_dir.join(&file_name);
        let mut data = Vec::new();
        let file = File::open(&path)
            .with_context(|| format!("unable to open order book file {}", path.display()))?;
        let gz = flate2::read::GzDecoder::new(file);
        let mut archive = tar::Archive::new(gz);
        let mut entries = archive.entries()?;
        let mut entry = entries
            .next()
            .ok_or_else(|| anyhow::anyhow!("empty tar archive in {}", file_name))??;
        entry.read_to_end(&mut data)?;
        let reader = BufReader::new(Cursor::new(data));
        Ok(Self {
            lines: reader.lines(),
        })
    }

    fn read_event(&mut self) -> Result<Option<L2Event>> {
        while let Some(line) = self.lines.next() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            match serde_json::from_str::<RawL2Message>(trimmed) {
                Ok(raw) => match raw.try_into() {
                    Ok(event) => return Ok(Some(event)),
                    Err(err) => {
                        eprintln!("skipping malformed L2 message: {err}");
                    }
                },
                Err(err) => {
                    eprintln!("skipping invalid JSON line: {err}");
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

impl TryFrom<RawL2Message> for L2Event {
    type Error = anyhow::Error;

    fn try_from(raw: RawL2Message) -> Result<Self> {
        let action = match raw.action.as_str() {
            "snapshot" => L2Action::Snapshot,
            "update" => L2Action::Update,
            other => bail!("unsupported action {other}"),
        };
        let ts = raw.ts.parse::<u64>()?;
        let asks = raw
            .asks
            .into_iter()
            .map(|level| {
                let RawLevel(price, size, _) = level;
                LevelUpdate { price, size }
            })
            .collect();
        let bids = raw
            .bids
            .into_iter()
            .map(|level| {
                let RawLevel(price, size, _) = level;
                LevelUpdate { price, size }
            })
            .collect();
        Ok(L2Event {
            action,
            ts,
            asks,
            bids,
        })
    }
}

struct L2Event {
    ts: u64,
    action: L2Action,
    asks: Vec<LevelUpdate>,
    bids: Vec<LevelUpdate>,
}

struct LevelUpdate {
    price: f64,
    size: f64,
}

enum L2Action {
    Snapshot,
    Update,
}

struct TradeEngine {
    stream: TradeStream,
    window: TradeWindow,
}

impl TradeEngine {
    fn new(base_dir: PathBuf, dates: Vec<NaiveDate>, window_ms: u64) -> Result<Self> {
        Ok(Self {
            stream: TradeStream::new(base_dir, dates),
            window: TradeWindow::new(window_ms),
        })
    }

    fn advance_to(&mut self, ts: u64) -> Result<()> {
        while let Some(trade) = self.stream.pop_if_before(ts)? {
            self.window.push(trade);
        }
        self.window.evict(ts);
        Ok(())
    }

    fn vwap(&self) -> f64 {
        self.window.vwap()
    }

    fn buy_volume(&self) -> f64 {
        self.window.buy_volume
    }

    fn sell_volume(&self) -> f64 {
        self.window.sell_volume
    }
}

struct TradeStream {
    base_dir: PathBuf,
    dates: Vec<NaiveDate>,
    day_idx: usize,
    current_iter: Option<csv::DeserializeRecordsIntoIter<Cursor<Vec<u8>>, TradeRecord>>,
    next_trade: Option<Trade>,
}

impl TradeStream {
    fn new(base_dir: PathBuf, dates: Vec<NaiveDate>) -> Self {
        Self {
            base_dir,
            dates,
            day_idx: 0,
            current_iter: None,
            next_trade: None,
        }
    }

    fn pop_if_before(&mut self, ts: u64) -> Result<Option<Trade>> {
        self.load_next_trade()?;
        if let Some(trade) = &self.next_trade {
            if trade.ts <= ts {
                return Ok(self.next_trade.take());
            }
        }
        Ok(None)
    }

    fn load_next_trade(&mut self) -> Result<()> {
        if self.next_trade.is_some() {
            return Ok(());
        }
        loop {
            if let Some(iter) = self.current_iter.as_mut() {
                if let Some(record) = iter.next() {
                    let row: TradeRecord = record?;
                    self.next_trade = Some(row.into_trade());
                    return Ok(());
                } else {
                    self.current_iter = None;
                }
            }

            if self.day_idx >= self.dates.len() {
                return Ok(());
            }

            let date = self.dates[self.day_idx];
            self.day_idx += 1;
            self.current_iter = Some(load_trade_iter(&self.base_dir, date)?);
        }
    }
}

fn load_trade_iter(
    base_dir: &Path,
    date: NaiveDate,
) -> Result<csv::DeserializeRecordsIntoIter<Cursor<Vec<u8>>, TradeRecord>> {
    let file_name = trade_file_name(date);
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

#[derive(Debug, Deserialize)]
struct TradeRecord {
    side: String,
    price: f64,
    size: f64,
    #[serde(rename = "created_time")]
    created_time: u64,
}

impl TradeRecord {
    fn into_trade(self) -> Trade {
        let side = if self.side.eq_ignore_ascii_case("sell") {
            TradeSide::Sell
        } else {
            TradeSide::Buy
        };
        Trade {
            ts: self.created_time,
            price: self.price,
            size: self.size,
            side,
        }
    }
}

#[derive(Clone)]
struct Trade {
    ts: u64,
    price: f64,
    size: f64,
    side: TradeSide,
}

#[derive(Clone)]
enum TradeSide {
    Buy,
    Sell,
}

struct TradeWindow {
    window_ms: u64,
    trades: VecDeque<Trade>,
    buy_volume: f64,
    sell_volume: f64,
    vwap_num: f64,
    vwap_den: f64,
}

impl TradeWindow {
    fn new(window_ms: u64) -> Self {
        Self {
            window_ms,
            trades: VecDeque::new(),
            buy_volume: 0.0,
            sell_volume: 0.0,
            vwap_num: 0.0,
            vwap_den: 0.0,
        }
    }

    fn push(&mut self, trade: Trade) {
        match trade.side {
            TradeSide::Buy => self.buy_volume += trade.size,
            TradeSide::Sell => self.sell_volume += trade.size,
        }
        self.vwap_num += trade.price * trade.size;
        self.vwap_den += trade.size;
        self.trades.push_back(trade);
    }

    fn evict(&mut self, now: u64) {
        if self.window_ms == 0 {
            self.clear();
            return;
        }
        let cutoff = now.saturating_sub(self.window_ms);
        while let Some(front) = self.trades.front() {
            if front.ts <= cutoff {
                let removed = self.trades.pop_front().unwrap();
                self.remove(&removed);
            } else {
                break;
            }
        }
    }

    fn remove(&mut self, trade: &Trade) {
        match trade.side {
            TradeSide::Buy => self.buy_volume -= trade.size,
            TradeSide::Sell => self.sell_volume -= trade.size,
        }
        self.vwap_num -= trade.price * trade.size;
        self.vwap_den -= trade.size;
    }

    fn clear(&mut self) {
        self.trades.clear();
        self.buy_volume = 0.0;
        self.sell_volume = 0.0;
        self.vwap_num = 0.0;
        self.vwap_den = 0.0;
    }

    fn vwap(&self) -> f64 {
        if self.vwap_den.abs() < f64::EPSILON {
            0.0
        } else {
            self.vwap_num / self.vwap_den
        }
    }
}

fn l2_file_name(date: NaiveDate) -> String {
    format!(
        "BTC-USDT-L2orderbook-400lv-{}.tar.gz",
        date.format("%Y-%m-%d")
    )
}

fn trade_file_name(date: NaiveDate) -> String {
    format!("BTC-USDT-trades-{}.zip", date.format("%Y-%m-%d"))
}
