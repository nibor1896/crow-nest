//! scan real container expert scale bytes for the NaN-encoding 0x7F (ue4m3
//! e=15,m=7 — hardware NaN per mma_probe2; engine CPU twin decodes 480).
use crow_nest_engine::cnq::Cnq;
use crow_nest_engine::geo::*;

fn main() {
    // #52: the probe defaults to the production -M container, like `decode` and `parity` (#51);
    // read only, no sidecar is written here
    let cnq_path = std::env::var("CROW_CNQ").unwrap_or_else(|_| "../converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq".into());
    let mut cnq = Cnq::open(&cnq_path);
    let sec = "text";
    let layers: Vec<usize> = std::env::args()
        .nth(1)
        .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![0, 9]);
    for l in layers {
        for (kind, name) in [
            ("gate_up", format!("model.language_model.layers.{l}.mlp.experts.gate_up_proj")),
            ("down", format!("model.language_model.layers.{l}.mlp.experts.down_proj")),
        ] {
            let t = cnq.find(&name, sec).clone();
            let per_expert = Cnq::byte_len(&t) / E as u64;
            // scale bytes sit at offsets 0..4 of every 36-byte block; sample the
            // stream and classify (pos % 36 < 4) -- read in 8 MB chunks
            let total = Cnq::byte_len(&t);
            let mut hist = [0u64; 256];
            let mut data_bytes = 0u64;
            let mut scale_bytes = 0u64;
            let chunk = 8_000_000u64;
            let mut off = 0u64;
            while off < total {
                let n = chunk.min(total - off);
                let raw = cnq.read_range(&t, off, n as usize);
                for (i, &b) in raw.iter().enumerate() {
                    let pos = off + i as u64;
                    if (pos % per_expert) == per_expert - 1 { continue; } // padding? keep simple
                    if pos % 36 < 4 {
                        scale_bytes += 1;
                        hist[b as usize] += 1;
                    } else {
                        data_bytes += 1;
                    }
                }
                off += n;
            }
            let ge78: u64 = hist[0x78..].iter().sum();
            let n7f = hist[0x7F];
            let mut top = [(0u8, 0u64); 8];
            for (b, &c) in hist.iter().enumerate() {
                if c > top[7].1 {
                    top[7] = (b as u8, c);
                    top.sort_by_key(|x| std::cmp::Reverse(x.1));
                }
            }
            println!(
                "{name} ({kind}): scale bytes={scale_bytes} data bytes={data_bytes} | >=0x78: {ge78} | 0x7F: {n7f} | top bytes: {:?}",
                top.iter().map(|(b, c)| (format!("{b:#04x}"), *c)).collect::<Vec<_>>()
            );
        }
    }
}
