//! Parity for the labeled-CSV export: `read_table` + two appended columns +
//! `to_csv_string` must reproduce, byte for byte, what pandas wrote in
//! `generate_labeled_csv.py` (Python generator, see `tests/fixtures/golden/README.md`)
//! (`pd.read_csv` + `to_csv(index=False)`), including NaN, exponents,
//! `-0.0`, booleans with gaps and quoted fields.

use pulsar_data::{read_table, to_csv_string, Column};
use pulsar_parity::golden_dir;
use serde_json::Value;

#[test]
fn labeled_csv_matches_pandas_byte_for_byte() {
    let path = golden_dir().join("export/labeled_csv.json");
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    for case in fx["cases"].as_array().unwrap() {
        let source = case["source"].as_str().unwrap();
        let mut table = read_table(golden_dir().join(source)).unwrap();
        let labels: Vec<i64> = serde_json::from_value(case["labels"].clone()).unwrap();
        let names: Vec<Option<String>> = labels
            .iter()
            .map(|l| case["names"][l.to_string()].as_str().map(str::to_owned))
            .collect();
        table.names.push("topological_cluster_id".into());
        table.columns.push(Column::Int(labels));
        table.names.push("topological_cluster_name".into());
        table.columns.push(Column::Text(names));
        let got = to_csv_string(&table);
        let want = case["csv"].as_str().unwrap();
        if got != want {
            let (g, w): (Vec<&str>, Vec<&str>) = (got.lines().collect(), want.lines().collect());
            let k = g
                .iter()
                .zip(&w)
                .position(|(a, b)| a != b)
                .unwrap_or(g.len().min(w.len()));
            panic!(
                "{source}: first difference at line {k}\n got: {:?}\nwant: {:?}",
                g.get(k),
                w.get(k)
            );
        }
    }
}
