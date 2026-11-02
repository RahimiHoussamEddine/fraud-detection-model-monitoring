# Source data

## OpenML `creditcard`, dataset ID 1597

- Record: <https://www.openml.org/d/1597>
- Metadata API: <https://www.openml.org/api/v1/json/data/1597>
- Download: <https://www.openml.org/data/v1/download/1673544/creditcard.arff>
- OpenML file ID: `1673544`
- Catalog MD5: `178bcf9bb1f31a3dfe12d0e577884add`
- Implementation download: the official OpenML Parquet mirror named in the
  dataset metadata (`https://data.openml.org/datasets/0000/1597/dataset_1597.pq`)
- Retrieved snapshot size: `73,231,184` bytes
- Retrieved snapshot SHA-256: `b7efcb35a428bbe22347a05d2437d9177bab07ce61e51214a17bec584ad9496d`
- Mirror HTTP ETag at retrieval: `1593844f40edbdabaf5bddec4649e1c4`
- Catalog target: `Class`
- Catalog row identifier / time: `Time`
- Catalog description: 284,807 transactions across two days; 492 fraud labels.

The benchmark is attributed to Andrea Dal Pozzolo, Olivier Caelen, and Gianluca
Bontempi, with a requested citation to Dal Pozzolo et al., *Calibrating
Probability with Undersampling for Unbalanced Classification*, IEEE SSCI 2015
([IEEE Xplore](https://doi.org/10.1109/SSCI.2015.33)).
The catalog marks the dataset public. The project does not redistribute raw
data; keep the downloaded file in this ignored directory and use the upstream
record for access and terms.

## Snapshot integrity

The Parquet mirror does not share the original ARFF MD5, so the retrieved
Parquet bytes are pinned by size and SHA-256. Record retrieval date, catalog
metadata, and schema audit in the versioned source manifest. Do not replace a
snapshot silently if upstream changes the file.

The initial quality audit found 1,081 exact repeated rows beyond their first
occurrence (19 are fraud-labeled) and 9,144 repeated model-feature vectors when
excluding `Time`. No model-feature vector had conflicting labels. The primary
analysis retains all source rows because the dataset has no transaction ID;
locked test metrics also include an exact-row-collapsed sensitivity view.
