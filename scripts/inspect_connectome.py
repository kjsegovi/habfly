from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc 

RAW_DIR = Path("data/raw")

EXPECTED_FILES = [
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
]

def inspect_feather(path: Path) -> None: 
    with pa.memory_map(str(path), "r") as source: 
        reader = ipc.open_file(source)

        row_count = sum(
            reader.get_batch(index).num_rows for index in range(reader.num_record_batches)
        )

        print(f"\n{path.name}")
        print(f"  Size: {path.stat().st_size,} bytes")
        print(f"  Rows: {row_count:,}")
        print(f"  Record batches: {reader.num_record_batches}")
        print(f"  Columns:")

        for field in reader.schema:
            print(f"   {field.name}: {field.type}")

def main() -> None: 
    for filename in EXPECTED_FILES:
        path = RAW_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"Required dataset file is missing: {path}")

        inspect_feather(path)

if __name__ == "__main__":
    main()