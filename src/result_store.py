import os

import pandas as pd


def _row_key(values):
    #Build a comparable key from one or more identifying cell values.
    return tuple(str(value).strip() for value in values)


def upsert_rows(path, new_rows, key_columns):
    #Update matching rows in an Excel file, append new ones, and leave all other rows unchanged.
    new_rows = pd.DataFrame(new_rows)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if os.path.exists(path):
        # Read the file at save time so changes made during a long run are kept.
        merged = pd.read_excel(path)
        missing_keys = [column for column in key_columns if column not in merged.columns]
        if missing_keys:
            raise ValueError(f"Cannot update {path}: missing key column(s) {missing_keys}.")
    else:
        merged = pd.DataFrame(columns=new_rows.columns)

    columns = list(merged.columns) + [column for column in new_rows.columns if column not in merged.columns]
    merged = merged.reindex(columns=columns).astype(object)
    records = merged.to_dict("records")
    position_by_key = {
        _row_key(record[column] for column in key_columns): position
        for position, record in enumerate(records)
    }
    for new_record in new_rows.to_dict("records"):
        key = _row_key(new_record[column] for column in key_columns)
        if key in position_by_key:
            records[position_by_key[key]].update(new_record)
        else:
            position_by_key[key] = len(records)
            records.append(new_record)

    merged = pd.DataFrame(records, columns=columns).infer_objects()
    # Replace the file only after a complete write, so an interrupted save cannot corrupt it.
    temp_path = os.path.join(os.path.dirname(path) or ".", f".tmp_{os.path.basename(path)}")
    try:
        merged.to_excel(temp_path, index=False)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return merged
