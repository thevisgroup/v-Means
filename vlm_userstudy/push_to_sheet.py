# -*- coding: utf-8 -*-
"""
Push outputs/vlm_responses.csv into the study Google Spreadsheet,
tab "VLM_Responses" (created if missing).

Default mode is APPEND-ONLY: rows already present in the sheet — matched
by (model_tag, run_id, timestamp) — are skipped, so manual notes or extra
columns you add by hand in the tab are never destroyed.

Use --replace to wipe and rewrite the whole tab instead.

Setup (once):
  1. Google Cloud console -> create a service account -> download JSON key.
  2. Enable "Google Sheets API" for that project.
  3. Share the study spreadsheet with the service-account email (Editor).
  4. export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
  5. export STUDY_SHEET_ID=<the long id in the spreadsheet URL>

Usage:  python push_to_sheet.py [--replace]
"""

import argparse
import csv
import os
import sys

import gspread

import config as C

TAB_NAME = "VLM_Responses"
KEY_COLS = ("model_tag", "run_id", "timestamp")


def row_key(header, row):
    idx = {h: i for i, h in enumerate(header)}
    try:
        return tuple(str(row[idx[k]]) for k in KEY_COLS)
    except (KeyError, IndexError):
        return None


def header_is_compatible(sheet_header, csv_header):
    """Allow manual Sheet columns only after the CSV-owned prefix."""
    return (len(sheet_header) >= len(csv_header)
            and sheet_header[:len(csv_header)] == csv_header)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replace", action="store_true",
                    help="clear the tab and rewrite everything")
    args = ap.parse_args()

    key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    sheet_id = os.environ.get("STUDY_SHEET_ID")
    if not key_path or not sheet_id:
        sys.exit("set GOOGLE_APPLICATION_CREDENTIALS and STUDY_SHEET_ID "
                 "(see docstring)")
    if not os.path.exists(C.CSV_PATH):
        sys.exit(f"nothing to push: {C.CSV_PATH} not found")

    with open(C.CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        sys.exit("CSV has no data rows")
    header, data = rows[0], rows[1:]

    gc = gspread.service_account(filename=key_path)
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(TAB_NAME)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=TAB_NAME,
                              rows=len(rows) + 10, cols=len(header) + 2)
        ws.update([header], "A1")

    if args.replace:
        ws.clear()
        ws.update(rows, "A1")
        print(f"REPLACED tab '{TAB_NAME}' with {len(data)} rows")
        return

    existing = ws.get_all_values()
    if not existing:
        ws.update([header], "A1")
        existing = [header]
    sheet_header = existing[0]
    if not header_is_compatible(sheet_header, header):
        sys.exit("the Sheet's leading columns differ from the CSV header — "
                 "resolve manually or run with --replace. Extra manual "
                 "columns are supported only after the CSV columns.")
    seen = {row_key(sheet_header, r) for r in existing[1:]}
    new = []
    for row in data:
        key = row_key(header, row)
        if key not in seen:
            new.append(row)
            seen.add(key)
    if not new:
        print("nothing new to append")
        return
    ws.append_rows(new, value_input_option="RAW")
    print(f"appended {len(new)} new rows to tab '{TAB_NAME}' "
          f"({len(data) - len(new)} already present)")


if __name__ == "__main__":
    main()
