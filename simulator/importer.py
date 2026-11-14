import csv
import io
import sqlite3

from simulator.database import list_assets, list_transactions

CSV_COLUMNS = ["date", "isin", "kind", "quantity", "unit_price", "fees", "note"]


def export_transactions(conn: sqlite3.Connection) -> str:
    """All trades as CSV text, oldest first. Assets are identified by their ISIN."""
    isins = {asset.id: asset.isin for asset in list_assets(conn)}
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for tx in list_transactions(conn):
        writer.writerow(
            [
                tx.date.isoformat(),
                isins[tx.asset_id],
                tx.kind,
                tx.quantity,
                tx.unit_price,
                tx.fees,
                tx.note,
            ]
        )
    return output.getvalue()
