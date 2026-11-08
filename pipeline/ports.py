"""Data source ports (Abstract Base Classes) and a synchronous Supabase adapter."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import requests

logger = logging.getLogger(__name__)


class Datasource(ABC):
    """Abstract interface for data ingestion."""

    @abstractmethod
    def fetch_records(self, limit: int, offset: int) -> List[Dict[str, Any]]:
        """Fetch a chunk of records from the data source.

        Args:
            limit: Maximum number of records to fetch
            offset: Number of records to skip (for pagination)
        """
        pass

    @abstractmethod
    def count_records(self) -> int:
        """Return the total number of records available."""
        pass

    def fetch(self, limit: int, offset: int) -> List[Dict[str, Any]]:
        """Fetch records with error handling.

        Args:
            limit: Maximum number of records to fetch
            offset: Number of records to skip
        """
        try:
            return self.fetch_records(limit, offset)
        except Exception as e:
            logger.error(f"Error fetching records: {e}")
            return []


class SupabaseDatasource(Datasource):
    """Synchronous Supabase datasource using requests."""

    def __init__(self, api_url: str, api_key: str, table_name: str):
        self.api_url = api_url
        self.api_key = api_key
        self.table_name = table_name
        self.base_url = api_url.rstrip('/') + '/rest/v1'

    def fetch_records(self, limit: int, offset: int) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/{self.table_name}?limit={limit}&offset={offset}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "apikey": self.api_key,
            "Content-Type": "application/json",
        }
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        raw = resp.json()
        # Map Supabase schema to pipeline-expected field names
        records = []
        for row in raw:
            txn_time = row.get("transaction_time") or ""
            txn_hour = ""
            if "T" in str(txn_time):
                try:
                    txn_hour = str(txn_time).split("T")[1][:2]
                except Exception:
                    pass
            try:
                amt = float(row.get("transaction_amount") or row.get("amount_paise") or 0)
            except (ValueError, TypeError):
                amt = 0.0
            records.append({
                "transaction_id": row.get("transaction_id", ""),
                "user_id": row.get("customer_id", ""),
                "amount": amt,
                "merchant": row.get("merchant_id", ""),
                "risk_score": float(row.get("transaction_amount") or 0),
                "fraud_label": "fraud" if row.get("is_synthetic") else "legit",
                "device_risk_level": row.get("device_id", "unknown"),
                "location_risk_level": row.get("customer_location", "unknown"),
                "time_of_day": txn_hour,
                # Additional fields for the instruction template
                "merchant_name": row.get("merchant_category", row.get("merchant_id", "")),
                "channel": row.get("payment_channel", ""),
                "geo_risk_level": row.get("merchant_location", "unknown"),
                "failed_attempts_24h": int(row.get("failed_transactions") or 0),
                "customer_age_days": int(row.get("customer_age") or 0),
                "account_age_months": int(int(row.get("account_age") or 0) / 30) if row.get("account_age") else 0,
                "customer_transaction_count": int(row.get("previous_transactions") or 0),
                "velocity_last_24h": int(row.get("previous_transactions") or 0),
                "merchant_transaction_count": int(row.get("previous_transactions") or 0),
                "merchant_risk": "low",
                "previous_failed_ratio": float(row.get("failed_transactions") or 0),
            })
        logger.info(f"Fetch {limit}/{offset}: mapped {len(records)} records from Supabase '{self.table_name}'")
        return records

    def count_records(self) -> int:
        url = f"{self.base_url}/{self.table_name}?limit=0"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Prefer": "count=exact",
            "Content-Type": "application/json",
        }
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        count_header = resp.headers.get('Count', '0')
        try:
            return int(count_header)
        except ValueError:
            pass
        # Fallback: fetch a page and count
        resp2 = requests.get(
            f"{self.base_url}/{self.table_name}?limit=1000&offset=0",
            headers={"Authorization": f"Bearer {self.api_key}", "apikey": self.api_key, "Content-Type": "application/json"}
        )
        resp2.raise_for_status()
        data = resp2.json()
        return len(data) if isinstance(data, list) else 0
