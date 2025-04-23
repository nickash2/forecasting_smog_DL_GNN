from knmy import knmy
from datetime import datetime
import pandas as pd
import time
from requests.exceptions import HTTPError
from typing import Dict, List, Optional
from pathlib import Path


class KNMIDataCollector:
    """Collector for KNMI meteorological data."""

    def __init__(
        self,
        output_dir: str = "data/data_raw",
        start_month: int = 7,
        start_day: int = 1,
        end_month: int = 12,
        end_day: int = 30,
    ):
        """
        Initialize KNMI data collector.

        Args:
            output_dir: Directory to save collected data
            start_month: Starting month for data collection
            start_day: Starting day for data collection
            end_month: Ending month for data collection
            end_day: Ending day for data collection
        """
        self.output_dir = Path(output_dir)
        self.start_month = start_month
        self.start_day = start_day
        self.end_month = end_month
        self.end_day = end_day

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_with_retry(
        self, year: int, city: str, stations: List[int], max_retries: int = 3
    ) -> pd.DataFrame:
        """Fetch data with retry mechanism."""
        for attempt in range(max_retries):
            try:
                s_moment = datetime(year, self.start_month, self.start_day, 0)
                e_moment = datetime(year, self.end_month, self.end_day, 23)

                _, _, _, data = knmy.get_knmi_data(
                    type="hourly",
                    stations=stations,
                    start=s_moment,
                    end=e_moment,
                    inseason=False,
                    parse=True,
                )
                return data

            except HTTPError as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Failed to fetch data for {city} {year} after {max_retries} attempts: {e}"
                    )
                print(f"Attempt {attempt + 1} failed, retrying after delay...")
                time.sleep(5 * (attempt + 1))

    def collect_data(
        self,
        cities: Dict[str, List[int]],
        years: List[int],
        delay_between_cities: int = 2,
        delay_between_years: int = 1,
    ) -> None:
        """
        Collect KNMI data for specified cities and years.

        Args:
            cities: Dictionary mapping city names to station IDs
            years: List of years to collect data for
            delay_between_cities: Delay in seconds between processing cities
            delay_between_years: Delay in seconds between processing years
        """
        for year in years:
            for city, stations in cities.items():
                print(f"Processing {city}...")
                try:
                    data = self.fetch_with_retry(year, city, stations)

                    # Handle duplicate columns and save
                    data = data.loc[:, ~data.columns.duplicated()]
                    output_path = self.output_dir / f"{year}_meteo_{city}.csv"

                    data.to_csv(
                        output_path, index=True, sep=";", decimal=".", encoding="utf-8"
                    )
                    print(f"Saved {city} data with shape {data.shape} for year {year}")

                    time.sleep(delay_between_cities)

                except Exception as e:
                    print(f"Error processing {city} for {year}: {str(e)}")
                    continue

            time.sleep(delay_between_years)
