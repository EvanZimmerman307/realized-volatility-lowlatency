import polars as pl

class DataLoader:
    @staticmethod
    def _csv(path: str) -> pl.DataFrame:
        return pl.read_csv(path)
    
    @staticmethod
    def _parquet(path: str) -> pl.DataFrame:
        return pl.read_parquet(path)

    @staticmethod
    def load_data_for_time_id(path: str, time_id: int) -> pl.DataFrame:
        df = DataLoader._parquet(path)
        df = df.filter(pl.col("time_id") == time_id)
        return df

    