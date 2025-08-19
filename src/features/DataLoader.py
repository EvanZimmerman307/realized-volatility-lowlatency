import polars as pl

class DataLoader:
    @staticmethod
    def _csv(path: str) -> pl.DataFrame:
        return pl.read_csv(path)
    
    @staticmethod
    def _parquet(path: str, logger) -> pl.DataFrame:
        try:
            return pl.read_parquet(path)
        except Exception as e:
            logger.info(e)

    @staticmethod
    def load_data_for_time_id(path: str, time_id: int, logger) -> pl.DataFrame:
        df = DataLoader._parquet(path, logger)
        df = df.filter(pl.col("time_id") == time_id)
        return df

    