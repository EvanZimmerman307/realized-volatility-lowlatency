import polars as pl

"""
CPU feature engineering for (stock_id, time_id pairs).
"""

class FeatureGenerator:
    
    def __init__(self, book_data: pl.DataFrame, trade_data: pl.DataFrame):
        self.EPS = 1e-9
        self.training_sample = pl.DataFrame({
            "seconds_in_bucket": range(0, 600),
        })
        self._book_data = book_data
        self._trade_data = trade_data
        self.training_sample = self.get_combined_data()
        self._calc_wap()
        self._calc_log_returns()
        self._calc_rolling_realized_volatility()
        self._calc_spread_last()
        self._calc_mid_last()
        self._calc_rel_spread()
        self._calc_ofi_last()
        self._fill_order_count_and_size()
        self._fill_time_id()
        self._calc_rolling_order_count()
        features_to_keep = ["seconds_in_bucket", "size", "order_count", "wap_last",
                            "seconds_since_last_wap", "log_return",	"rv_30s",
                            "rv_120s",	"spread_last",	"mid_last",	"rel_spread_last",	"ofi_last",
                            "seconds_since_last_trade",	"last_trade_gap_geq_cap", "order_intensity_60s"]
        self.training_sample = self.training_sample.select(features_to_keep)
        
    
    def get_combined_data(self):
        join_keys = ["seconds_in_bucket", "time_id"]
        # Full outer join book and trade data
        combined_data = self._book_data.join(
            self._trade_data, on=join_keys, how="outer", suffix="_trade"
        )
        # Left join with training_sample on seconds_in_bucket
        final = self.training_sample.join(
            combined_data, on="seconds_in_bucket", how="left"
        )
        return final
    
    
    def _calc_wap(self):
        EPS = self.EPS
        # compute wap
        wap = (
            self.training_sample["bid_price1"] * self.training_sample["ask_size1"] +
            self.training_sample["ask_price1"] * self.training_sample["bid_size1"]
        ) / (self.training_sample["bid_size1"] + self.training_sample["ask_size1"] + EPS)
        self.training_sample = self.training_sample.with_columns(pl.Series("wap_last", wap))

        # Create last_wap_second
        mask = pl.col("wap_last").is_not_null()
        self.training_sample = self.training_sample.with_columns([
            pl.when(mask).then(pl.col("seconds_in_bucket")).otherwise(None).alias("last_wap_second")
        ])

        # Forward fill last_wap_second
        self.training_sample = self.training_sample.with_columns(
            pl.col("last_wap_second").fill_null(strategy="forward").alias("last_wap_second")
        )

        # secs since last wap (prefix fix + optional cap)
        self.training_sample = self.training_sample.with_columns(
            (pl.col("seconds_in_bucket") - pl.col("last_wap_second")).alias("seconds_since_last_wap")
        )
        self.training_sample = self.training_sample.with_columns(
            pl.when(pl.col("seconds_since_last_wap").is_null())
            .then(pl.col("seconds_in_bucket") + 1)   # prefix before first quote
            .otherwise(pl.col("seconds_since_last_wap"))
            .alias("seconds_since_last_wap")
        )
        STALE_CAP = 15
        self.training_sample = self.training_sample.with_columns(
            pl.min_horizontal([pl.col("seconds_since_last_wap"), pl.lit(STALE_CAP)])
            .alias("seconds_since_last_wap")
        )

        # forward-fill wap (no backfill; leading nulls will remain null, which is fine)
        self.training_sample = self.training_sample.with_columns(
            pl.col("wap_last").fill_null(strategy="forward")
        )

    
    def _calc_log_returns(self):
        # treat non-positive or missing WAP as null BEFORE log
        lr = (
            pl.when(pl.col("wap_last") > 0)
            .then(pl.col("wap_last"))
            .otherwise(None)     
            .log()
            .diff()              # first observed element -> null
            .fill_null(0.0)      # set first diff to 0
            .alias("log_return")
        )
        self.training_sample = self.training_sample.with_columns(lr)

    
    def _realized_volatility(self):
        # Assumes 'log_return' column exists in self.training_sample
        return (
            (pl.col("log_return") ** 2).sum().sqrt()
        )

    def _calc_rolling_realized_volatility(self):
        """
        For rows before 30 seconds (or 120 seconds), 
        the rolling window will include as many previous values as are available, up to the window size
        """
        self.training_sample = self.training_sample.with_columns(
            (pl.col("log_return") ** 2)
            .rolling_sum(window_size=30, min_periods=1)
            .sqrt()
            .alias("rv_30s")
        )
        self.training_sample = self.training_sample.with_columns(
            (pl.col("log_return") ** 2)
            .rolling_sum(window_size=120, min_periods=1)
            .sqrt()
            .alias("rv_120s")
        )

        # Fill first value with 0 for both columns
        self.training_sample = self.training_sample.with_columns([
            pl.col("rv_30s").fill_null(0).alias("rv_30s"),
            pl.col("rv_120s").fill_null(0).alias("rv_120s"),
        ])
    
    def _calc_spread_last(self):
        self.training_sample = self.training_sample.with_columns(
            (pl.col("ask_price1") - pl.col("bid_price1")).alias("spread_last")
        )
        self.training_sample = self.training_sample.with_columns(
            pl.col("spread_last").fill_null(strategy="forward")
        )
    
    def _calc_mid_last(self):
        self.training_sample = self.training_sample.with_columns(
            ((pl.col("ask_price1") + pl.col("bid_price1")) / 2).alias("mid_last")
        )
        self.training_sample = self.training_sample.with_columns(
            pl.col("mid_last").fill_null(strategy="forward")
        )
    
    def _calc_rel_spread(self):
        self.training_sample = self.training_sample.with_columns(
            (pl.col("spread_last") / (pl.col("mid_last") + self.EPS)).alias("rel_spread_last")
        )
    
    def _calc_ofi_last(self):
        self.training_sample = self.training_sample.with_columns(
            ((pl.col("bid_size1") - pl.col("ask_size1")) / 
             (pl.col("bid_size1") + pl.col("ask_size1") + self.EPS)).alias("ofi_last")
        )
        self.training_sample = self.training_sample.with_columns(
            pl.col("ofi_last").fill_null(strategy="forward")
        )
    
    def _fill_order_count_and_size(self):
        # Calculate mask for non-null trade
        mask = pl.col("size").is_not_null()

        # Get seconds_in_bucket only where size is not null, forward fill nulls
        self.training_sample = self.training_sample.with_columns([
            pl.when(mask)
            .then(pl.col("seconds_in_bucket"))
            .otherwise(None)
            .alias("last_trade_second")
        ])
        self.training_sample = self.training_sample.with_columns(
            pl.col("last_trade_second").fill_null(strategy="forward")
        )

        self.training_sample = self.training_sample.with_columns(
            (pl.col("seconds_in_bucket") - pl.col("last_trade_second")).alias("seconds_since_last_trade")
        )

        # count up from 1 for the time before the initial trade
        self.training_sample = self.training_sample.with_columns(
            pl.when(pl.col("seconds_since_last_trade").is_null())
            .then(pl.col("seconds_in_bucket") + 1)
            .otherwise(pl.col("seconds_since_last_trade"))
            .alias("seconds_since_last_trade")
        )

        cap = 15
        self.training_sample = self.training_sample.with_columns(
            pl.min_horizontal([
                pl.col("seconds_since_last_trade"),
                pl.lit(cap)
            ]).alias("seconds_since_last_trade")
        )

        self.training_sample = self.training_sample.with_columns(
            (pl.col("seconds_since_last_trade") >= cap).cast(pl.Int8).alias("last_trade_gap_geq_cap")
        )

        self.training_sample = self.training_sample.with_columns(
            pl.col("order_count").fill_null(0)
        )
        self.training_sample = self.training_sample.with_columns(
            pl.col("size").fill_null(0)
        )
    
    def _calc_rolling_order_count(self):
        """
        For rows before 60 seconds 
        the rolling window will include as many previous values as are available, up to the window size
        """
        self.training_sample = self.training_sample.with_columns(
            (pl.col("order_count"))
            .rolling_sum(window_size=60, min_periods=1)
            .alias("order_intensity_60s")
        )
    
    def _fill_time_id(self):
        self.training_sample = self.training_sample.with_columns(
            pl.col("time_id").fill_null(strategy="forward")
        )
        

        
    
    


        
    

