import pandas as pd

def calculate_mbom_completion(ebom_df, mbom_tc_df, mbom_oracle_df):
    # Calculate MBOM Oracle values
    mbom_oracle_makes = mbom_oracle_df[mbom_oracle_df['Make or Buy'] == 'MAKE']
    mbom_oracle_buys = mbom_oracle_df[mbom_oracle_df['Make or Buy'] == 'BUY']
    mbom_oracle_make_count = mbom_oracle_makes.shape[0]
    mbom_oracle_buy_count = mbom_oracle_buys.shape[0]

    # Calculate MBOM TeamCenter values
    mbom_tc_makes = mbom_tc_df[mbom_tc_df['Make or Buy'] == 'MAKE']
    mbom_tc_buys = mbom_tc_df[mbom_tc_df['Make or Buy'] == 'BUY']
    mbom_tc_make_count = mbom_tc_makes.shape[0]
    mbom_tc_buy_count = mbom_tc_buys.shape[0]

    # Calculate EBOM values
    ebom_makes = ebom_df[ebom_df['Make or Buy'] == 'Make']
    ebom_buys = ebom_df[ebom_df['Make or Buy'] == 'Buy']
    ebom_make_count = ebom_makes.shape[0]
    ebom_buy_count = ebom_buys.shape[0]

    # Calculate percentages
    data = [
        {
            "source": "Oracle",
            "make_or_buy": "Make",
            "percent_matched": (mbom_oracle_make_count / ebom_make_count) * 100 if ebom_make_count else 0
        },
        {
            "source": "Oracle",
            "make_or_buy": "Buy",
            "percent_matched": (mbom_oracle_buy_count / ebom_buy_count) * 100 if ebom_buy_count else 0
        },
        {
            "source": "TeamCenter",
            "make_or_buy": "Make",
            "percent_matched": (mbom_tc_make_count / ebom_make_count) * 100 if ebom_make_count else 0
        },
        {
            "source": "TeamCenter",
            "make_or_buy": "Buy",
            "percent_matched": (mbom_tc_buy_count / ebom_buy_count) * 100 if ebom_buy_count else 0
        }
    ]

    return pd.DataFrame(data)