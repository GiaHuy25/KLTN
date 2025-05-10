import math
import numpy as np
import pandas as pd

def paginate_dataframe(df: pd.DataFrame, page: int = 1, per_page: int = 5) -> dict:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    total_records = len(df)

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page

    paginated_df = df.iloc[start_idx:end_idx]

    paginated_data = paginated_df.to_dict(orient='records')

    return {
        "data": paginated_data,
        "total_record": total_records,
        "page": page,
        "per_page": per_page
    }