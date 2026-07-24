import numpy as np
import pandas as pd

def skew_calc(df):
    """
    Diagnoses skewness for every numeric column in a DataFrame and recommends a transformation based on the column's skewness and
    minimum value. Binary, encoded, and ID columns are excluded, since skewness isn't a meaningful for them.
    It returns a DataFrame with the following columns:
    Feature, Skewness, Degree, Direction, Recommended Transformation
    """
    results = []
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        series = df[col].dropna()

        if series.empty or series.nunique() <= 2:
            continue

        skewness = series.skew()

        if abs(skewness) <= 0.5:
            degree = "Approximately Symmetric"
        elif abs(skewness) <= 1:
            degree = "Moderately Skewed"
        else:
            degree = "Highly Skewed"

        if skewness >= 0:
            direction = "Positive"
        else:
            direction = "Negative"

        if abs(skewness) <= 0.5:
            recommendation = "None needed"
        else:
            min_val = series.min()
            if min_val > 0:
                recommendation = "Box-Cox or Yeo-Johnson"
            elif min_val == 0:
                recommendation = "log(x+1) or Yeo-Johnson"
            else:
                recommendation = "Yeo-Johnson"

        results.append({
            "Feature": col,
            "Skewness": round(skewness, 6),
            "Degree": degree,
            "Direction": direction,
            "Recommended Transformation": recommendation
        })

    return pd.DataFrame(results)
