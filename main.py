import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation
import os

# 설정값
WINDOW = 30
STD_FACTOR = 3
MAD_FACTOR = 3

def process_column(series):
    # 1. NaN 제거
    series = series.dropna()

    # 2. rolling std로 unstable 제거
    rolling_std = series.rolling(WINDOW).std()
    threshold = rolling_std.median() * STD_FACTOR
    stable = series[rolling_std < threshold]

    # 3. log 변환
    stable = stable[stable > 0]
    log_data = np.log10(stable)

    if len(log_data) == 0:
        return np.nan, np.nan

    # 4. baseline / spike 분리
    median = np.median(log_data)
    mad = median_abs_deviation(log_data)

    lower = median - MAD_FACTOR * mad
    upper = median + MAD_FACTOR * mad

    baseline = log_data[(log_data >= lower) & (log_data <= upper)]
    spike = log_data[log_data > upper]

    # 5. 평균 계산 (log → original)
    baseline_mean = 10 ** baseline.mean() if len(baseline) > 0 else np.nan
    spike_mean = 10 ** spike.mean() if len(spike) > 0 else np.nan

    return baseline_mean, spike_mean


def main():
    input_file = "input.csv"   # 파일 이름 고정 (같은 폴더)
    output_file = "export.csv"

    df = pd.read_csv(input_file, sep=None, engine='python')

    # 시간 컬럼 제거하고 센서만 처리
    columns = df.columns.tolist()
    time_col = columns[0]
    sensor_cols = columns[1:]

    results = {}

    for col in sensor_cols:
        print(f"Processing {col}...")
        baseline_mean, spike_mean = process_column(df[col])

        results[col + "_baseline"] = baseline_mean
        results[col + "_spike"] = spike_mean

    # 결과 저장
    result_df = pd.DataFrame([results])
    result_df.to_csv(output_file, index=False)

    print("export.csv 생성 완료 ✅")


if __name__ == "__main__":
    main()
