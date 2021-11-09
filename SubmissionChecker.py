import pandas as pd
import numpy as np
from datetime import datetime

def check_df(df):
    EXPECTED_COLS = ["forecast_date", "target", "horizon", "q0.025", "q0.25",
                     "q0.5", "q0.75", "q0.975"]
    LEN_EXP_COLS = len(EXPECTED_COLS)

    TARGETS = ["DAX", "temperature", "wind"]

    TARGET_VALS = dict(DAX=["1 day", "2 day", "5 day", "6 day", "7 day"],
                       temperature=[str(i) + " hour" for i in np.arange(36, 92, 12)],
                       wind=[str(i) + " hour" for i in np.arange(36, 92, 12)])

    TARGET_PLAUS = dict(DAX=[-20, 20],
                        temperature=[-30, 50],
                        wind=[0, 300])

    COLS_QUANTILES = ["q0.025", "q0.25", "q0.5", "q0.75", "q0.975"]

    print("Start checking...")
    print("---------------------------")
    col_names = df.columns

    print("Checking the Columns...")
    # Check column length
    if len(col_names) != LEN_EXP_COLS:
        print("Dataset contains ", len(col_names), "columns. Required are", LEN_EXP_COLS)
        print("Stopping early...")
        quit()

    if set(col_names) != set(EXPECTED_COLS):
        print("Dataset does not contain the required columns (or more).")
        missing_cols = set(EXPECTED_COLS) - set(col_names)
        print("The missing columns are:", missing_cols)
        print("Stopping early...")
        quit()

    for i, col in enumerate(EXPECTED_COLS):
        if col == col_names[i]:
            continue
        else:
            print("Columns not in correct order. Order should be:", EXPECTED_COLS)
            print("Your order is:", col_names.values)
            print("Stopping early...")
            quit()

    # Date Col
    print("Checking type of columns...")
    try:
        df["forecast_date"] = pd.to_datetime(df["forecast_date"], format="%Y-%m-%d",
                                             errors="raise")
    except (pd.errors.ParserError, ValueError):
        print("Could not parse Date in format YYYY-MM-DD")
        print("Stopping early...")
        quit()

    try:
        df["target"] = df["target"].astype("object", errors="raise")
    except ValueError:
        print("Cannot convert target column to String.")
        print("Stopping early...")
        quit()

    try:
        df["horizon"] = df["horizon"].astype("object", errors="raise")
    except ValueError:
        print("Cannot convert horizon column to String.")
        print("Stopping early...")
        quit()

    for cq in COLS_QUANTILES:
        if pd.to_numeric(df[cq], errors="coerce").isna().any():
            print("Some elements in", cq, "column are not numeric.")
            print("Stopping early...")
            quit()

            # %%
    print("Checking if the Dates make sense...")

    if len(pd.unique(df["forecast_date"])) > 1:
        print("forecast_date needs to be the same in all rows.")
        print("Stopping early...")
        quit()

    if df["forecast_date"][0] < datetime.strptime(datetime.strftime(datetime.today(), '%Y-%m-%d'), '%Y-%m-%d'):
        print("----WARNING: Forecast date should not be in the past")
        # warnings.warn("Forecast date should not be in the past.")

    if df["forecast_date"][0].weekday() != 2:
        print("----WARNING: Forecast date should be a Wednesday")
        # warnings.warn("Forecast date should be a Wednesday")

    print("Checking targets...")

    if not df["target"].isin(TARGETS).all():
        print("Target column can only contain 'DAX', 'temperature' and 'wind'. Check spelling.")
        print("Stopping early...")
        quit()

    for target in TARGETS:

        if len(df[df["target"] == target]) != 5:
            print("Exactly 5 rows need to have target =", target)
            print("Stopping early...")
            quit()

        if (df[df["target"] == target]["horizon"] != TARGET_VALS[target]).any():
            print("Target", target, "horizons need to be (in this order):", TARGET_VALS[target])
            print("Stopping early...")
            quit()

        if (df[df["target"] == target][COLS_QUANTILES] < TARGET_PLAUS[target][0]).any(axis=None) or \
                (df[df["target"] == target][COLS_QUANTILES] > TARGET_PLAUS[target][1]).any(axis=None):
            print("----WARNING: Implausible values for", target, "detected. You may want to re-check.")
            # warnings.warn("Implausible values for "+str(target)+" detected. You may want to re-check them.")

    print("Checking quantiles...")

    for i, row in df.iterrows():
        diffs = row[COLS_QUANTILES].diff()
        if diffs[1:].isna().any():
            print("Something is wrong with your quantiles.")
            print("Stopping early...")
            quit()
        diffs[0] = 0
        if (diffs < 0).any():
            print("Predictive quantiles in row", i, "are not ordered correctly (need to be non-decreasing)")
            print("Stopping early...")
            quit()

    print("---------------------------")
    print("Looks good!")

def check_df_excl_weather(df, exclude_weather):
    EXPECTED_COLS = ["forecast_date", "target", "horizon", "q0.025", "q0.25",
                     "q0.5", "q0.75", "q0.975"]
    LEN_EXP_COLS = len(EXPECTED_COLS)

    if exclude_weather:
        TARGETS = ["DAX"]
    else:
        TARGETS = ["DAX", "temperature", "wind"]

    TARGET_VALS = dict(DAX=["1 day", "2 day", "5 day", "6 day", "7 day"],
                       temperature=[str(i) + " hour" for i in np.arange(36, 92, 12)],
                       wind=[str(i) + " hour" for i in np.arange(36, 92, 12)])

    TARGET_PLAUS = dict(DAX=[-20, 20],
                        temperature=[-30, 50],
                        wind=[0, 300])

    COLS_QUANTILES = ["q0.025", "q0.25", "q0.5", "q0.75", "q0.975"]

    print("Start checking...")
    print("---------------------------")
    col_names = df.columns

    print("Checking the Columns...")
    # Check column length
    if len(col_names) != LEN_EXP_COLS:
        print("Dataset contains ", len(col_names), "columns. Required are", LEN_EXP_COLS)
        print("Stopping early...")
        quit()

    if set(col_names) != set(EXPECTED_COLS):
        print("Dataset does not contain the required columns (or more).")
        missing_cols = set(EXPECTED_COLS) - set(col_names)
        print("The missing columns are:", missing_cols)
        print("Stopping early...")
        quit()

    for i, col in enumerate(EXPECTED_COLS):
        if col == col_names[i]:
            continue
        else:
            print("Columns not in correct order. Order should be:", EXPECTED_COLS)
            print("Your order is:", col_names.values)
            print("Stopping early...")
            quit()

    # Date Col
    print("Checking type of columns...")
    try:
        df["forecast_date"] = pd.to_datetime(df["forecast_date"], format="%Y-%m-%d",
                                             errors="raise")
    except (pd.errors.ParserError, ValueError):
        print("Could not parse Date in format YYYY-MM-DD")
        print("Stopping early...")
        quit()

    try:
        df["target"] = df["target"].astype("object", errors="raise")
    except ValueError:
        print("Cannot convert target column to String.")
        print("Stopping early...")
        quit()

    try:
        df["horizon"] = df["horizon"].astype("object", errors="raise")
    except ValueError:
        print("Cannot convert horizon column to String.")
        print("Stopping early...")
        quit()

    for cq in COLS_QUANTILES:
        if pd.to_numeric(df[cq], errors="coerce").isna().any():
            print("Some elements in", cq, "column are not numeric.")
            print("Stopping early...")
            quit()

    print("Checking if the Dates make sense...")

    if len(pd.unique(df["forecast_date"])) > 1:
        print("forecast_date needs to be the same in all rows.")
        print("Stopping early...")
        quit()

    if df["forecast_date"][0] < datetime.strptime(datetime.strftime(datetime.today(), '%Y-%m-%d'), '%Y-%m-%d'):
        print("----WARNING: Forecast date should not be in the past")
        # warnings.warn("Forecast date should not be in the past.")

    if df["forecast_date"][0].weekday() != 2:
        print("----WARNING: Forecast date should be a Wednesday")
        # warnings.warn("Forecast date should be a Wednesday")

    print("Checking targets...")

    if not df["target"].isin(TARGETS).all():
        print("Target column can only contain " + TARGETS + ". Check spelling.")
        print("Stopping early...")
        quit()

    for target in TARGETS:

        if len(df[df["target"] == target]) != 5:
            print("Exactly 5 rows need to have target =", target)
            print("Stopping early...")
            quit()

        if (df[df["target"] == target]["horizon"] != TARGET_VALS[target]).any():
            print("Target", target, "horizons need to be (in this order):", TARGET_VALS[target])
            print("Stopping early...")
            quit()

        if (df[df["target"] == target][COLS_QUANTILES] < TARGET_PLAUS[target][0]).any(axis=None) or \
                (df[df["target"] == target][COLS_QUANTILES] > TARGET_PLAUS[target][1]).any(axis=None):
            print("----WARNING: Implausible values for", target, "detected. You may want to re-check.")
            # warnings.warn("Implausible values for "+str(target)+" detected. You may want to re-check them.")

    print("Checking quantiles...")

    for i, row in df.iterrows():
        diffs = row[COLS_QUANTILES].diff()
        if diffs[1:].isna().any():
            print("Something is wrong with your quantiles.")
            print("Stopping early...")
            quit()
        diffs[0] = 0
        if (diffs < 0).any():
            print("Predictive quantiles in row", i, "are not ordered correctly (need to be non-decreasing)")
            print("Stopping early...")
            quit()

    print("---------------------------")
    print("Looks good!")


submission_date = datetime.strftime(datetime.now(), '%Y-%m-%d')
df = pd.read_csv('/Users/franziska/Dropbox/DataPTSFC/Submissions/' + submission_date.replace('-','') + '_ChandlerBing.csv')

check_df(df)