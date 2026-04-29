import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    return mo, pa, pd, pq


@app.cell
def _(pa, pq):
    PARQUET_PATH = "students/emmazhang/sampled_user_month_traj.parquet"
    table = pq.read_table(PARQUET_PATH)

    new_cols = []
    for col in table.columns:
        if pa.types.is_dictionary(col.type):
            new_cols.append(col.cast(pa.string()))
        else:
            new_cols.append(col)

    table = pa.table(new_cols, names=table.column_names)

    PARQUET_PATH = "students/emmazhang/sampled_user_month_traj.parquet"
    table = pq.read_table(PARQUET_PATH)

    fixed_cols = []
    for col in table.columns:
        if pa.types.is_dictionary(col.type):
            fixed_cols.append(col.cast(pa.string()))
        else:
            fixed_cols.append(col)

    table = pa.table(fixed_cols, names=table.column_names)
    return (table,)


@app.cell
def _(table):
    df = table.to_pandas()
    df
    return (df,)


@app.cell
def _(df):
    ## Columns in dataset
    sorted(list(df.columns))
    return


@app.cell
def _(df):
    work = df[
        [
            "participantId",
            "calendarMonth",
            "month_role",
            "notesWritten",
            "notesRated",
            "notesRequested",
            "hits",
            "correctHelpfuls",
            "numRequestsResultingInCrh",
        ]
    ].copy()

    work 
    return (work,)


@app.cell
def _(work):
    sorted(work["month_role"].dropna().unique().tolist())
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## **Q1: What proportion of notes, ratings, and requests come from users in each activity level?**
    """)
    return


@app.cell
def _(work):
    q1 = (
        work.groupby("month_role", dropna=False)[
            ["notesWritten", "notesRated", "notesRequested"]
        ]
        .sum()
        .reset_index()
    )

    q1["prop_notes"] = (q1["notesWritten"] / q1["notesWritten"].sum() * 100).round(2)
    q1["prop_ratings"] = (q1["notesRated"] / q1["notesRated"].sum() * 100).round(2)
    q1["prop_requests"] = (q1["notesRequested"] / q1["notesRequested"].sum() * 100).round(2)

    q1 = q1.sort_values("prop_notes", ascending=False).reset_index(drop=True)
    q1
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Notes
    Note production is heavily concentrated among writer roles.
    ## Ratings
    There are also some ratings coming from writer groups, especially single_digit_writer and single_note_writer, which suggests users classified as writers can still rate notes too.
    ## Requests
    Request activity is strongly concentrated among requestor roles.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## **Q2: What proportion of helpful notes come from users in each activity level? What proportion of helpful ratings on helpful notes come from users in each activity level? What proportion of requests that ended in helpful notes came from users in each activity level?**
    """)
    return


@app.cell
def _(work):
    q2 = (
        work.groupby("month_role", dropna=False)[
            ["hits", "correctHelpfuls", "numRequestsResultingInCrh"]
        ]
        .sum()
        .reset_index()
    )

    q2["prop_helpful_notes"] = (q2["hits"] / q2["hits"].sum() * 100).round(2)
    q2["prop_helpful_ratings"] = (
        q2["correctHelpfuls"] / q2["correctHelpfuls"].sum() * 100
    ).round(2)
    q2["prop_helpful_requests"] = (
        q2["numRequestsResultingInCrh"] / q2["numRequestsResultingInCrh"].sum() * 100
    ).round(2)

    q2 = q2.sort_values("prop_helpful_notes", ascending=False).reset_index(drop=True)
    q2
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Helpful notes
    - single_digit_writer - 38.05%
    - single_note_writer - 23.08%
    - double_digit_writer - 21.91%
    - 4_digit_writer - 14.16%

    This shows that the most helpful notes come from writer roles.
    - single_digit_writer contributes the largest share of helpful notes, even more than double_digit_writer or triple_digit_writer, which suggests the most useful note-writing is not only coming from the most extreme high-volume writers.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Helpful ratings on helpful notes
    - double_digit_rater - 39.03%
    - single_digit_rater - 24.09%
    - single_note_rater - 7.26%
    - triple_digit_rater - 7.73%

    This shows that rater roles, especially double_digit_rater, account for a very large share of ratings that correctly identify helpful notes.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Requests that ended in helpful notes
    - single_post_requestor - 36.47%
    - single_digit_requestor - 33.85%
    - double_digit_requestor - 13.53%

    This shows that the majority of requests that eventually lead to a helpful note come from requestor roles, especially the lower-volume requestor groups.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## **Q3: How long have users who've entered the program at different times, and who entered with different activity levels, stayed before attrition?**
    """)
    return


@app.cell
def _(pd, work):
    work_time = work.copy()
    work_time["calendarMonth"] = pd.to_datetime(work_time["calendarMonth"], errors="coerce")
    work_time
    return (work_time,)


@app.cell
def _(work_time):
    # per user
    first_rows = (
        work_time.sort_values(["participantId", "calendarMonth"])
        .groupby("participantId", as_index=False)
        .first()[["participantId", "calendarMonth", "month_role"]]
        .rename(
            columns={
                "calendarMonth": "entry_month",
                "month_role": "entry_month_role",
            }
        )
    )

    last_rows = (
        work_time.groupby("participantId", as_index=False)["calendarMonth"]
        .max()
        .rename(columns={"calendarMonth": "last_month"})
    )

    user_attrition = first_rows.merge(last_rows, on="participantId", how="left")

    user_attrition["months_stayed"] = (
        (user_attrition["last_month"].dt.to_period("M") - user_attrition["entry_month"].dt.to_period("M"))
        .apply(lambda x: x.n)
        + 1
    )

    user_attrition
    return (user_attrition,)


@app.cell
def _(user_attrition):
    # by activity level
    attrition_by_role = (
        user_attrition.groupby("entry_month_role", dropna=False)
        .agg(
            users=("participantId", "count"),
            mean_months_stayed=("months_stayed", "mean"),
            median_months_stayed=("months_stayed", "median"),
            min_months_stayed=("months_stayed", "min"),
            max_months_stayed=("months_stayed", "max"),
        )
        .reset_index()
        .sort_values("median_months_stayed", ascending=False)
    )

    attrition_by_role
    return


@app.cell
def _(user_attrition):
    # by month
    attrition_by_entry_month = (
        user_attrition.groupby("entry_month", dropna=False)
        .agg(
            users=("participantId", "count"),
            mean_months_stayed=("months_stayed", "mean"),
            median_months_stayed=("months_stayed", "median"),
            min_months_stayed=("months_stayed", "min"),
            max_months_stayed=("months_stayed", "max"),
        )
        .reset_index()
        .sort_values("entry_month")
    )

    attrition_by_entry_month
    return


@app.cell
def _(user_attrition):
    attrition_by_month_and_role = (
        user_attrition.groupby(["entry_month", "entry_month_role"], dropna=False)
        .agg(
            users=("participantId", "count"),
            mean_months_stayed=("months_stayed", "mean"),
            median_months_stayed=("months_stayed", "median"),
        )
        .reset_index()
        .sort_values(["entry_month", "entry_month_role"])
    )

    attrition_by_month_and_role
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
