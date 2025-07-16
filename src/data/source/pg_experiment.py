import os
from path import PG_EXPERIMENT_PATH

# NOTE, this code is coupled with dataset filesystem structure:
# PG_EXPERIMENT_PATH
# ├── assesment.csv
# ├── experiment.csv
# ├── recordings
# │   ├── stageI
# │   │   └── {id}
# │   │       └── {rec_id}.ogg
# │   └── stageII
# │       └── {id}
# │           └── {rec_id}.ogg
# └── tones_with_label.xls

EXPERIMENT_CSV = os.path.join(PG_EXPERIMENT_PATH, "experiment.csv")
ASSESMENT_CSV = os.path.join(PG_EXPERIMENT_PATH, "assesment.csv")
AUDIO_PATH = os.path.join(PG_EXPERIMENT_PATH, "recordings/")

import polars as pl
import polars.selectors as cs


# Note, this is fast enought; so I won't cache this
def get_pg_experiment_dataset(extension=".ogg"):
    """_summary_
    Returns:
        _type_: `df_assesment_pronunciation`, `df_assesment_tone`
    """
    df_experiment = (
        pl.read_csv(EXPERIMENT_CSV, null_values=["NULL"])
        .select(["id", "univ", "gender", "mother"])
        .with_columns(pl.col("mother").str.to_lowercase().alias("mother"))
    )
    # Note, I'm not dropping nulls here, because there're `Invitation` results (88 of them) that don't have gender and mother

    df_assesment = pl.read_csv(ASSESMENT_CSV, null_values=["NULL"])
    df_assesment.drop("id_evaluator")
    excluded_rows = ["id_student"]
    REGEX_EXPR = r"^([a-zA-Z]\d+)([tp])(\d*)$"
    df_assesment = (
        df_assesment.unpivot(index=excluded_rows, on=cs.exclude(excluded_rows))
        .drop_nulls()
        .with_columns(
            [
                pl.col("variable").str.extract(REGEX_EXPR, 1).alias("word_id"),
                pl.col("variable").str.extract(REGEX_EXPR, 2).alias("type"),
            ]
        )
    )

    rec_pl_expr = (
        pl.when(pl.col("word_id").str.starts_with("q"))
        .then(
            pl.concat_str(
                [
                    pl.lit("stageII"+os.path.sep),
                    pl.col("id_student"),
                    pl.lit(os.path.sep),
                    pl.col("word_id"),
                    pl.lit(extension),
                ]
            )
        )
        .otherwise(
            pl.concat_str(
                [
                    pl.lit("stageI"+os.path.sep),
                    pl.col("id_student"),
                    pl.lit(os.path.sep),
                    pl.col("word_id"),
                    pl.lit(extension),
                ]
            )
        )
        .alias("rec_path")
    )

    df_assesment_tone = (
        df_assesment.filter(pl.col("type") == "t")
        .sort(["id_student", "variable"])
        .group_by(["id_student", "word_id"])
        .agg(pl.col("value").implode().alias("tone_assesment"))
        .with_columns(rec_pl_expr)
    )
    df_assesment_tone = df_assesment_tone.join(
        df_experiment, left_on="id_student", right_on="id"
    )
    df_assesment_pronunciation = (
        df_assesment.filter(pl.col("type") == "p")
        .drop("variable", "type")
        .with_columns(rec_pl_expr)
    )

    # TODO: check if recordings exists
    df_assesment_pronunciation = df_assesment_pronunciation.join(
        df_experiment, left_on="id_student", right_on="id"
    )

    return df_assesment_pronunciation, df_assesment_tone


if __name__ == "__main__":
    # Note, use `python -m data.pg_experiment` from `src/` directory to run this code 
    a, b = get_pg_experiment_dataset()
    print(a)
    print(b)
