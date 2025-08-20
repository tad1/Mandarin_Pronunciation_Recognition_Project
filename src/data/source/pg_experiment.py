import os

from src.path import PG_EXPERIMENT_PATH

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
ASSESMENT_CSV_NEW = os.path.join(PG_EXPERIMENT_PATH, "new_assesment.csv")
TONES_WITH_LABEL_XLS = os.path.join(PG_EXPERIMENT_PATH, "tones_with_label.xls")
AUDIO_PATH = os.path.join(PG_EXPERIMENT_PATH, "recordings/")
AUDIO_PATH_NEW = os.path.join(PG_EXPERIMENT_PATH, "new_recordings/")

import polars as pl
import polars.selectors as cs


# Note, this is fast enough; so I won't cache this
def get_pg_experiment_dataframe(extension=".ogg", verbose=False, newVersion=False):
    """_summary_
    Returns:
        _type_: `df_assessment_pronunciation`, `df_assessment_tone`
    """
    df_experiment = (
        pl.read_csv(EXPERIMENT_CSV, null_values=["NULL"])
        .select(["id", "univ", "gender", "mother"])
        .with_columns(pl.col("mother").str.to_lowercase().alias("mother"))
    )
    # Note, I'm not dropping nulls here, because there are `Invitation` results (88 of them) that don't have a gender and mother
    
    # Clean values of L1 language (fixes spelling, whitespaces, and maps multilanguage to `multi`)
    df_experiment = df_experiment.with_columns((
        pl.col("mother")
        .str.strip_chars()
        .str.to_lowercase()
        .str.replace_all(r"\s*/\s*", ", ")
        .str.replace_all(r"\s*,\s*", ", ")
        .map_elements(
            lambda x: "multi" if "," in x else x,
            return_dtype=pl.String
        )
        .map_elements(
            lambda x: {
                "belarusian": "belarussian",
                "polski": "polish"
            }.get(x, x),
            return_dtype=pl.String
        ))
    )

    df_assessment = pl.read_csv(ASSESMENT_CSV_NEW if newVersion else ASSESMENT_CSV, null_values=["NULL"])
    if not newVersion: df_assessment.drop("id_evaluator")
    excluded_rows = ["id" if newVersion else "id_student"]
    REGEX_EXPR = r"^\s*([a-zA-Z]\d+)([tp])(\d*)\s*$"
    df_assessment = (
        df_assessment.unpivot(index=excluded_rows, on=cs.exclude(excluded_rows))
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
                    # adding AUDIO_PATH removes mess with paths up in code; at a cost of removing protability of dataframe
                    #   we don't need portability, the generation code is portable
                    pl.lit(AUDIO_PATH_NEW if newVersion else AUDIO_PATH + os.path.sep),
                    pl.lit("stageII"+os.path.sep),
                    pl.col("id" if newVersion else "id_student"),
                    pl.lit(os.path.sep),
                    pl.col("word_id"),
                    pl.lit(extension),
                ]
            )
        )
        .otherwise(
            pl.concat_str(
                [
                    pl.lit(AUDIO_PATH_NEW if newVersion else AUDIO_PATH + os.path.sep),
                    pl.lit("stageI"+os.path.sep),
                    pl.col("id" if newVersion else "id_student"),
                    pl.lit(os.path.sep),
                    pl.col("word_id"),
                    pl.lit(extension),
                ]
            )
        )
        .map_elements(os.path.normpath, return_dtype=pl.Utf8)
        .alias("rec_path")
    )
    
    stage_pl_expr = (
        pl.when(pl.col("word_id").str.starts_with("q"))
        .then(pl.lit(2))
        .otherwise(pl.lit(1))
        .alias("stage")
    )

    df_assessment_tone = (
        df_assessment.filter(pl.col("type") == "t")
        .sort(["id" if newVersion else "id_student", "variable"])
        .group_by(["id" if newVersion else "id_student", "word_id"])
        .agg(pl.col("value").implode().alias("tone_assesment"))
        .with_columns(rec_pl_expr, stage_pl_expr)
    )
    
    
    df_tone_truth = pl.read_excel(
        os.path.join(PG_EXPERIMENT_PATH, "tones_with_label.xls"),
        sheet_name="tones_with_label",
    )
    
    df_tone_truth = (df_tone_truth.with_columns(
        [pl.col("word").str.extract(REGEX_EXPR, 1).alias("word_id"),]
        ).sort(["word"])
        .group_by("word_id")
        .agg(pl.col("tone").implode().alias("target_tone"))
    )

    df_assessment_tone = df_assessment_tone.join(
        df_experiment, left_on="id" if newVersion else "id_student", right_on="id"
    )
    df_assessment_tone = df_assessment_tone.join(
        df_tone_truth, left_on="word_id", right_on="word_id"
    )
    df_assessment_pronunciation = (
        df_assessment.filter(pl.col("type") == "p")
        .drop("variable", "type")
        .with_columns(rec_pl_expr, stage_pl_expr)
    )

    df_assessment_pronunciation = df_assessment_pronunciation.join(
        df_experiment, left_on="id" if newVersion else "id_student", right_on="id"
    )

    def filter_existing_files(df: pl.DataFrame) -> pl.DataFrame:
        """
        More efficient version for large datasets.
        """
        paths = df["rec_path"].to_list()
        mask = [os.path.exists(path) for path in paths]
        
        dropped = df.filter(~pl.Series(mask))
        if dropped.height > 0:
            print(f"get_pg_experiment_dataset(): WARNING, Dropped {dropped.height} rows with missing files")
            if verbose:
                print(dropped)
        
        return df.filter(pl.Series(mask))

    df_assessment_tone = filter_existing_files(df_assessment_tone)
    df_assessment_pronunciation = filter_existing_files(df_assessment_pronunciation)

    return df_assessment_pronunciation, df_assessment_tone


if __name__ == "__main__":
    # Note, use `python -m data.pg_experiment` from `src/` directory to run this code 
    a, b = get_pg_experiment_dataframe()
    print(a)
    print(b)
