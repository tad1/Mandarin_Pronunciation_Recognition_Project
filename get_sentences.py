import polars as pl


def get_info(sentence_ids):
    """Get the info for a given sentence_id"""
    df_csv = pl.read_csv("../../../Data/cv-corpus-20.0-2024-12-06/zh-CN/validated.tsv", separator="\t")
    df_csv = df_csv.filter(pl.col("sentence_id").is_in(sentence_ids))
    df_csv = df_csv.select(["sentence_id", "path", "sentence"])
    return df_csv


