import pandas as pd
import numpy as np
from transformOggToWav import oggToWav
import os

stageIpath = "../../../recordings/stageI"
valid_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]


def getStageIDataset(addTones=False, onlyPolish=True):

    # Read the demographics data and keep only Polish native speakers
    demographics = pd.read_csv("../../../experiment.csv").drop(columns="agent")
    demographics = demographics[
        ~(demographics["recordingsI"].isnull() & demographics["recordingsII"].isnull())
    ]
    if onlyPolish:
        demographics = demographics[
            demographics["mother"].str.contains("olish|olski", case=False, na=False)
        ]

    # Read the assesment data,
    # remove the id_evaluator column and any rows with id_student null
    assesmentDf = pd.read_csv("../../../assesment.csv")
    assesmentCleanDf = assesmentDf.drop(columns="id_evaluator")
    assesmentCleanDf = assesmentCleanDf[
        assesmentCleanDf.drop(columns="id_student").notnull().any(axis=1)
    ]
    # Keeping only the stage I words
    assesmentCleanDf = assesmentCleanDf.drop(
        columns=[col for col in assesmentCleanDf.columns if col.startswith("q")]
    )
    # Keeping only the assesments of Polish native speakers
    assesmentCleanDf = assesmentCleanDf[
        assesmentCleanDf["id_student"].isin(demographics["id"])
    ]

    # oggToWav()  # Transform all stageI ogg to wav, as torchaudio had issues loading the ogg
    # Store the file paths if aXp is not NaN
    rows = []
    for _, row in assesmentCleanDf.iterrows():
        student_id = int(row["id_student"])

        for i in valid_indices:
            p_col = f"a{i}p"
            t_col = f"a{i}t"

            if p_col in row and not pd.isna(row[p_col]):
                pron = row[p_col]
                tone = row[t_col] if t_col in row else np.nan
                file_path = f"{stageIpath}/{student_id}/a{i}.wav"
                if os.path.exists(file_path):
                    if not addTones:
                        rows.append({"filepath": file_path, "pron": int(pron)})
                    else:
                        rows.append(
                            {
                                "filepath": file_path,
                                "pron": int(pron),
                                "tone": int(tone) if not pd.isna(tone) else np.nan,
                            }
                        )

    final_df = pd.DataFrame(rows)
    return final_df
