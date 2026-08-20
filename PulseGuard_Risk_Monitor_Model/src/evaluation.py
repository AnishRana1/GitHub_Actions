def make_summary(df):
    total = len(df)
    predicted = int(df["ml_predicted_risk"].fillna(False).sum())
    return {
        "points": total,
        "predicted_risk_events": predicted,
        "predicted_risk_rate": predicted / total if total else 0,
    }
