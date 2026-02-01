import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def split_by_participant(X_features, y, meta, test_size=0.25, random_state=69):
    """
    Split data into train/test sets based on participant IDs.
    """
    pids = meta["pid"].unique()

    train_pids, test_pids = train_test_split(
        pids, test_size=test_size, random_state=random_state
    )

    train_mask = meta["pid"].isin(train_pids)
    test_mask = meta["pid"].isin(test_pids)

    X_train = X_features[train_mask.values]
    y_train = y[train_mask.values]

    X_test = X_features[test_mask.values]
    y_test = y[test_mask.values]

    print("Train participants:", meta.loc[train_mask, "pid"].unique())
    print("Test participants:", meta.loc[test_mask, "pid"].unique())

    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)

    return X_train, X_test, y_train, y_test


def train_random_forest(X_features, y, meta, test_size=0.25, random_state=69):
    """
    Train a Random Forest classifier with participant-wise split.
    """
    X_train, X_test, y_train, y_test = split_by_participant(
        X_features,
        y,
        meta,
        test_size=test_size,
        random_state=random_state)

    model = RandomForestClassifier(
        n_estimators=700,
        random_state=random_state,
        n_jobs=1)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return model, X_test, y_test, y_pred, y_proba
