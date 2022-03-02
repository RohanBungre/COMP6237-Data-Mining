import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


def dcg(y, pred):
    rel_i = np.where(np.asarray(list(map(lambda y_hat: y_hat == y, pred))), 1, 0)

    return np.sum((2 ** rel_i - 1) / np.log2(np.arange(1, len(rel_i) + 1) + 1))


def ndcg(y, pred):
    return dcg(y, pred) / dcg(y, [y])


# TODO: parallelize
def custom_cross_validate(pipeline, X, y, cv=5):

    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=10061999)

    ndcg_scores = []
    f1_scores = []

    for train_index, test_index in kf.split(X, y):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        pipeline.fit(X_train, y_train)

        preds = pipeline.predict_proba(X_test)
        preds_labels = pipeline.predict(X_test)

        scores = []
        for y_test_target, pred in zip(y_test, preds):
            pred = [
                label
                for label, __ in sorted(
                    list(zip(pipeline.classes_, pred)), key=lambda x: x[1], reverse=True
                )[:5]
            ]
            
            scores.append(ndcg(y_test_target, pred))
        
        
        ndcg_scores.append(np.mean(scores))
        f1_scores.append(f1_score(y_test, preds_labels, average='micro'))


    return ndcg_scores, f1_scores
