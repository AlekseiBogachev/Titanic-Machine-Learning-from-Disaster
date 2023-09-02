import datetime
import joblib
import notifiers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import percentileofscore
import seaborn as sns

from sklearn.metrics import (
    get_scorer,
    roc_curve,
)

from sklearn.model_selection import (
    cross_val_predict,
    cross_val_score,
)


def make_notifier():
    """Настраивает логирование в телеграм. Возвращает функцию
    для отправки сообщений.
    """

    logger_params = joblib.load("notifier_params.pkl")

    def notifier_func(text):
        now = datetime.datetime.now()
        text_with_dt = f'{now.strftime("%d-%m-%Y %H:%M:%S")}\n{text}'

        notifier = notifiers.get_notifier(logger_params["notifier"])
        notifier.notify(
            message=text_with_dt,
            token=logger_params["token"],
            chat_id=logger_params["chat_id"],
        )

    return notifier_func


def plot_ecdf_with_target(data, target):
    sns.displot(data, stat="proportion", kind="ecdf", height=5, aspect=1)

    quantile = percentileofscore(data, target) / 100

    plt.plot([0, target, target], [quantile, quantile, 0], "-.r")
    plt.plot([target], [quantile], "or")

    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.title("ECDF результатов соревнования")

    plt.grid()

    plt.show()


def check_target_imbalance(vals):
    mean_val = vals.mean()

    print(f"Доля выживших пассажиров - {mean_val: .2%}")
    print(f"Доля погибших пассажиров - {1 - mean_val: .2%}")


def plot_corr_matrix(
    df, size=(7, 7), vmin=-1, vmax=1, method="pearson", annot=True
):
    corr = df.corr(method=method)

    mask = np.triu(np.ones_like(corr, dtype=bool))

    f, ax = plt.subplots(figsize=size)

    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        annot=annot,
        vmax=vmax,
        vmin=vmin,
        center=0.0,
        square=True,
        linewidths=1.0,
        cbar_kws={"shrink": 0.5},
        ax=ax,
    )

    ax.set_title("Матрица корреляции")

    plt.show()


def get_cv_scores(estimator, X, y, cv, n_jobs, score="accuracy"):
    """Выполняет кросс-валидацию и возвращает значения целевой метрики.
    По сути, обёртка для cross_val_score с заданными значениями параметров
    cv и n_jobs.
    """

    scores = cross_val_score(
        estimator,
        X=X,
        y=y,
        cv=cv,
        scoring=score,
        n_jobs=n_jobs,
    )

    return scores


def get_train_score(estimator, X, y, score="accuracy"):
    """Обучает модель на тренировочной выборке и на ней же оценивает
    заданную метрику.
    """

    scorer = get_scorer(score)

    return scorer(estimator.fit(X, y), X, y)


def get_cv_pred(estimator, X, y, cv, n_jobs, method="predict"):
    """Обёртка вокруг cross_val_predict, возвращающая массив с предсказаниями
    на каждом из фолдов.
    """

    y_pred = cross_val_predict(
        estimator, X, y, cv=cv, method=method, n_jobs=n_jobs
    )

    if method == "predict_proba":
        y_pred = y_pred[:, -1]

    return y_pred


def plot_roc_curve_for_random_clf():
    """Выводит на экран кривую ROC для классификатора, предсказывающего
    целевую переменную случайным образом.
    """

    fig, ax = plt.subplots(1, 1)

    ax.plot([0, 1], [0, 1], "k--", label="Random classifier")
    ax.grid()
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    ax.legend(loc="lower right")

    ax.set_title("ROC curve")

    return fig, ax


def plot_roc_curve(y_train, y_scores, label, ax, style=""):
    """Печатае кривую ROC на оси ax."""

    fpr, tpr, thresholds = roc_curve(y_train, y_scores)

    ax.plot(fpr, tpr, style, label=label)

    ax.legend(loc="lower right")

    return ax


def agg_scores(scores, label, score_names):
    """Возвращает датафрейм, содержащий среднее, медианное, минимальное и максимальное
    значения метрики, полученное с помощью кросс-валидации, и оценку метрики на
    тренировочном наборе.
    """

    res = (
        scores.query("classifier == @label")[score_names]
        .agg(["min", "median", "mean", "max"])
        .transpose()
        .rename(columns=lambda string: "_".join(["cv", string]))
    )

    cols_with_train_scores = [
        "train_" + score_name for score_name in score_names
    ]
    train_score = scores.query("classifier == @label").loc[
        0, cols_with_train_scores
    ]

    res.loc[:, "train_score"] = train_score.transpose().to_numpy()

    return res


def score_box_plot(score, metric, aspect=1, rot=0):
    g = sns.catplot(aspect=aspect)

    g = sns.scatterplot(
        data=score,
        x="classifier",
        y="train_" + metric,
        markers="X",
        label=metric + " на тренировочном наборе",
    )

    g = sns.boxplot(
        data=score, x="classifier", y=metric, color="white", showmeans=True
    )

    g.tick_params(axis="x", rotation=rot)
    g.set_xlabel("Классификатор")
    g.set_ylabel(metric)
    g.set_title(
        f"Значения метрики {metric} по результатм\nкросс-валидации для разных классификаторов"
    )
    sns.move_legend(g, "lower right")

    plt.show()


def evaluate_model(
    estimator, X, y, label, metrics, ax, cv, n_jobs, method="predict_proba"
):
    """Печатает диаграмму размаха для результатов кросс-валидации указанной метрики.
    Отмечает на диаграмме результат, полученный на тренировочном наборе данных.
    """

    cols = ["classifier"] + metrics + ["train_" + value for value in metrics]

    scores = pd.DataFrame(columns=cols)

    for metric in metrics:
        scores[metric] = get_cv_scores(
            estimator, X, y, score=metric, cv=cv, n_jobs=n_jobs
        )
        scores["train_" + metric] = get_train_score(
            estimator, X, y, score=metric
        )

    y_scores = get_cv_pred(estimator, X, y, method=method, cv=cv, n_jobs=n_jobs)
    ax = plot_roc_curve(y, y_scores, label=label, ax=ax)

    scores["classifier"] = label

    return scores, ax


def compare_models(
    classifiers,
    methods,
    labels,
    X,
    y,
    cv,
    n_jobs,
    box_plot_aspect=1.0,
    box_plot_xrot=0,
):
    all_scores = pd.DataFrame()

    fig, ax = plot_roc_curve_for_random_clf()

    for i, (classifier, method, label) in enumerate(
        zip(classifiers, methods, labels)
    ):
        print(f"{i+1}. {label}")

        metrics_list = ["accuracy", "f1", "roc_auc"]

        scores, ax = evaluate_model(
            estimator=classifier,
            X=X,
            y=y,
            metrics=metrics_list,
            label=label,
            method=method,
            ax=ax,
            cv=cv,
            n_jobs=n_jobs,
        )

        print(f"Значения метрик для {label}")
        display(agg_scores(scores, label=label, score_names=metrics_list))

        all_scores = pd.concat([all_scores, scores])

    plt.show()

    for metric in metrics_list:
        score_box_plot(
            all_scores, metric=metric, aspect=box_plot_aspect, rot=box_plot_xrot
        )
        plt.show()
