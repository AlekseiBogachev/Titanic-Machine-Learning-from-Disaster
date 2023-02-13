import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (#accuracy_score,
                             #f1_score,
                             get_scorer, 
                             roc_curve,
                            )
from sklearn.model_selection import (cross_val_predict, 
                                     cross_val_score,
                                    )


def plot_corr_matrix(df, size=(7, 7), vmin=-1, vmax=1, method='pearson', annot=True):
    corr = df.corr(method=method)
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    f, ax = plt.subplots(figsize=size)
    
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr,
                mask=mask,
                cmap=cmap,
                annot=annot,
                vmax=vmax,
                vmin=vmin,
                center=0.0,
                square=True,
                linewidths=1.0,
                cbar_kws={'shrink': 0.5},
                ax=ax
               )
    
    ax.set_title('Матрица корреляции')
    
    plt.show()
    
    
def print_mi_scores(df, y_train, n_neighbors, random_state=None):
    
    fig, ax = plt.subplots()
    
    mi_scores = (pd
                 .DataFrame(mutual_info_classif(df, 
                                                y_train, 
                                                n_neighbors=n_neighbors, 
                                                random_state=random_state), 
                            columns=['mutual_info'], 
                            index=df.columns
                           )
                 .sort_values(by='mutual_info', ascending=False)
    )

#     print(f'n_neighbors={n_neighbors}')
#     display(mi_scores)

    (mi_scores
     .round(3)
     .sort_values(by='mutual_info', ascending=True)
     .plot(kind='barh', grid=False, title=f'Mutual information\nn_neighbors={n_neighbors}', ax=ax)
    )

    ax.bar_label(ax.containers[0])
    ax.set_xlim([0, 0.25])
    ax.legend(loc='lower right')
    plt.show()
    
    return mi_scores


def print_score(estimator, features, target, score='accuracy', cv=5, n_jobs=-1):
    '''Вычисляет среднее, минимальное и максимальное значения метрики,
    полученное с помощью кросс-валидации.
    '''
    
    scores = cross_val_score(estimator,
                             X=features,
                             y=target,
                             cv=cv,
                             scoring=score,
                             n_jobs=n_jobs,
                            )
    
    res = pd.DataFrame(scores, columns=[score]).agg(['mean', 'median', 'min', 'max']).transpose()
    
    scorer = get_scorer(score)
    res.loc[score, 'results_on_train_set'] = scorer(estimator.fit(features, target), features, target)
    
    return res


def plot_roc_curve_for_random_clf():
    '''Выводит на экран кривую ROC для классификатора, предсказывающего
       целевую переменную случайным образом.
    '''
    
    fig, ax = plt.subplots(1, 1)
    
    ax.plot([0,1], [0,1], 'k--', label='Random classifier')
    ax.grid()
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    ax.legend(loc='lower right')

    ax.set_title('ROC curve')
    
    return fig, ax


def plot_roc_curve(y_train, y_scores, ax, style='', label=None):
    '''Печатае кривую ROC на оси ax.'''
    
    fpr, tpr, thresholds = roc_curve(y_train, y_scores)
    
    ax.plot(fpr, tpr, style, label=label)
    
    return ax


def evaluate_model(model, X, y, label, ax=None, method='predict_proba', cv=5, n_jobs=-1):
    '''Выводит на экран сводку о производительности модели, состоящую из метрик
    accuracy, f1, roc_auc и кривую ROC. Также возвращат значения метрик в датафрейме df,
    печатает кривую ROC на оси ax.
    '''
    
    df = pd.DataFrame(columns= ['mean', 'median', 'min', 'max'])
    
    for metric in ['accuracy', 'f1', 'roc_auc']:
        df = pd.concat([df, print_score(model, X, y, score=metric, cv=cv, n_jobs=n_jobs)])
        
    print(f'Значения метрик для {label}')
    
    display(df)
        
    y_scores = cross_val_predict(model, X, y, cv=cv, method=method, n_jobs=n_jobs)
    if method == 'predict_proba':
        y_scores = y_scores[:, -1]
    
    if ax is None:
        fig, ax = plot_roc_curve_for_random_clf()        
        ax = plot_roc_curve(y, y_scores, label=label, ax=ax)
        plt.show()
        
        return df
    
    else:
        ax = plot_roc_curve(y, y_scores, label=label, ax=ax)
        
        return df, ax
    
    
def plot_accuracy(df, aspect=1, rot=0):
    g = sns.catplot(aspect=aspect)

    g = sns.scatterplot(data=df, 
                      x='classifier', 
                      y='accuracy_on_train_set', 
                      markers='X', 
                      label='Accuracy на тренировочном наборе'
                     )

    g = sns.boxplot(data=df, x='classifier', y='accuracy_on_CV', color='white', showmeans=True)

    g.set_ylim([0.6, 1])
    g.tick_params(axis='x', rotation=rot)
    g.set_xlabel('Классификатор')
    g.set_ylabel('Accuracy')
    g.set_title('Значения метрики accuracy по результатм\nкросс-валидации для разных классификаторов')
    sns.move_legend(g, 'lower right')

    plt.show()
    
    
def get_performance_of_one_model(model, X, y, label, scoring='accuracy', cv=5, n_jobs=-1, method='predict_proba'):
    
    fig, ax = plot_roc_curve_for_random_clf()
    
    scores = pd.DataFrame(columns=['classifier', 'accuracy_on_CV', 'accuracy_on_train_set'])
    
    print(label)
    
    df, ax = evaluate_model(model=model, X=X, y=y, method=method, label=label, cv=cv, ax=ax)

    score = pd.DataFrame(cross_val_score(estimator=model, 
                                         X=X, 
                                         y=y, 
                                         scoring=scoring,
                                         cv=cv,
                                         n_jobs=n_jobs,
                                        ),
                         columns=['accuracy_on_CV']
                        )
    score['classifier'] = label
    score['accuracy_on_train_set'] = df.loc['accuracy', 'results_on_train_set']
    scores = scores.append(score)
    
    print() 
    
    plt.legend()
    plt.show()

    print()

    plot_accuracy(scores)