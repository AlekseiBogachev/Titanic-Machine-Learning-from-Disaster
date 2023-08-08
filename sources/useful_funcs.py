import datetime
import joblib
import notifiers
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
import seaborn as sns



def make_notifier():
    '''Настраивает логирование в телеграм. Возвращает функцию
    для отправки сообщений.
    '''
    
    logger_params = joblib.load('notifier_params.pkl')
    
    def notifier_func(text):
        now = datetime.datetime.now()
        text_with_dt = f'{now.strftime("%d-%m-%Y %H:%M:%S")}\n{text}'
        
        notifier = notifiers.get_notifier(logger_params['notifier'])
        notifier.notify(message=text_with_dt,
                        token=logger_params['token'],
                        chat_id=logger_params['chat_id'],
                       )
        
    return notifier_func



def plot_ecdf_with_target(data, target):
    sns.displot(data, stat='proportion', kind='ecdf', height=5, aspect=1)
    
    quantile = percentileofscore(data, target) / 100
    
    plt.plot([0, target, target], [quantile, quantile, 0], '-.r')
    plt.plot([target], [quantile], 'or')
    
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    
    plt.title('ECDF результатов соревнования')
    
    plt.grid()
    
    plt.show()
    
    

def check_target_imbalance(vals):
    mean_val = vals.mean()
    
    print(f'Доля выживших пассажиров - {mean_val: .2%}')
    print(f'Доля погибших пассажиров - {1 - mean_val: .2%}')
    
    
    
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