import numpy as np
from matplotlib import pyplot as plt

import pandas as pd
from pandas import DataFrame
import more_itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.impute import SimpleImputer

from multiprocessing import Pool
from xgboost import XGBClassifier
import multiprocessing

pd.options.display.max_rows=900
pd.options.display.max_columns=900

import seaborn as sns
from IPython.display import display

from data_science_utils import dataframe as df_utils
from data_science_utils import models as model_utils
from data_science_utils import plots as plot_utils
from data_science_utils.dataframe import column as column_utils
from data_science_utils import misc as misc
from data_science_utils import preprocessing as pp_utils

import warnings
import traceback
np.set_printoptions(threshold=np.nan)
warnings.filterwarnings('ignore')
import sys, os
import missingno as msno
import random
import gc

import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import RUSBoostClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from data_science_utils.dataframe import get_specific_cols

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import regularizers
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import BaseWrapper
from keras.layers import Dense, Activation
from keras import optimizers
from keras import backend as K
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TerminateOnNaN

_EPSILON = K.epsilon()

def log_loss(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    out = -(y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred))
    return K.mean(out, axis=-1)

def root_mean_squared_error(y_true, y_pred):
    # error = K.square((1-y_true)*(y_pred - y_true)+(4*y_true)*(y_pred - y_true))
    error = K.square((y_pred - y_true))
    return K.sqrt(K.mean(error, axis=-1))

class BinaryClassifierKerasDNN:
    def __init__(self, network_config,lr=0.005,
                 n_iter=[500,500,500], columns=[],
                 scale_input=True, impute=True, raise_null=True,verbose=False,plot=True):
        self.network_config = network_config
        self.columns = columns
        assert len(columns) > 0 or prefixes is not None
        self.scale_input = scale_input
        self.scaler = StandardScaler()
        self.impute = impute
        self.imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.imp_inf = SimpleImputer(missing_values=np.inf, strategy='mean')
        self.raise_null = raise_null
        self.cols = None
        self.n_iter = n_iter
        self.verbose = verbose
        self.plot = plot
        self.lr = lr

    def check_null_(self, X):
        nans = np.isnan(X)
        infs = np.isinf(X)
        nan_summary = np.sum(np.logical_or(nans, infs))
        if nan_summary > 0:
            raise ValueError("nans/inf in frame = %s" % (nan_summary))

    def get_cols_(self, X):
        cols = list(self.columns)
        return cols

    def fit(self, X, y, sample_weight=None):
        cols = self.get_cols_(X)
        self.cols = cols
        X = X[cols]
                
        from sklearn.utils import shuffle
        X, y = shuffle(X, y)
            
            
        if self.impute:
            X = self.imp.fit_transform(X)
            X = self.imp_inf.fit_transform(X)
        if self.scale_input:
            X = self.scaler.fit_transform(X)
        if self.raise_null:
            self.check_null_(X)
            
        model = Sequential()
        i = 0
        for layer in self.network_config:
            if i==0:
                model.add(Dense(layer['neurons'],activation=layer['activation'],input_dim=X.shape[1],use_bias=True))
                if "dropout" in layer:
                    model.add(Dropout(layer["dropout"]))
                else:
                    model.add(Dropout(0.2))
            else:
                model.add(Dense(layer['neurons'],activation=layer['activation'],use_bias=True))
                if "dropout" in layer:
                    model.add(Dropout(layer["dropout"]))
                else:
                    model.add(Dropout(0.1))
            i=i+1
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        
        adam = optimizers.Adam(lr=self.lr, clipnorm=2, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)
        model.compile(optimizer=adam, loss=log_loss)
        
        
        X1,X2,X3 = np.split(X, [int(.33*len(X)), int(.66*len(X))])
        y1,y2,y3 = np.split(y, [int(.33*len(y)), int(.66*len(y))])
        
        X1,X2,X3 = pd.DataFrame(X1),pd.DataFrame(X2),pd.DataFrame(X3)
        y1,y2,y3 = pd.Series(y1),pd.Series(y2),pd.Series(y3)
        
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.0000005,epsilon=0.0001)
        reduce_lr2 = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=3, min_lr=0.0000005,epsilon=0.00001)
        terminate_on_nan = TerminateOnNaN()
        es = EarlyStopping(monitor='val_loss', min_delta=0.000001, patience=12, verbose=0,)
        es2 = EarlyStopping(monitor='val_loss', min_delta=0.000001, patience=10, verbose=0,)
        # print(K.get_value(model.optimizer.lr))
        X_train,y_train,X_val,y_val = pd.concat((X1,X2),axis=0),pd.concat((y1,y2),axis=0),X3,y3
        training_loss = []
        test_loss = []
        history = model.fit(X_train, y_train,
                        epochs=self.n_iter[0],
                        batch_size=1024,
                        shuffle=True,
                        validation_data=(X_val, y_val),
                        verbose=self.verbose,
                        callbacks=[es,terminate_on_nan,reduce_lr])
        training_loss.extend(history.history['loss'])
        test_loss.extend(history.history['val_loss'])
        # K.set_value(model.optimizer.lr, self.lr/2)
        # print(K.get_value(model.optimizer.lr))
        
        X_train,y_train,X_val,y_val = pd.concat((X2,X3),axis=0),pd.concat((y2,y3),axis=0),X1,y1
        history = model.fit(X_train, y_train,
                        epochs=self.n_iter[1],
                        batch_size=1024,
                        shuffle=True,
                        validation_data=(X_val, y_val),
                        verbose=self.verbose,
                        callbacks=[es2,terminate_on_nan,reduce_lr2])
        training_loss.extend(history.history['loss'])
        test_loss.extend(history.history['val_loss'])
        
        # print(K.get_value(model.optimizer.lr))
        K.set_value(model.optimizer.lr, self.lr/100)
        # print(K.get_value(model.optimizer.lr))
        
        X_train,y_train,X_val,y_val = pd.concat((X1,X3),axis=0),pd.concat((y1,y3),axis=0),X2,y2
        history = model.fit(X_train, y_train,
                        epochs=self.n_iter[2],
                        batch_size=4096,
                        shuffle=True,
                        validation_data=(X_val, y_val),
                        verbose=self.verbose,
                        callbacks=[es2,terminate_on_nan,reduce_lr2])
        training_loss.extend(history.history['loss'])
        test_loss.extend(history.history['val_loss'])
        # print(K.get_value(model.optimizer.lr))
        
        if self.plot:
            plt.figure(figsize=(14,10))
            plt.plot(training_loss,label="Training Loss")
            plt.plot(test_loss,label="Test Loss")
            plt.title("Training and Test Loss (Last 3 Training): %s (Last 3 Test): %s"%(training_loss[-3:],test_loss[-3:]))
            plt.ylim((min(min(training_loss),min(test_loss)),min(max(training_loss),max(test_loss))))
            plt.legend()
            plt.show()
        
        
        gc.collect()
        self.classifier = model
        return self

    def partial_fit(self, X, y):
        return self.fit(X, y)

    def predict_proba(self, X, y='ignored'):
        Inp = X
        cols = self.cols
        Inp = Inp[cols]
        if self.impute:
            Inp = self.imp.transform(Inp)
            Inp = self.imp_inf.transform(Inp)
        if self.scale_input:
            Inp = self.scaler.transform(Inp)

        if self.raise_null:
            self.check_null_(Inp)
        probas = self.classifier.predict(Inp)
        gc.collect()
        return probas
    
    def predict(self,X,y='ignored'):
        return self.predict_proba(X)>0.5

    def inverse_transform(self, X, copy=None):
        raise NotImplementedError()

    def fit_transform(self, X, y, sample_weight=None):
        self.fit(X, y, sample_weight=sample_weight)
        return self.transform(X, y)

def moving_average(a, n=5,padding=2):
    a = np.concatenate((np.array(a[0:padding]),a,np.array(a[-padding-1:-1])))
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def hampel(vals_orig, k=6, t0=2):
    '''
    vals: pandas series of values from which to remove outliers
    k: size of window (including the sample; 7 is equal to 3 on either side of value)
    '''

    #Make copy so original not edited
    vals = pd.Series(vals_orig).copy()

    #Hampel Filter
    L = 1.4826
    rolling_median = vals.rolling(window=k, center=True).median()
    MAD = lambda x: np.median(np.abs(x - np.median(x)))
    rolling_MAD = vals.rolling(window=k, center=True).apply(MAD)
    threshold = t0 * L * rolling_MAD
    difference = np.abs(vals - rolling_median)

    '''
    Perhaps a condition should be added here in the case that the threshold value
    is 0.0; maybe do not mark as outlier. MAD may be 0.0 without the original values
    being equal. See differences between MAD vs SDV.
    '''

    outlier_idx = difference > threshold
    vals[outlier_idx] = np.nan
    return vals.values

def nan_fill(a):
    a = a.copy()
    nan_idx = np.where(np.isnan(a))[0]
    a[nan_idx] = a[nan_idx-1]
    return a

def smoothen(a):
    a = nan_fill(a)
    a = hampel(a)
    while np.sum(np.isnan(a))>0:
        a = nan_fill(a)
    a = moving_average(a)
    return a


from imblearn.metrics import classification_report_imbalanced

def brier_scorer(y,y_pred,y_pred_proba):
    from sklearn.metrics import brier_score_loss
    return brier_score_loss(y,y_pred)

def balanced_accuracy_scorer(y,y_pred,y_pred_proba):
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(y,y_pred,adjusted=True)


def average_precision_scorer(y,y_pred,y_pred_proba):
    from sklearn.metrics import average_precision_score
    return average_precision_score(y,y_pred_proba)

def precision_scorer(y,y_pred,y_pred_proba):
    from sklearn.metrics import precision_score
    return precision_score(y,y_pred)

def recall_scorer(y,y_pred,y_pred_proba):
    from sklearn.metrics import recall_score
    return recall_score(y,y_pred)

def f1_scorer(y,y_pred,y_pred_proba):
    from sklearn.metrics import f1_score
    return f1_score(y,y_pred)

def accuracy_scorer(y,y_pred,y_pred_proba):
    from sklearn.metrics import accuracy_score
    return accuracy_score(y,y_pred)

def plot_classifier(X_train,y_train,y_pred_train,y_pred_train_proba,
                    X_test,y_test,y_pred,y_pred_proba,
                    plot_data=False,plot_results=True):
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import accuracy_score
    print("="*100)
    print("Classification Report:")
    print(classification_report_imbalanced(y_test,y_pred))
    # plot_utils.precision_recall_curve_binary(y_train, y_pred_train_proba)
    # plot_utils.precision_recall_curve_binary(y_test, y_pred_proba)
    if plot_data:
        plot_reduced_dim(X_train,y_train,X_test,y_test,
                         title1="Actual Training Data",
                         title2="Actual Test Data")
    
    ## 
    if plot_results:
        plot_reduced_dim(X_train,y_train==y_pred_train,X_test,y_test==y_pred,
                         title1="Correct Predictions in Train (Green), Accuracy = %.4f, AP = %.4f"%(accuracy_score(y_train,y_pred_train),average_precision_score(y_train,y_pred_train_proba)),
                         title2="Correct Predictions in Test (Green), Accuracy = %.4f, AP = %.4f"%(accuracy_score(y_test,y_pred),average_precision_score(y_test, y_pred_proba)),
                        palette={True:"g",False:"r"})
    plt.show();

def plot_decision_boundary_2d(X,classifier):
    if X.shape[1]!=2:
        raise ValueError("Decision Boundary Plotting only works in 2D")
    xmin,ymin = X.min() - 1
    xmax,ymax = X.max() + 1
    x_values = np.arange(xmin,xmax,(xmax-xmin)/100)
    y_values = np.arange(ymin,ymax,(ymax-ymin)/100)
    x_y_values = []
    for x_value in x_values:
        for y_value in y_values:
            x_y_values.append([x_value,y_value])
    X = pd.DataFrame(x_y_values,columns=X.columns)
    y = classifier.predict(X)
    y = [True if yi else False for yi in y]
    plt.figure(figsize=(8,8))
    X = X.values
    sns.scatterplot(x=X[:,0],y=X[:,1],hue=y)
    plt.title("Decision Boundary")
    plt.xlabel("Dim 1 (X)")
    plt.xlabel("Dim 2 (Y)")
    plt.show()



    

def run_classifier(classifier,scorers,X_train,y_train,X_test,y_test,sample_weight=None,
                   plot=False,plot_data=True,plot_results=True):
    verify_pandas(X_train,y_train)
    verify_pandas(X_test,y_test)
    
    if sample_weight is not None:
    	classifier.fit(X_train,y_train.values,sample_weight=sample_weight);
    else:
    	classifier.fit(X_train,y_train.values);
    y_pred = classifier.predict(X_test)
    y_pred_train = classifier.predict(X_train)
    
    y_pred = y_pred.squeeze()
    y_pred_train = y_pred_train.squeeze()
    
    
    y_pred_proba = classifier.predict_proba(X_test)
    y_pred_train_proba = classifier.predict_proba(X_train)
    if len(y_pred_proba.shape)==2 and y_pred_proba.shape[1]==2:
        y_pred_proba = y_pred_proba[:,1]
        y_pred_train_proba = y_pred_train_proba[:,1]
        
    if plot:
        plot_classifier(X_train,y_train,y_pred_train,y_pred_train_proba,
                    X_test,y_test,y_pred,y_pred_proba,plot_data=plot_data,plot_results=plot_results)
        
    score_dict = {}
    for scorer in scorers:
        score_dict[scorer.__name__] = [scorer(y_train,y_pred_train,y_pred_train_proba),scorer(y_test,y_pred,y_pred_proba)]
        
    return score_dict

    

from imblearn.datasets import make_imbalance
from collections import Counter

def verify_pandas(X,y):
    if type(X)!=pd.DataFrame:
        raise ValueError("Only Pandas Dataframe supported for X")
    if type(y)!=pd.Series:
        raise ValueError("Only Pandas Series Supported for y")
    y = pd.Series([True if yi else False for yi in y],index=y.index)
    return X,y

def plot_pie(y,ax=None):
    target_stats = Counter(y)
    labels = sorted(list(target_stats.keys()))
    sizes = list([target_stats[label] for label in labels])
    explode = tuple([0.01] * len(target_stats))

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return '{p:.2f}%  ({v:d})'.format(p=pct, v=val)
        return my_autopct

    if ax is None:
        fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, shadow=True,
           autopct=make_autopct(sizes))
    ax.axis('equal')

def generate_data(data_size = 10000, # before imbalance, approx
                x_min_max = 35,
                shape_param_1 = 5, # try values like 3,5,7,[10] and see how graph changes
                shape_param_2 = 10, # try 5,10,15,20
                shape_param_3 = 1.5, # try between 1-5
                shape_param_4 = 100, # try 50,100,150,200
                add_shapes_for_extra_complexity = True,
                plot=False):
    
    min_x,max_x = -x_min_max,x_min_max
    interval = (max_x-min_x)/data_size
    x = np.arange(min_x,max_x,interval)
    y_line = np.clip((x**3)/shape_param_4 - shape_param_1*x + shape_param_2*np.sin(x/shape_param_3),-80,80)
    
    y = np.random.uniform(min(y_line)-10,max(y_line)+10,len(x))
    target = y>y_line
    if add_shapes_for_extra_complexity:
        circle1 = np.where((x-10)**2 + (y-33)**2 < 81)
        target[circle1] = ~(np.mean(target[circle1])>0.5)
        
        circle2 = np.where((x+10)**2 + (y+47)**2 < 64)
        target[circle2] = ~(np.mean(target[circle2])>0.5)
        circle3 = np.where((x-25)**2 + (y+70)**2 < 100)
        target[circle3] = ~(np.mean(target[circle3])>0.5)
        
        circle4 = np.where((x+20)**2 + (y-70)**2 < 49)
        target[circle4] = ~(np.mean(target[circle4])>0.5)
        
    X = pd.DataFrame({"x":x,"y":y})
    
    if plot:
    
        plt.figure(figsize=(14,5));
        sns.scatterplot(x,y,hue=target);
        plt.title("Data: Ratio +/- = %.3f"%(np.sum(target)/np.sum(~target)));
        plt.show();
    
    return X,pd.Series(target)

def plot_imbalance(y,y_res):
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(12,6))
    # pie chart
    
    plot_pie(y,ax1)
    plot_pie(y_res,ax2)
    ax1.set_title("Target before Imbalance")
    ax2.set_title("Target after Imbalance")
    
    plt.show()
    
def imbalance(X,y,pos_neg_frac, plot=False):
    X,y = verify_pandas(X,y)
    pos_examples = np.sum(y)
    negative_examples = np.sum([False if yew else True for yew in y])
    pos_neg_frac_now = pos_examples/negative_examples
    if pos_neg_frac_now<=pos_neg_frac:
        # print("WARN: Current Pos/Neg example Ratio = %.3f < Expected Pos/Neg Ratio = %.3f. Cannot Undersample!"%(pos_neg_frac_now,pos_neg_frac))
        return X,y
    
    num_pos_examples_needed = int(pos_neg_frac*negative_examples)
    X_res,y_res = make_imbalance(X, y,sampling_strategy={0: negative_examples, 1: num_pos_examples_needed})
    X_res = pd.DataFrame(X_res)
    y_res=pd.Series(y_res)
    if plot:
        if X.shape[1] == 2:
            f, (ax1,ax2) = plt.subplots(nrows=1, ncols=2,figsize=(18,9))
            sns.scatterplot(X.values[:,0],X.values[:,1],hue=y,ax=ax1);
            ax1.set_title("Before Imbalance");
            
            sns.scatterplot(X_res.values[:,0],X_res.values[:,1],hue=y_res,ax=ax2);
            ax2.set_title("After Imbalance");
            plt.show();

        print("="*100)
        plot_imbalance(y,y_res)
        
    
    return X_res,y_res

def plot_reduced_dim(X,y,X_res,y_res,title1=None,title2=None,palette=None):
    from sklearn.manifold import TSNE
    X,y = verify_pandas(X,y)
    
    if X.shape[1]>2:
        X_embedded = TSNE(n_components=2,random_state=47,n_iter=400,angle=0.6).fit_transform(X)
        X_embedded_res = TSNE(n_components=2,random_state=47,n_iter=400,angle=0.6).fit_transform(X_res)
    else:
        X_embedded = X.values
        X_embedded_res = X_res.values
    
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(16,6))
    #sns.lineplot(x,y_line,ax=ax1);
    sns.scatterplot(X_embedded[:,0],X_embedded[:,1],hue=y,ax=ax1,palette=palette);
    if title1 is None:
        title1 = "Before, Ratio +/- = %.3f"%(np.sum(y)/np.sum(~y))
    if title2 is None:
        title2 = "After, Ratio +/- = %.3f"%(np.sum(y_res)/np.sum(~y_res))
    ax1.set_title(title1);

    #sns.lineplot(x,y_line,ax=ax2);
    sns.scatterplot(X_embedded_res[:,0],X_embedded_res[:,1],hue=y_res,ax=ax2,palette=palette);
    ax2.set_title(title2);
    plt.show();
    
def visualize_overall_train_test_data(X,y,X_train,y_train,X_test,y_test):
    from sklearn.manifold import TSNE
    if X.shape[1]>2:
        X_embedded = TSNE(n_components=2,random_state=47,n_iter=400,angle=0.6).fit_transform(X)
        X_embedded_train = TSNE(n_components=2,random_state=47,n_iter=400,angle=0.6).fit_transform(X_train)
        X_embedded_test = TSNE(n_components=2,random_state=47,n_iter=400,angle=0.6).fit_transform(X_test)
    else:
        X_embedded = X.values
        X_embedded_train = X_train.values
        X_embedded_test = X_test.values
    
    f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3,figsize=(18,5))
    sns.scatterplot(X_embedded[:,0],X_embedded[:,1],hue=y,ax=ax1);
    ax1.set_title("Overall Data");
    

    sns.scatterplot(X_embedded_train[:,0],X_embedded_train[:,1],hue=y_train,ax=ax2);
    ax2.set_title("Training Data");
    
    sns.scatterplot(X_embedded_test[:,0],X_embedded_test[:,1],hue=y_test,ax=ax3);
    ax3.set_title("Test Data");
    plt.show();

def visualize_3d(X,y,algorithm="tsne",title="Data in 3D"):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    if algorithm=="tsne":
        reducer = TSNE(n_components=3,random_state=47,n_iter=400,angle=0.6)
    elif algorithm=="pca":
        reducer = PCA(n_components=3,random_state=47)
    else:
        raise ValueError("Unsupported dimensionality reduction algorithm given.")
    
    if X.shape[1]>3:
        X = reducer.fit_transform(X)
    else:
        if type(X)==pd.DataFrame:
        	X=X.values
    
    marker_shapes = ["circle","diamond", "circle-open", "square",  "diamond-open", "cross","square-open",]
    traces = []
    for hue in np.unique(y):
        X1 = X[y==hue]

        trace = go.Scatter3d(
            x=X1[:,0],
            y=X1[:,1],
            z=X1[:,2],
            mode='markers',
            name = str(hue),
            marker=dict(
                size=12,
                symbol=marker_shapes.pop(),
                line=dict(
                    width=int(np.random.randint(3,10)/10)
                ),
                opacity=int(np.random.randint(6,10)/10)
            )
        )
        traces.append(trace)


    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(
                title='Dim 1'),
            yaxis=dict(
                title='Dim 2'),
            zaxis=dict(
                title='Dim 3'), ),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=traces, layout=layout)
    iplot(fig)

    
def visualize_2d(X,y,algorithm="tsne",title="Data in 2D",figsize=(8,8)):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    if algorithm=="tsne":
        reducer = TSNE(n_components=2,random_state=47,n_iter=400,angle=0.6)
    elif algorithm=="pca":
        reducer = PCA(n_components=2,random_state=47)
    else:
        raise ValueError("Unsupported dimensionality reduction algorithm given.")
    if X.shape[1]>2:
        X = reducer.fit_transform(X)
    else:
        if type(X)==pd.DataFrame:
        	X=X.values
    f, (ax1) = plt.subplots(nrows=1, ncols=1,figsize=figsize)
    sns.scatterplot(X[:,0],X[:,1],hue=y,ax=ax1);
    ax1.set_title(title);
    plt.show();

def add_noise(X,y,noise_level = 0.2,noise_type = "randomize_labels",
              classifier=XGBClassifier(max_depth=6),plot=False):
    # plotting in 2 component with tsne, plot noised and non noised
    # Show possible max accuracy, precision and recall.
    # for boundary noise we use a classifier to find the boundary
    # randomize_labels|near_decision_boundary_labels|data
    import bisect
    
    X,y = verify_pandas(X,y)
    
    
    total_noised_samples_needed = int(len(X)*noise_level)
    noised_indexes = []
    X_res,y_res = X.copy(),y.copy()
    if noise_type is None or noise_level<0.01:
        pass
    elif noise_type=="data":
        for col in X.columns:
            mean = X[col].mean()
            std = X[col].std()*noise_level
            noise = np.random.normal(mean, std, [X.shape[0]])
            X_res[col] = X[col] + std
    elif noise_type=="randomize_labels":
        noised_indexes = np.random.randint(0,len(y),int(noise_level*len(y)))
    elif noise_type=="near_decision_boundary_labels":
        classifier.fit(X,y)
        probas = classifier.predict_proba(X)[:,1]
        sorted_probas = sorted(probas)
        numbers = np.arange(1,len(sorted_probas))
        inverted_numbers = reversed(numbers)
        proba_count_dict = {}
        min_error = len(X)
        final_width = 0.5
        for width in np.arange(0.001,0.499,0.001):
            lower = 0.5 - width
            upper = 0.5 + width
            lidx = bisect.bisect_left(sorted_probas,lower)
            uidx = bisect.bisect_right(sorted_probas,upper)
            length = uidx - lidx
            error = np.abs(3*total_noised_samples_needed-length)
            if error > min_error:
                final_width = width
            min_error = min(min_error,error)
        boundary_indexes = np.where(np.abs(probas-0.5)<final_width)[0]
        index_selector = np.random.randint(0,len(boundary_indexes),total_noised_samples_needed)
        noised_indexes = boundary_indexes[index_selector]
        
        
        
    else:
        raise ValueError("Unknown Noise Type")

    X_res,y_res = pd.DataFrame(X_res,columns=X.columns,index=X.index),pd.Series(y_res,index=y.index)
    
    
    y_res[noised_indexes] = ~y_res[noised_indexes]
    if plot:
        print("="*100)
        if noise_type in ["randomize_labels","near_decision_boundary_labels"]:
            actual_noise_frac = len(noised_indexes)/len(X)
            sys.stdout.write("\n\rNumber of Target Labels noised = %s, Noise fraction=%.3f"%(len(noised_indexes),actual_noise_frac))
        print("\n\nBest Possible Metrics After Introducing Noise:")
        print(classification_report(y_res,y))
        plot_reduced_dim(X,y,X_res,y_res,
                        title1="Before Noise",title2="After Noise")
    
    return X_res,y_res


def oversample(X,y,method="smote",pos_neg_frac=0.5,plot=False):
    from imblearn.over_sampling import SMOTE
    from imblearn.over_sampling import ADASYN
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.combine import SMOTEENN
    from imblearn.combine import SMOTETomek
    sampler = None
    X,y = verify_pandas(X,y)
    pos_neg_frac_now = np.sum(y)/np.sum(~y)
    if pos_neg_frac<=pos_neg_frac_now:
        print("Oversampling isn't need since Pos/Neg current = %.3f is greater than passed Pos/Neg ratio = %.3f"%(pos_neg_frac_now,pos_neg_frac))
        return X,y
    cols = X.columns
    if method is None:
        return X,y
    elif method=="smote":
        sampler = SMOTE(sampling_strategy=pos_neg_frac)
    elif method=="adasyn":
        sampler = ADASYN(sampling_strategy=pos_neg_frac)
    elif method=="randomoversampler":
        sampler = RandomOverSampler(sampling_strategy=pos_neg_frac)
    elif method=="smoteenn":
        sampler = SMOTEENN(sampling_strategy=pos_neg_frac)
    elif method=="smotetomek":
        sampler = SMOTETomek(sampling_strategy=pos_neg_frac)
    else:
        raise ValueError("Over sampler not found")
    
    X_res,y_res = X.copy(deep=True),y.copy(deep=True)
    X_res,y_res = sampler.fit_resample(X_res,y_res)
    X_res = pd.DataFrame(X_res,columns=cols)
        
    y_res=pd.Series(y_res)
        
    if plot:
        print("="*100+"\nPlotting Imbalance and Noise after Oversampling")
        plot_imbalance(y,y_res)
        plot_reduced_dim(X,y,X_res,y_res,title1="Before Oversampling",title2="After Oversampling")
    return X_res,y_res

def split(X,y,test_size=0.3,random_state=42):
    verify_pandas(X,y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    X_train, X_test = pd.DataFrame(X_train,columns=X.columns), pd.DataFrame(X_test,columns=X.columns)
    y_train, y_test =  pd.Series(y_train), pd.Series(y_test)
    return X_train, X_test, y_train, y_test


