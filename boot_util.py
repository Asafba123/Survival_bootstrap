import numpy as np
import pandas as pd
import torch
from torch import nn
import torchtuples as tt
import functools

from pycox import models
from pycox.preprocessing.label_transforms import LabTransCoxTime
from pycox.models.cox_time import MLPVanillaCoxTime

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from pycox.simulations import *
import scipy

class SimStudyNonLinearPH(SimStudyLinearPH):
    '''Survival simulations study for non-linear prop. hazard model
        h(t | x) = h0 exp[g(x)], where g(x) is non-linear.
    Parameters:
        h0: Is baseline constant.
        right_c: Time for right censoring.
    '''
    @staticmethod
    def g(covs):
        x = covs
        x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
        beta = 2/3
        linear = SimStudyLinearPH.g(x)
        nonlinear =  beta * (x0**2 + x2**2 + x0*x1 + x0*x2 + x1*x2)
        return linear + nonlinear


class SimStudyNonLinearNonPH_smooth(SimStudyNonLinearPH):
    '''Survival simulations study for non-linear non-prop. hazard model.
        h(t | x) = h0 * exp[g(t, x)], 
        with constant h_0, and g(t, x) = a(x) + b(x)*t.
        Cumulative hazard:
        H(t | x) = h0 / b(x) * exp[a(x)] * (exp[b(x) * t] - 1)
        Inverse:
        H^{-1}(v, x) = 1/b(x) log{1 +  v * b(x) / h0 exp[-a(x)]}
    Parameters:
        h0: Is baseline constant.
        right_c: Time for right censoring.
    '''
    def __init__(self, h0=0.02, right_c=30., c0=30., surv_grid=None):
        super().__init__(h0, right_c, c0, surv_grid)

    @staticmethod
    def a(x):
        _, _, x2 = x[:, 0], x[:, 1], x[:, 2]
        return x2 + SimStudyNonLinearPH.g(x) 
    
    @staticmethod
    def b(x):
        x0, x1, _ = x[:, 0], x[:, 1], x[:, 2]
        return 0.1 + (0.2 * (x0 + x1) + 0.5 * x0 * x1)**2

    @staticmethod
    def g(t, covs):
        x = covs
        return SimStudyNonLinearNonPH.a(x) + SimStudyNonLinearNonPH.b(x) * t
    
    def inv_cum_hazard(self, v, covs):
        x = covs
        return 1 / self.b(x) * np.log(1 + v * self.b(x) / self.h0 * np.exp(-self.a(x)))
    
    def cum_hazard(self, t, covs):
        x = covs
        return self.h0 / self.b(x) * np.exp(self.a(x)) * (np.exp(self.b(x)*t) - 1)
    
class SimStudyDeep1(SimStudyLinearPH):
    '''Survival simulations study for linear prop. hazard model
        h(t | x) = h0 exp[g(x)], where g(x) is linear.

    Parameters:
        h0: Is baseline constant.
        right_c: Time for right censoring.
    '''
    def __init__(self, h0=0.1, right_c=40, c0=28., surv_grid=None):
        print(right_c)
        super().__init__(h0, right_c, c0, surv_grid)

    @staticmethod
    def sample_covs(n):
        U = np.zeros(5)
        SIGMA = np.zeros((5,5)) + 0.5
        np.fill_diagonal(SIGMA,1)
        x = np.random.multivariate_normal(U,SIGMA,size=n)
        x = scipy.stats.norm.cdf(x)
        x = x*2
        return x

    @staticmethod
    def g(covs):
        x = covs
        x1, x2, x3, x4, x5, = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
        return (x1**2)*(x2**3) + np.log(x3+1) + (x4*x5+1)**0.5 + np.exp(x5/2) - 8.2

    def inv_cum_hazard(self, v, covs):
        '''The inverse of the cumulative hazard.'''
        return (2*v / (self.h0 * np.exp(self.g(covs))))**0.5

    def cum_hazard(self, t, covs):
        '''The the cumulative hazard function.'''
        return 0.5*(self.h0 * np.exp(self.g(covs)))*(t**2)


    

class SimStudyDeep1(SimStudyLinearPH):
    '''Survival simulations study for linear prop. hazard model
        h(t | x) = h0 exp[g(x)], where g(x) is linear.

    Parameters:
        h0: Is baseline constant.
        right_c: Time for right censoring.
    '''
    def __init__(self, h0=0.1, right_c=40, c0=28., surv_grid=None):
        print(right_c)
        super().__init__(h0, right_c, c0, surv_grid)

    @staticmethod
    def sample_covs(n):
        U = np.zeros(5)
        SIGMA = np.zeros((5,5)) + 0.5
        np.fill_diagonal(SIGMA,1)
        x = np.random.multivariate_normal(U,SIGMA,size=n)
        x = scipy.stats.norm.cdf(x)
        x = x*2
        return x

    @staticmethod
    def g(covs):
        x = covs
        x1, x2, x3, x4, x5, = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
        return (x1**2)*(x2**3) + np.log(x3+1) + (x4*x5+1)**0.5 + np.exp(x5/2) - 8.2

    def inv_cum_hazard(self, v, covs):
        '''The inverse of the cumulative hazard.'''
        return (2*v / (self.h0 * np.exp(self.g(covs))))**0.5

    def cum_hazard(self, t, covs):
        '''The the cumulative hazard function.'''
        return 0.5*(self.h0 * np.exp(self.g(covs)))*(t**2)


class SimStudyDeep2(SimStudyLinearPH):
    '''Survival simulations study for linear prop. hazard model
        h(t | x) = h0 exp[g(x)], where g(x) is linear.

    Parameters:
        h0: Is baseline constant.
        right_c: Time for right censoring.
    '''
    def __init__(self, h0=0.1, right_c=80, c0=45., surv_grid=None):
        print(right_c)
        super().__init__(h0, right_c, c0, surv_grid)

    @staticmethod
    def sample_covs(n):
        U = np.zeros(5)
        SIGMA = np.zeros((5,5)) + 0.5
        np.fill_diagonal(SIGMA,1)
        x = np.random.multivariate_normal(U,SIGMA,size=n)
        x = scipy.stats.norm.cdf(x)
        x = x*2
        return x

    @staticmethod
    def g(covs):
        x = covs
        x1, x2, x3, x4, x5, = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
        return (((x1**2)*(x2**3) + np.log(x3+1) + (x4*x5+1)**0.5 + np.exp(x5/2))**2)/20 - 6

    def inv_cum_hazard(self, v, covs):
        '''The inverse of the cumulative hazard.'''
        return (2*v / (self.h0 * np.exp(self.g(covs))))**0.5

    def cum_hazard(self, t, covs):
        '''The the cumulative hazard function.'''
        return 0.5*(self.h0 * np.exp(self.g(covs)))*(t**2)

    
    
class CoxTime(models.cox_cc._CoxCCBase):
    """The Cox-Time model from [1]. A relative risk model without proportional hazards, trained
    with case-control sampling.
    
    Arguments:
        net {torch.nn.Module} -- A PyTorch net.
    
    Keyword Arguments:
        optimizer {torch or torchtuples optimizer} -- Optimizer (default: {None})
        device {str, int, torch.device} -- Device to compute on. (default: {None})
            Preferably pass a torch.device object.
            If 'None': use default gpu if available, else use cpu.
            If 'int': used that gpu: torch.device('cuda:<device>').
            If 'string': string is passed to torch.device('string').
        shrink {float} -- Shrinkage that encourage the net got give g_case and g_control
            closer to zero (a regularizer in a sense). (default: {0.})
        labtrans {pycox.preprocessing.label_tranforms.LabTransCoxTime} -- A object for transforming
            durations. Useful for prediction as we can obtain durations on the original scale.
            (default: {None})
    References:
    [1] Håvard Kvamme, Ørnulf Borgan, and Ida Scheel.
        Time-to-event prediction with neural networks and Cox regression.
        Journal of Machine Learning Research, 20(129):1–30, 2019.
        http://jmlr.org/papers/v20/18-424.html
    """
    make_dataset = models.data.CoxTimeDataset
    label_transform = LabTransCoxTime

    def __init__(self, net, optimizer=None, device=None, shrink=0., labtrans=None, loss=None):
        self.labtrans = labtrans
        super().__init__(net, optimizer, device, shrink, loss)

    def make_dataloader_predict(self, input, batch_size, shuffle=False, num_workers=0):
        input, durations = input
        input = tt.tuplefy(input)
        durations = tt.tuplefy(durations)
        new_input = input + durations 
        dataloader = super().make_dataloader_predict(new_input, batch_size, shuffle, num_workers)
        return dataloader

    def predict_surv_df(self, input, max_duration=None, batch_size=8224, verbose=False, baseline_hazards_=None,
                        eval_=True, num_workers=0):
        surv = super().predict_surv_df(input, max_duration, batch_size, verbose, baseline_hazards_,
                                       eval_, num_workers)
        if self.labtrans is not None:
            surv.index = self.labtrans.map_scaled_to_orig(surv.index)
        return surv

    def compute_baseline_hazards(self, input=None, target=None, max_duration=None, sample=None, batch_size=8224,
                                set_hazards=True, eval_=True, num_workers=0):
        if (input is None) and (target is None):
            if not hasattr(self, 'training_data'):
                raise ValueError('Need to fit, or supply a input and target to this function.')
            input, target = self.training_data
        df = self.target_to_df(target)
        if sample is not None:
            if sample >= 1:
                df = df.sample(n=sample)
            else:
                df = df.sample(frac=sample)
            df = df.sort_values(self.duration_col)
        input = tt.tuplefy(input).to_numpy().iloc[df.index.values]
        base_haz = self._compute_baseline_hazards(input, df, max_duration, batch_size, eval_, num_workers)
        if set_hazards:
            self.compute_baseline_cumulative_hazards(set_hazards=True, baseline_hazards_=base_haz)
        return base_haz

    def _compute_baseline_hazards(self, input, df_train_target, max_duration, batch_size, eval_=True,
                                  num_workers=0):
        if max_duration is None:
            max_duration = np.inf
        all_d = torch.tensor(input).reshape([torch.tensor(input).shape[1],torch.tensor(input).shape[2]]).to("cuda")
        def compute_expg_at_risk(ix, t):
            sub = all_d[ix:]
            n = sub.shape[0]
            t = np.repeat(t, n).reshape(-1, 1).astype('float32')
            bb = torch.tensor(t).to("cuda")
            with torch.no_grad():
                return float(torch.exp(self.net.eval()(sub,bb)).sum().to("cpu"))
            #return np.exp(self.predict((sub, t), batch_size, True, eval_, num_workers=num_workers)).flatten().sum()
        
        if not df_train_target[self.duration_col].is_monotonic_increasing:
            raise RuntimeError(f"Need 'df_train_target' to be sorted by {self.duration_col}")
        #input = tt.tuplefy(input)
        df = df_train_target.reset_index(drop=True)
        times = (df
                 .loc[lambda x: x[self.event_col] != 0]
                 [self.duration_col]
                 .loc[lambda x: x <= max_duration]
                 .drop_duplicates(keep='first'))
        at_risk_sum = (pd.Series([compute_expg_at_risk(ix, t) for ix, t in times.iteritems()],
                                 index=times.values)
                       .rename('at_risk_sum'))
        events = (df
                  .groupby(self.duration_col)
                  [[self.event_col]]
                  .agg('sum')
                  .loc[lambda x: x.index <= max_duration])
        base_haz =  (events
                     .join(at_risk_sum, how='left', sort=True)
                     .pipe(lambda x: x[self.event_col] / x['at_risk_sum'])
                     .fillna(0.)
                     .rename('baseline_hazards'))
        return base_haz

    def _predict_cumulative_hazards(self, input, max_duration, batch_size, verbose, baseline_hazards_,
                                    eval_=True, num_workers=0):
#         def expg_at_time(t):
#             t = np.repeat(t, n_cols).reshape(-1, 1).astype('float32')
#             if tt.tuplefy(input).type() is torch.Tensor:
#                 t = torch.from_numpy(t)
#             return np.exp(self.predict((input, t), batch_size, True, eval_, num_workers=num_workers)).flatten()
        
        all_d = torch.tensor(input).to("cuda")
        def expg_at_time(t):
            t = np.repeat(t, n_cols).reshape(-1, 1).astype('float32')
            t = torch.tensor(t).to("cuda")
            with torch.no_grad():
                return np.exp(self.net.eval()(all_d,t).to("cpu")).flatten()                

        
        
        if tt.utils.is_dl(input):
            raise NotImplementedError(f"Prediction with a dataloader as input is not supported ")
        input = tt.tuplefy(input)
        max_duration = np.inf if max_duration is None else max_duration
        baseline_hazards_ = baseline_hazards_.loc[lambda x: x.index <= max_duration]
        n_rows, n_cols = baseline_hazards_.shape[0], input.lens().flatten().get_if_all_equal()
        hazards = np.empty((n_rows, n_cols))

        for idx, t in enumerate(baseline_hazards_.index):
            if verbose:
                print(idx, 'of', len(baseline_hazards_))
            hazards[idx, :] = expg_at_time(t)
        hazards[baseline_hazards_.values == 0] = 0.  # in case hazards are inf here
        hazards *= baseline_hazards_.values.reshape(-1, 1)
        return pd.DataFrame(hazards, index=baseline_hazards_.index).cumsum()

    def partial_log_likelihood(self, input, target, batch_size=8224, eval_=True, num_workers=0):
        def expg_sum(t, i):
            sub = input_sorted.iloc[i:]
            n = sub.lens().flatten().get_if_all_equal()
            t = np.repeat(t, n).reshape(-1, 1).astype('float32')
            return np.exp(self.predict((sub, t), batch_size, True, eval_, num_workers=num_workers)).flatten().sum()

        durations, events = target
        df = pd.DataFrame({self.duration_col: durations, self.event_col: events})
        df = df.sort_values(self.duration_col)
        input = tt.tuplefy(input)
        input_sorted = input.iloc[df.index.values]

        times =  (df
                  .assign(_idx=np.arange(len(df)))
                  .loc[lambda x: x[self.event_col] == True]
                  .drop_duplicates(self.duration_col, keep='first')
                  .assign(_expg_sum=lambda x: [expg_sum(t, i) for t, i in zip(x[self.duration_col], x['_idx'])])
                  .drop([self.event_col, '_idx'], axis=1))
        
        idx_name_old = df.index.name
        idx_name = '__' + idx_name_old if idx_name_old else '__index'
        df.index.name = idx_name

        pll = df.loc[lambda x: x[self.event_col] == True]
        input_event = input.iloc[pll.index.values]
        durations_event = pll[self.duration_col].values.reshape(-1, 1)
        g_preds = self.predict((input_event, durations_event), batch_size, True, eval_, num_workers=num_workers).flatten()
        pll = (pll
               .assign(_g_preds=g_preds)
               .reset_index()
               .merge(times, on=self.duration_col)
               .set_index(idx_name)
               .assign(pll=lambda x: x['_g_preds'] - np.log(x['_expg_sum']))
               ['pll'])

        pll.index.name = idx_name_old
        return pll


class MLPVanillaCoxTime(nn.Module):
    """A version of torchtuples.practical.MLPVanilla that works for CoxTime.
    The difference is that it takes `time` as an additional input and removes the output bias and
    output activation.
    """
    def __init__(self, in_features, num_nodes, batch_norm=True, dropout=None, activation=nn.ReLU,
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        in_features += 1
        out_features = 1
        output_activation = None
        output_bias=False
        self.net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout,
                                           activation, output_activation, output_bias, w_init_)

    def forward(self, input, time):
        input = torch.cat([input, time], dim=1)
        return self.net(input)


class MixedInputMLPCoxTime(nn.Module):
    """A version of torchtuples.practical.MixedInputMLP that works for CoxTime.
    The difference is that it takes `time` as an additional input and removes the output bias and
    output activation.
    """
    def __init__(self, in_features, num_embeddings, embedding_dims, num_nodes, batch_norm=True,
                 dropout=None, activation=nn.ReLU, dropout_embedding=0.,
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        in_features += 1
        out_features = 1
        output_activation = None
        output_bias=False
        self.net = tt.practical.MixedInputMLP(in_features, num_embeddings, embedding_dims, num_nodes,
                                              out_features, batch_norm, dropout, activation,
                                              dropout_embedding, output_activation, output_bias, w_init_)

    def forward(self, input_numeric, input_categoric, time):
        input_numeric = torch.cat([input_numeric, time], dim=1)
        return self.net(input_numeric, input_categoric)

def _reconcile(s1, s2):
    pts = np.unique(np.sort(np.concatenate([s1._x, s2._x])))
    # Handle case when endpoints are inf
    cpts = pts.copy()
    cpts[0] = min(np.min(cpts[1:]), 0.) - 1
    cpts[-1] = max(np.max(cpts[:-1]), 0.) + 1
    mps = (cpts[1:] + cpts[:-1]) / 2.
    return [(pts, s(mps)) for s in (s1, s2)]


def _same_support(s1, s2):
    return np.all(s1._x[[0, -1]] == s2._x[[0, -1]])


def require_compatible(f):
    @functools.wraps(f)
    def wrapper(self, other, *args, **kwargs):
        if isinstance(other, StepFunction) and not _same_support(self, other):
            raise TypeError("Step functions have different support: %s vs. %s" % (
                self._x[[0, -1]], other._x[[0, -1]]))
        return f(self, other, *args, **kwargs)
    return wrapper


class StepFunction:
    '''A step function.'''

    def __init__(self, x, y):
        '''Initialize step function with breakpoints x and function values y.
        x and y are arrays such that
            f(z) = y[k], x[k] <= z < x[k + 1], 0 <= k < K.
        Thus, len(x) == len(y) + 1 and domain of f is (x[0], x[K]).
        '''
        if len(x) != 1 + len(y):
            raise RuntimeError("len(x) != 1 + len(y)")
        self._x = np.array(x)
        self._y = np.array(y)
        self._compress()

    @property
    def K(self):
        '''The number of steps.'''
        return len(self._y)

    def _compress(self):
        # Combine steps which have equal values
        ny = np.concatenate([[np.nan], self._y, [np.nan]])
        ys = np.diff(ny) != 0
        self._x = self._x[ys]
        self._y = self._y[ys[:-1]]

    def _binary_op(self, other, op, desc):
        if isinstance(other, StepFunction):
            (s1_x, s1_y), (s2_x, s2_y) = _reconcile(self, other)
            return StepFunction(s1_x, op(s1_y, s2_y))
        # Fall back to normal semantics otherwise
        return StepFunction(self._x, op(self._y, other))

    def __add__(self, other):
        return self._binary_op(other, operator.add, "add")

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self._binary_op(other, operator.sub, "subtract")

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        return self._binary_op(other, operator.mul, "multiply")

    def __rmul__(self, other):
        return self * other

    def __div__(self, other):
        return self._binary_op(other, operator.div, "divide")

    def __rdiv__(self, other):
        return (self ** -1) * other

    # Unary operations

    def __neg__(self):
        return StepFunction(self._x, -self._y)

    def __pow__(self, p):
        return StepFunction(self._x, pow(self._y, p))

    def __abs__(self):
        return StepFunction(self._x, abs(self._y))

    # Equality and comparison operators

    @require_compatible
    def __eq__(self, other):
        return (np.array_equal(self._x, other._x) and 
                np.array_equal(self._y, other._y))

    @require_compatible
    def __lt__(self, other):
        diff = other - self
        return np.all(diff._y > 0)

    @require_compatible
    def __le__(self, other):
        diff = other - self
        return np.all(diff._y >= 0)

    @require_compatible
    def __gt__(self, other):
        return -self < -other

    @require_compatible
    def __ge__(self, other):
        return -self <= -other

    def __call__(self, s):
        return self._y[np.minimum(np.searchsorted(self._x, s, side="right") - 1,self._y.shape[0]-1)]

    def __repr__(self):
        return "StepFunction(x=%s, y=%s)" % (repr(self._x), repr(self._y))

    def integral(self):
        nz = self._y != 0
        d = np.diff(self._x)
        return (d[nz] * self._y[nz]).sum()
    
    
def avg_baseline_hazards(networks, input, target , max_duration, batch_size, eval_=True):
    target = target.sort_values("duration")
    input = tt.tuplefy(input).to_numpy().iloc[target.index.values]
    df_train_target = target
    
    
    if max_duration is None:
        max_duration = np.inf
    all_d = torch.tensor(input).reshape([torch.tensor(input).shape[1],torch.tensor(input).shape[2]]).to("cuda")
    
    def compute_expg_at_risk(ix, t):
        sub = all_d[ix:]
        n = sub.shape[0]
        t = np.repeat(t, n).reshape(-1, 1).astype('float32')
        bb = torch.tensor(t).to("cuda")
#        ret = 0
#         with torch.no_grad():
#             for net in networks:
#                 ret += float(torch.exp(net.net.eval()(sub,bb)).sum().to("cpu"))
#        return ret/len(networks)
        with torch.no_grad():
            ret = networks[0].net.eval()(sub,bb)
            for net in networks[1:]:
                ret += net.net.eval()(sub,bb)
        #print(ret)
        return float(torch.exp(ret/len(networks)).sum().to("cpu"))
        #return ret
        #return np.exp(self.predict((sub, t), batch_size, True, eval_, num_workers=num_workers)).flatten().sum()

    if not df_train_target["duration"].is_monotonic_increasing:
        raise RuntimeError(f"Need 'df_train_target' to be sorted by duration")
    
    df = df_train_target.reset_index(drop=True)
    times = (df
             .loc[lambda x: x["duration"] != 0]
             ["duration"]
             .loc[lambda x: x <= max_duration]
             .drop_duplicates(keep='first'))
    
    at_risk_sum = (pd.Series([compute_expg_at_risk(ix, t) for ix, t in times.iteritems()],
                             index=times.values)
                   .rename('at_risk_sum'))
    events = (df
              .groupby("duration")
              [["event"]]
              .agg('sum')
              .loc[lambda x: x.index <= max_duration])
    base_haz =  (events
                 .join(at_risk_sum, how='left', sort=True)
                 .pipe(lambda x: x["event"] / x['at_risk_sum'])
                 .fillna(0.)
                 .rename('baseline_hazards'))
    return base_haz


def predict_avg_cumulative_hazards(networks, input, max_duration, batch_size, verbose, baseline_hazards_,eval_=True):
        all_d = torch.tensor(input).to("cuda")
        def expg_at_time(t):
            t = np.repeat(t, n_cols).reshape(-1, 1).astype('float32')
            t = torch.tensor(t).to("cuda")
#             with torch.no_grad():
#                 ret = np.exp(networks[0].net.eval()(all_d,t).to("cpu")).flatten()
#                 for net in networks[1:]:
#                     ret += np.exp(net.net.eval()(all_d,t).to("cpu")).flatten()
#            return ret/len(networks)
            with torch.no_grad():
                ret = networks[0].net.eval()(all_d,t).to("cpu").flatten()
                for net in networks[1:]:
                    ret += net.net.eval()(all_d,t).to("cpu").flatten()
            #print(ret)
            return np.exp(ret/len(networks))

        
        
        if tt.utils.is_dl(input):
            raise NotImplementedError(f"Prediction with a dataloader as input is not supported ")
        input = tt.tuplefy(input)
        max_duration = np.inf if max_duration is None else max_duration
        baseline_hazards_ = baseline_hazards_.loc[lambda x: x.index <= max_duration]
        n_rows, n_cols = baseline_hazards_.shape[0], input.lens().flatten().get_if_all_equal()
        hazards = np.empty((n_rows, n_cols))

        for idx, t in enumerate(baseline_hazards_.index):
            if verbose:
                print(idx, 'of', len(baseline_hazards_))
            hazards[idx, :] = expg_at_time(t)
        hazards[baseline_hazards_.values == 0] = 0.  # in case hazards are inf here
        hazards *= baseline_hazards_.values.reshape(-1, 1)
        return pd.DataFrame(hazards, index=baseline_hazards_.index).cumsum()


def get_transformers(df,cols_standardize,cols_leave):

    labtrans = CoxTime.label_transform()
    get_target = lambda df: (df["duration"].values, df["event"].values)
    labtrans.fit(*get_target(df))

    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [(col, None) for col in cols_leave]
    x_mapper = DataFrameMapper(standardize + leave)
    x_mapper.fit(df)
    return get_target,labtrans,x_mapper


def train_model(conf,df_train,df_val,transformers,verbose=False):
    get_target,labtrans,x_mapper = transformers
    x_train = x_mapper.transform(df_train).astype('float32')
    x_val = x_mapper.transform(df_val).astype('float32')
    
    y_train = labtrans.transform(*get_target(df_train))
    y_val = labtrans.transform(*get_target(df_val))
    val = tt.tuplefy(x_val, y_val)
    
    in_features = x_train.shape[1]
    num_nodes = [conf["layer_size"]]*conf["depth"]
    batch_norm = True
    dropout = conf["dropout"]
    net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout)
    model = CoxTime(net, tt.optim.Adam, labtrans=labtrans)
    
    lrfinder = model.lr_finder(x_train, y_train, conf["batch_size"], tolerance=2)
    model.optimizer.set_lr(lrfinder.get_best_lr())
    
    epochs = 1500
    callbacks = [tt.callbacks.EarlyStopping(patience=conf["patience"])]
    log = model.fit(x_train, y_train, conf["batch_size"], epochs, callbacks, verbose,
                    val_data=val.repeat(1).cat(),n_control=conf["control"])
    
    return model


def get_test_avg_srv(conf,nets,df_train,df_test,transformers):

    grid_start, grid_end, grid_step = conf["grid"]
    grid = np.arange(grid_start,grid_end,grid_step)
    get_target,labtrans,x_mapper = transformers

    x_train = x_mapper.transform(df_train).astype('float32')
    x_test = x_mapper.transform(df_test).astype('float32')

    y_train = labtrans.transform(*get_target(df_train))
    durations_test, events_test = get_target(df_test)

    yt = pd.DataFrame(np.matrix(y_train).T,columns=["duration","event"])
    base_haz = avg_baseline_hazards(nets, x_train, yt , None, conf["batch_size"], eval_=True)
    surv = np.exp(-1*predict_avg_cumulative_hazards(nets, x_test, None, conf["batch_size"], False, base_haz,eval_=True))
    surv.reindex()
    surv = surv.reset_index()
    surv["duration"] = labtrans.map_scaled_to_orig(surv.duration)
    surv = surv.set_index("duration",drop=True)
    
    ret = pd.DataFrame(grid,columns=["t"])
    for samp in range(surv.shape[1]):
            survi = surv
            st_boot = StepFunction(np.concatenate([[0],survi.index]),survi[samp])
            res_boot = st_boot(grid)
            ret[samp] = res_boot

    return ret