from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from src.utils import utils, plotter, config, consts
from fns import *


# this file performs 1d fit on the data
# target columns are 'D_M' and 'delta_M', cb+lin for D_M and gauss+dstd0 for delta_M


# roofit_ans = {
#     'deltam': {
#         'sig_gauss': [145.42, 0.74, 40000],
#         'bkg_dstd0': [139.333, 0.772863, 0.0355922, 2.79913, 22000]
#     },
#     'dm': {
#         'sig_cb': [1862, 7.5, 7.5, 1.5, 1.5, 4, 150, 27000],
#         'bkg_lin': [30, -70000]
#     }
# }
roofit_ans = {
    'deltam': {
        'sig_gauss': [145.42, 0.74, 40000],
        'bkg_dstd0': [139.333, 0.1, 1, 3, 22000]
    },
    'dm': {
        'sig_cb': [1862, 7.5, 7.5, 1.5, 1.5, 4, 150, 27000],
        'bkg_lin': [30, -70000]
    }
}
nbins = 300

def formatfloat(f):
    ret = f'{f:.2f}'
    if f < 0:
        ret = f'({ret})'
    return ret


def find_1dfit(data: pd.Series, target_sig_model='cb', target_bkg_model='lin'):
    """
    Find the best fit of the data with the given target models
    Args:
        data: pd.Series. The data to be fitted
        target_sig_model: str. The signal model to be used. Should be either `gauss`, `gaussian`, `cb`, `crystalball` or `asymcb`
        target_bkg_model: str. The background model to be used. Should be either `dstd0`, `exp` or `lin`

    Returns:
        tuple. The coefficients of the background and signal models
    """
    # TODO: similar scale for the parameters to make fit better
    utils.log(f'Fitting {data.name} with sig_model={target_sig_model} and bkg_model={target_bkg_model}')
    bins, edges = np.histogram(data, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2
    count = bins
    if data.name == 'delta_M':
        # extrapolate 10 bins to the left and set 0
        count = np.concatenate((np.zeros(10), count))
        centers = np.concatenate((np.linspace(centers[0] - 10*(centers[1] - centers[0]), centers[0], 10), centers))

    if target_bkg_model == 'dst0':
        # bkg level = height of highest 10 mass bins
        bkglevel = np.mean(count[-int(nbins*0.1):])
    else:
        # bkg level = mean of lowest 10 counts
        bkglevel = np.mean(np.sort(count)[:int(nbins*0.1)])

    # centre = highest count's index
    center = centers[np.argmax(count)]
    sigma = data.std() / 2
    p0_bkg, p0_sig = None, None

    # useful for setting bounds
    max_exp, epsilon = 200, 0.001
    len_data, max_data = len(data), data.max()
    eff_inf = (len_data + max_data) * 10

    if target_sig_model in ['gauss', 'gaussian']:
        target_sig_fn = gaussian_1d
        p0_sig = [center, sigma, max(count) - bkglevel]
        lower_bounds_sig = [0, 0, 0]
        upper_bounds_sig = [eff_inf, eff_inf, len_data]
        # mean, sigma, amp
    elif target_sig_model == 'asymcb':
        target_sig_fn = asymcb_1d
        # find p0 from cb_1d
        p0_bkg, p0_sig = find_1dfit(data, target_sig_model='cb', target_bkg_model=target_bkg_model)
        mean, sigma, alpha, n, amp = p0_sig
        p0_sig = [mean, 0.93 * sigma, 0.9 * sigma, 0.8 * alpha, 0.65 * alpha, 0.96 * n, 0.9 * n, 1.03 * amp]
        nmin = 80 if data.name == 'delta_M' else 0
        lower_bounds_sig = [0, 0, 0, 0, 0, nmin, nmin, 0]
        upper_bounds_sig = [eff_inf, eff_inf, eff_inf, max_exp, max_exp, max_exp, max_exp, len_data]
        # mean, sigmaL, sigmaR, alphaL, alphaR, nL, nR, amp
    elif target_sig_model in ['cb', 'crystalball']:
        target_sig_fn = cb_1d
        if data.name == 'delta_M':
            p0_sig = [center, sigma, 1, center, max(count) - bkglevel]
        else:
            p0_sig = [center, sigma, 5, 1, max(count) - bkglevel]
        lower_bounds_sig = [0, 0, 0, 0, 0]
        upper_bounds_sig = [eff_inf, eff_inf, 5, max_exp, len_data]
        # mean, sigma, alpha, n, amp
    else:
        raise ValueError('target_func should be either `gauss`, `gaussian` or `cb`')

    if target_bkg_model == 'lin':
        target_bkg_fn = linear_1d
        x_1, x_2 = centers[0], centers[-1]
        y_1, y_2 = count[0], count[-1]
        slope = (y_2 - y_1) / (x_2 - x_1)
        p0_bkg = [slope, - slope * x_1 + y_1] if p0_bkg is None else p0_bkg
        lower_bounds_bkg = [-np.inf, -np.inf]
        upper_bounds_bkg = [np.inf, np.inf]
        # a, b where f(x) = a * x + b
    elif target_bkg_model == 'exp':
        target_bkg_fn = exp_1d
        p0_bkg = [epsilon, epsilon, bkglevel] if p0_bkg is None else p0_bkg
        lower_bounds_bkg = [-np.inf, -np.inf, -np.inf]
        upper_bounds_bkg = [np.inf, max_exp/max_data, np.inf]
        # a, b, c where f(x) = a * exp(b * x) + c
    elif target_bkg_model == 'dstd0':
        target_bkg_fn = deltambg_1d
        if p0_bkg is None:
            p0_bkg = roofit_ans['deltam']['bkg_dstd0'][0:4] + [bkglevel]
        else:
            p0_bkg = [min(data), p0_bkg[1] + 0.1, p0_bkg[2] + 1, p0_bkg[3] * 1.2, bkglevel]
        # p0_bkg = roofit_ans['deltam']['bkg_dstd0'][0:4] + [bkglevel] if p0_bkg is None else p0_bkg
        lower_bounds_bkg = [data.min(), 0, 0, 0, 0]
        upper_bounds_bkg = [max_data, 100, 100, 100, eff_inf]
        # m0, varA, varB, varC, amp where f(x)/amp = (1 - exp(-(x-m0)/varC)) * (x/m0)**varA + varB * (x/m0 - 1)
    elif target_bkg_model in ['gauss', 'gaussian']:
        target_bkg_fn = gaussian_1d
        p0_bkg = [center, sigma*100, bkglevel] if p0_bkg is None else p0_bkg
        lower_bounds_bkg = [-np.inf, 0, -np.inf]
        upper_bounds_bkg = [np.inf, np.inf, np.inf]
        # mean, sigma, amp
    else:
        raise ValueError('target_bkg_model should be either `dstd0`, `exp`, `lin`, `gauss` or `gaussian`')

    n_bkg_params = len(p0_bkg)
    def target_fn(x, *params):
        bkg_params = params[:n_bkg_params]
        sig_params = params[n_bkg_params:]
        ret = target_bkg_fn(x, *bkg_params) + target_sig_fn(x, *sig_params)
        if any(np.isinf(ret)):
            # utils.log('Inf values detected for target_fn')
            # replace inf with max
            ret[ret == np.inf] = np.nan
            ret = np.nan_to_num(ret, nan=max(ret))
        if any(np.isnan(ret)):
            # utils.log('Nan values detected for target_fn')
            ret = pd.Series(ret)
            ret = (ret.ffill() + ret.bfill()) / 2
            ret = ret.ffill().bfill().values
        return ret

    p0 = p0_bkg + p0_sig
    lower_bounds = lower_bounds_bkg + lower_bounds_sig
    upper_bounds = upper_bounds_bkg + upper_bounds_sig
    try:
        for varname in ['centers', 'count', 'p0_bkg', 'p0_sig']:
            varvalue = locals()[varname]
            if any(np.isnan(varvalue)):
                utils.log(f'Nan values detected for {varname}')
            elif any(np.isinf(varvalue)):
                utils.log(f'Inf values detected for {varname}')
        fit = curve_fit(target_fn, centers, count, maxfev=100000,
                        p0=p0, bounds=(lower_bounds, upper_bounds),
                        nan_policy='omit')
    except RuntimeError as e:
        utils.log('Failed to fit the data')
        raise e
    except ValueError as e:
        # inf or nan values detected
        utils.log(e.args[0] + ' for ' + data.name + ' with ' + target_sig_model + ' and ' + target_bkg_model)
        if "`x0` is infeasible" in e.args[0]:
            print('bkg:')
            for i, (lb, ub) in enumerate(zip(lower_bounds_bkg, upper_bounds_bkg)):
                print(f'\t\t{lb} <= {p0_bkg[i]} <= {ub}')
            print('sig:')
            for i, (lb, ub) in enumerate(zip(lower_bounds_sig, upper_bounds_sig)):
                print(f'\t\t{lb} <= {p0_sig[i]} <= {ub}')

        raise e
    bkg_coeffs = list(fit[0][:len(p0_bkg)])
    sig_coeffs = list(fit[0][len(p0_bkg):])
    return bkg_coeffs, sig_coeffs


def plot_fits(data, target_col, sig_model='cb', bkg_model='lin', target_dir=None):
    """
    Plot the fit of the data with the given target models
    Args:
        data: pd.DataFrame. The data to be fitted
        target_col: str. The column to be fitted
        sig_model: str. The signal model to be used. Should be either `gauss`, `gaussian` or `cb`
        bkg_model: str. The background model to be used. Should be either `dstd0`, `exp`, `lin`, `gauss` or `gaussian`
        target_dir: str. The directory to save the plots. If None, the default directory will be used

    Returns:
        None
    """
    if sig_model not in ['gauss', 'gaussian', 'cb', 'crystalball', 'asymcb']:
        raise ValueError('mode should be either `gauss`, `gaussian` or `cb`')
    if bkg_model not in ['dstd0', 'exp', 'lin', 'gauss', 'gaussian']:
        raise ValueError('bkg_model should be either `dstd0`, `exp`, `lin`, `gauss` or `gaussian`')
    target_dir = (target_dir or config.plot_dirs[4]).rstrip('/')

    bkg_popt, sig_popt = find_1dfit(data[target_col], target_sig_model=sig_model, target_bkg_model=bkg_model)
    # utils.log('bkg', [round(p, 2) for p in bkg_popt])
    # utils.log('sig', [round(p, 2) for p in sig_popt])

    fit_str = ''
    if bkg_model == 'lin':
        fit_str += f'{formatfloat(bkg_popt[0])} * x + {formatfloat(bkg_popt[1])} + '
        bkg_func = linear_1d
    elif bkg_model == 'exp':
        fit_str += f'{formatfloat(bkg_popt[0])} * exp({formatfloat(bkg_popt[1])} * x) + {formatfloat(bkg_popt[2])} + '
        bkg_func = exp_1d
    elif bkg_model in ['gauss', 'gaussian']:
        fit_str += f'{formatfloat(bkg_popt[2])} * G(m={bkg_popt[0]:.2f}, s={bkg_popt[1]:.2f}) + '
        bkg_func = gaussian_1d
    else:  # bkg_model == 'dstd0'
        fit_str += f'{formatfloat(bkg_popt[4])} * DstD0BG(m0={bkg_popt[0]:.2f}, A={bkg_popt[1]:.2f}, B={bkg_popt[2]:.2f}, C={bkg_popt[3]:.2f}) + '
        bkg_func = deltambg_1d
    if sig_model in ['gauss', 'gaussian']:
        sig_model = 'gauss'
        fit_str += f'{formatfloat(sig_popt[2])} * G(m={sig_popt[0]:.2f}, s={sig_popt[1]:.2f})'
        sig_func = gaussian_1d
    elif sig_model == 'asymcb':
        # `\n`` because the string is too long
        fit_str += f'\n{formatfloat(sig_popt[7])} * CB(m={sig_popt[0]:.2f}, sL={sig_popt[1]:.2f}, sR={sig_popt[2]:.2f}, aL={sig_popt[3]:.2f}, aR={sig_popt[4]:.2f}, nL={sig_popt[5]:.2f}, nR={sig_popt[6]:.2f})'
        sig_func = asymcb_1d
    else:  # sig_model in ['cb', 'crystalball']
        sig_model = 'cb'
        fit_str += f'{formatfloat(sig_popt[4])} * CB(m={sig_popt[0]:.2f}, s={sig_popt[1]:.2f}, a={sig_popt[2]:.2f}, n={sig_popt[3]:.2f})'
        sig_func = cb_1d

    # e.g. "deltaM_cb_lin"
    file_info = f'{target_col.replace("_", "")}_{sig_model[0]}_{bkg_model[0]}'

    # plot fit curve on top of the histogram
    fig = plt.figure(figsize=plotter.figsize)
    fig.set_label(f'1D fit of {target_col}')
    _, edges, _ = plt.hist(data[target_col], bins=nbins, label='data')
    x = (edges[:-1] + edges[1:]) / 2
    y = bkg_func(x, *bkg_popt) + sig_func(x, *sig_popt)
    plt.plot(x, y, label='fit')
    plt.legend()
    plt.title(f"1D fit of {target_col}:\n{fit_str}")
    plotter.check_filename(f'{target_dir}/overlay_{file_info}.png', 
                           check_ext=False, check_relpath=False, check_exists=True)
    plt.savefig(f'{target_dir}/overlay_{file_info}.png')
    plt.close()
    utils.log(f"Saved fit plot of {target_col} with sig_model={sig_model} and bkg_model={bkg_model} to {target_dir}")

    # plot residuals as a bar chart
    fig = plt.figure(figsize=plotter.figsize)
    fig.set_label(f"residuals of 1D fit of {target_col}")
    histogram = np.histogram(data[target_col], bins=nbins)
    x = histogram[1]
    x = (x[:-1] + x[1:]) / 2
    y = (bkg_func(x, *bkg_popt) + sig_func(x, *sig_popt) - histogram[0]) / histogram[0]  # (fit - data) / data
    y = np.where((np.isnan(y)) | (y > 1), 1, y)  # set nan and >1 values to 1
    y = 100 * np.where(y < -1, -1, y)  # set < -1 values to -1 and multiply by 100 for percentage
    plt.bar(x, y, width=(x[1] - x[0])*0.8)  # width to make margins between bars
    plt.title(f"% residuals of 1D fit of {target_col}:\n{fit_str}")
    plotter.check_filename(f'{target_dir}/residuals_{file_info}.png',
                            check_ext=False, check_relpath=False, check_exists=True)
    plt.savefig(f'{target_dir}/residuals_{file_info}.png')
    plt.close()
    utils.log(f"Saved residuals plot of {target_col} with sig_model={sig_model} and bkg_model={bkg_model} to {target_dir}")


def main():
    full_df = utils.load('full')[['D_M', "delta_M"]]
    proba_df = utils.load('proba_full.csv')
    model_name = "bdt"
    ratio = "all"

    # apply sneha's mass cuts
    full_df = full_df[(full_df['D_M'] < consts.sneha_masscuts['dmmax']) &
                      (full_df['D_M'] > consts.sneha_masscuts['dmmin'])]
    full_df = full_df[full_df['delta_M'] < consts.sneha_masscuts['deltammax']]
    proba_df = proba_df.loc[full_df.index]

    for cut in range(10):
        cut = cut / 10
        data = full_df[proba_df[f"{model_name}_{ratio}"] > cut]
        data = data[(data['D_M'] < consts.dmmax) & (data['D_M'] > consts.dmmin)]
        data = data[(data['delta_M'] < consts.deltammax) & (data['delta_M'] > consts.deltammin)]
        target_dir = f'{config.plot_dirs[4]}/{model_name}{ratio}/{int(cut * 100)}'
        
        # cb's were used for testing before applying asymcb
        plot_fits(data, 'D_M', 'gauss', 'lin', target_dir=target_dir)
        plot_fits(data, 'D_M', 'gauss', 'gauss', target_dir=target_dir)
        # plot_fits(data, 'D_M', 'cb', 'lin', target_dir=target_dir)
        # plot_fits(data, 'D_M', 'cb', 'gauss', target_dir=target_dir)
        plot_fits(data, 'D_M', 'asymcb', 'lin', target_dir=target_dir)
        plot_fits(data, 'D_M', 'asymcb', 'gauss', target_dir=target_dir)
        plot_fits(data, 'delta_M', 'gauss', 'dstd0', target_dir=target_dir)
        plot_fits(data, 'delta_M', 'cb', 'dstd0', target_dir=target_dir)
        plot_fits(data, 'delta_M', 'asymcb', 'dstd0', target_dir=target_dir)


def plot_samples():
    xs = np.linspace(138, 155, nbins)
    ysb = deltambg_1d(xs, 139, 1, 0.3, 2, 20000 * 100 / nbins)
    yss = gaussian_1d(xs, 145.2, 1, 40000 * 100 / nbins)
    plt.plot(xs, ysb+yss)
    plt.savefig(config.plot_dirs[4] + '/delta_sample.png')
    plt.close()

    xs = np.linspace(1800, 1915, nbins)
    ysb = linear_1d(xs, 30, -70000 * 100 / nbins)
    yss = asymcb_1d(xs, 1862, 7.5, 7.5, 1.5, 1.5, 4, 150, 27000 * 100 / nbins)
    plt.plot(xs, ysb+yss)
    plt.ylim(0, max(ysb+yss)*1.1)
    filename = config.plot_dirs[4] + '/d_sample.png'
    plt.savefig(filename)
    plt.close()

    utils.log(f'saved plot at {filename}')


def plot_fitsamples():
    full_df = utils.load('full')
    target_col = 'delta_M'
    data = full_df[(full_df[target_col] < 155) & (full_df[target_col] > 139)]

    utils.log('plotting')
    bins, edges = np.histogram(data[target_col], bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2

    deltam_vars = roofit_ans['deltam']['sig_gauss']
    fit_sig = gaussian_1d(centers, deltam_vars[0], deltam_vars[1], deltam_vars[2]*100/nbins)
    deltam_varb = roofit_ans['deltam']['bkg_dstd0']
    fit_bkg = deltambg_1d(centers, deltam_varb[0],
                          varA=deltam_varb[1], varB=deltam_varb[2], varC=deltam_varb[3],
                          amp=deltam_varb[4]*100/nbins)
    plt.plot(centers, fit_sig + fit_bkg, color='blue')
    plt.scatter(centers, bins, color='red')
    filename = config.plot_dirs[4] + '/delta_fitsample.png'
    plt.savefig(filename)
    plt.close()

    utils.log(f'saved plot at {filename}')


def plot_temp():
    full_df = utils.load('full')
    target_col = 'delta_M'
    data = full_df[(full_df[target_col] < consts.deltammax) & (full_df[target_col] > consts.deltammin)]

    utils.log('plotting')
    bins, edges = np.histogram(data[target_col], bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2

    fit_sig = asymcb_1d(centers, 1862, 7.5, 7.5, 1.5, 1.5, 4, 150, 27000*100/nbins)
    fit_bkg = linear_1d(centers, 30, -70000*100/nbins)
    plt.plot(centers, fit_sig + fit_bkg, color='blue')
    plt.scatter(centers, bins, color='red')
    filename = config.plot_dirs[4] + '/d_fitsample.png'
    plt.savefig(filename)

    utils.log(f'saved plot at {filename}')


if __name__ == "__main__":
    @utils.alert
    def hehe():
        main()
        plot_samples()
        plot_fitsamples()
    hehe()
