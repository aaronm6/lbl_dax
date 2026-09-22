import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import varray as va
import ldax_analysis as lan
from matplotlib.patches import Ellipse
from scipy.optimize import curve_fit

def get_all_data(dates, dataset_labels, rq_path, rq_names):
    #Collect all data from selected files and store them in dict
    d = {}
    for pair in dates:
        date = pair.split(",")[0]
        numOfFiles = pair.split(",")[1]

        if len(numOfFiles.split("-")) == 2:
            startPosition = int(numOfFiles.split("-")[0])
            endPosition = int(numOfFiles.split("-")[1])
        else:
            startPosition = 0
            endPosition = int(numOfFiles)

        files = []
        label = dataset_labels[dates.index(pair)]
        print("Collecting data for " + label)
        for i in range(startPosition, endPosition, 1):
            if i < 10:
                    file = date + "_00000" + str(i)
            elif i < 100:
                file = date + "_0000" + str(i)
            else:
                file = date + "_000" + str(i)

            files.append(rq_path+file+'_RQ.vrz')

        d[label] = lan.concat_RQ_files([f'{file}' for file in files], rq_names)
        print("Finished collecting data for " + label)
    return d

def plot2d(
    x,
    y,
    x_range=None,
    y_range=None,
    x_bins=50,
    y_bins=50,
    flag_log=False,
    flag_mask=True,
    y_log_scale=False,
    cmap=None,
):
    if cmap is None:
        cmap = matplotlib.cm.viridis

    if x_range is None:
        x_range = [x.min(), x.max()]
    if y_range is None:
        y_range = [y.min(), y.max()]

    if x_range[1] < x_range[0]:
        x_range = x_range[::-1]
    if y_range[1] < y_range[0]:
        y_range = y_range[::-1]

    # x bin edges (always linear)
    x_xe = np.linspace(x_range[0], x_range[1], x_bins + 1)

    # y bin edges
    if y_log_scale:
        if y_range[0] <= 0:
            raise ValueError("y_range must be positive for logarithmic y scale.")
        x_ye = np.logspace(
            np.log10(y_range[0]),
            np.log10(y_range[1]),
            y_bins + 1,
        )
    else:
        x_ye = np.linspace(y_range[0], y_range[1], y_bins + 1)

    # Histogram
    n = np.histogramdd(np.column_stack((x, y)), bins=[x_xe, x_ye])[0]

    if flag_mask:
        n_max = n[n > 0].max()
        n_min = n[n > 0].min()
        n = np.ma.masked_where(n <= 0, n)
    else:
        if flag_log:
            n[n <= 0] = 0.1
        n_max = n.max()
        n_min = n.min()

    # Color normalization
    if flag_log:
        n_Norm = matplotlib.colors.LogNorm(vmin=n_min, vmax=n_max)
    else:
        n_Norm = matplotlib.colors.Normalize(vmin=n_min, vmax=n_max)

    h = plt.pcolormesh(
        x_xe,
        x_ye,
        n.T,
        norm=n_Norm,
        cmap=cmap,
        shading="auto",
    )

    h.set_edgecolor("face")
    h.set_linewidth(0.001)

    if y_log_scale:
        plt.yscale("log")

    if flag_log and not flag_mask:
        cMax = h.get_clim()[1]
        h.set_clim((0.8, cMax))

    return n

# Gaussian model
def gaussian(x, A, mu, sigma, C):
    return A * np.exp(-0.5 * ((x - mu) / sigma)**2) + C

def apply_1d_gaussian_fit(dataset, label, range, fit_interval = 15):
    tempS1Area = []
    tempS1Error = []
    y_offset = 0

    master_cut = dataset['mhf_cut'] & dataset['drift_time_cut'] & \
        dataset['tba_cut'] & \
        dataset['bad_area_cut'] & \
        dataset['pulse_width_cut'] & \
        dataset['peak_cut']
    
    s1_area_cut = dataset['ss_s1_phe'][master_cut]
    counts, bins = np.histogram(s1_area_cut, range=range, bins=100)

    highest_5_idx = np.argsort(counts)[-5:]
    sorted_highest_5_idx = np.sort(highest_5_idx)
    temp_index  = sorted_highest_5_idx[2] 
    x = 0.5 * (bins[:-1] + bins[1:])[temp_index-fit_interval:temp_index+fit_interval]
    

    # Initial guesses
    A0 = counts.max() - counts.min()
    mu0 = x[np.argmax(counts[temp_index-fit_interval:temp_index+fit_interval])]
    sigma0 = np.std(np.repeat(x, counts[temp_index-fit_interval:temp_index+fit_interval].astype(int)))  # or (x.max()-x.min())/6
    C0 = counts.min()

    # Fit
    popt, pcov = curve_fit(
        gaussian,
        x,
        counts[temp_index-fit_interval:temp_index+fit_interval],
        sigma=np.sqrt(np.where(counts[temp_index-fit_interval:temp_index+fit_interval] == 0, 1, counts[temp_index-fit_interval:temp_index+fit_interval])),
        p0=[A0, mu0, sigma0, C0]
    )

    _, fitted_s1_area, _, _ = popt
    _, fitted_s1_error, _, _ = np.sqrt(np.diag(pcov))

    xfine = np.linspace(bins[0], bins[-1], 500)

    plt.hist(
            0.5 * (bins[:-1] + bins[1:]),
            bins=bins,
            weights=counts,
            histtype="step",
        )

    
    # Show Fit Interval
    plt.axvline(y=bins[temp_index-fit_interval], color='black', label='Fit Interval', linestyle="--")
    plt.axvline(y=bins[temp_index+fit_interval], color='black', linestyle="--")

    # Plot Gaussian Fit
    plt.plot(
        xfine,
        gaussian(xfine, *popt),
        "r-",
        lw=2,
        label="Gaussian Fit - %.2f +/- %.2f phe" % (fitted_s1_area, fitted_s1_error)
    )

        
    dataset['fitted_s1_area'] = fitted_s1_area
    dataset['fitted_s1_error'] = fitted_s1_error


    plt.xlabel("S1 Area [phe]")
    plt.ylabel("Normalized Counts")
    plt.title(label)
    plt.legend()
    plt.show()