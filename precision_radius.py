import argparse
from pathlib import Path
from rioxarray import open_rasterio
import numpy as np
from shutil import rmtree
from rasterio.enums import Resampling
import pandas as pd
import geopandas as gpd
from tqdm import trange
import seaborn as sns
from matplotlib import pyplot as plt
from datetime import date

# Set the arguments directly
class Args:
    reference = Path("D:/ITC/Thesis/Scripts/results/results/renormalize/raster/ref/ref_newnorm.tif")
    predictions = Path("D:/ITC/Thesis/Scripts/results/results/renormalize/raster/ts/Timeshift_newDB_norm_perTimeStep.tif")
    output_figures = Path("D:/ITC/Thesis/Scripts/results/results/renormalize/accuracy/output")
    cell_size = 25000

args = Args()

clip_percentile = 99.99
geopackage_name = 'layers.gpkg'

def normalize(data):
    norm_data = data.copy()
    mins, maxs = np.nanmin(norm_data.values), np.nanmax(norm_data.values)
    norm_data.values = (norm_data.values - mins) / (maxs - mins)
    return norm_data

def sort_data(data, top_k):
    data_i_flat = data.values.flatten()
    data_i_flat = np.nan_to_num(data_i_flat, 0)
    order_flat_i = (top_k + 1) * np.ones_like(data_i_flat)
    order_flat_i[np.argsort(data_i_flat)[-top_k:]] = np.arange(top_k, 0, -1)
    order_data = data.copy()
    order_data.values = order_flat_i.reshape(data.shape)
    return order_data

def top_k_classification(data, top_k):
    data_i_flat = data.values.flatten()
    data_i_flat = np.nan_to_num(data_i_flat, 0)
    order_flat_i = np.zeros_like(data_i_flat)
    order_flat_i[np.argsort(data_i_flat)[-top_k:]] = 1
    order_data = data.copy()
    order_data.values = order_flat_i.reshape(data.shape)
    return order_data

def save_means(means, fname):
    means = np.stack(means)
    df = pd.DataFrame(data=means)
    df.to_excel(fname)

def geotiff_to_geopandas(imgdata):
    gt = imgdata.rio.transform()
    pixelSizeX = gt[0]
    pixelSizeY = abs(gt[4])
    res = (pixelSizeX + pixelSizeY) / 4
    buff = res

    x, y, intensity = imgdata.x.values, imgdata.y.values, imgdata.values
    x, y = np.meshgrid(x, y)
    x, y, intensity = x.flatten(), y.flatten(), intensity.flatten()

    centroids = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_xy(x, y, crs=imgdata.rio.crs))
    centroids['deforestation'] = intensity
    centroids = centroids.dropna()
    
    return centroids

def prepare_data(reference, predictions):
    real_predictions = predictions.copy()
    real_predictions.rio.write_nodata(np.nan, inplace=True)
    
    real_reference = reference.copy()
    real_reference.rio.write_nodata(np.nan, inplace=True)
    
    alldata = np.concatenate([real_predictions.values.flatten(), real_reference.values.flatten()])
    max_clip = np.percentile(alldata, clip_percentile)
    real_predictions.values = np.clip(real_predictions.values, 0, max_clip)
    real_reference.values = np.clip(real_reference.values, 0, max_clip)
    
    return real_reference, real_predictions

def resample(data, downscale_factor, cell_size):
    return data.rio.reproject(
        data.rio.crs,
        resolution=(downscale_factor * cell_size, downscale_factor * cell_size),
        resampling=Resampling.sum,
    )

def get_dates_from_biweek(numbers, year_0=2022, month_0=1, day_0=1):
    dates = []
    for number in numbers:
        year = year_0 + (number - 1) // 24
        month = month_0 + (((number - 1) % 24) // 2)
        day = day_0 + 15 * ((number - 1) % 2)
        dates.append(date(year, month, day))
    return dates

def calculate_error(reference, predictions, max_downscale, cell_size, output_figures):
    means_diff, means_abs_diff = [], []
    for downscale_factor in trange(1, max_downscale + 1):
        reference_sampled = resample(reference, downscale_factor, cell_size)
        predictions_sampled = resample(predictions, downscale_factor, cell_size)
        
        predictions_sampled.rio.to_raster(output_figures / f'images/prediction_d{downscale_factor}.tif')
        reference_sampled.rio.to_raster(output_figures / f'images/reference_d{downscale_factor}.tif')
        
        diff_sampled = predictions_sampled.copy()
        diff_sampled.values = predictions_sampled.values - reference_sampled.values
        diff_sampled.rio.to_raster(output_figures / f'images/error_d{downscale_factor}.tif')
        
        means = np.concatenate([np.array([downscale_factor]).astype(np.int32), [np.nanmean(diff_sampled.values)]])
        means_abs = np.concatenate([np.array([downscale_factor]).astype(np.int32), [np.nanmean(np.abs(diff_sampled.values))]])
        
        means_diff.append(means)
        means_abs_diff.append(means_abs)
        
    means_diff = pd.DataFrame(means_diff, columns=['Downscale', 'Mean']).set_index('Downscale')
    means_diff.to_excel(output_figures / 'data/diff_results.xlsx')
    
    means_abs_diff = pd.DataFrame(means_abs_diff, columns=['Downscale', 'Mean']).set_index('Downscale')
    means_abs_diff.to_excel(output_figures / 'data/diff_abs_results.xlsx')
    
    sns.set_theme(rc={'figure.figsize': (14, 5)})
    lplot = sns.lineplot(data=means_abs_diff, dashes=False)
    fig = lplot.get_figure()
    plt.title('Average Absolute Error')
    plt.xlabel('Downscale Factor')
    plt.ylabel('Average Absolute Error')
    plt.legend(title='Downscale')
    sns.move_legend(fig.axes[0], "upper left", bbox_to_anchor=(1, 1))
    fig.savefig(output_figures / f'graphics/avg_abs_error.png')
    plt.close(fig)
    
    lplot = sns.lineplot(data=means_diff, dashes=False)
    fig = lplot.get_figure()
    plt.title('Average Error')
    plt.xlabel('Downscale Factor')
    plt.ylabel('Average Error')
    plt.legend(title='Downscale')
    sns.move_legend(fig.axes[0], "upper left", bbox_to_anchor=(1, 1))
    fig.savefig(output_figures / f'graphics/avg_error.png')
    plt.close(fig)

def calculate_priority_cells(reference, predictions, max_downscale, cell_size, output_figures, top_k):
    means_priority2 = []
    for downscale_factor in trange(1, max_downscale + 1):
        reference_sampled = resample(reference, downscale_factor, cell_size)
        predictions_sampled = resample(predictions, downscale_factor, cell_size)
        
        top_ks = []
        top_k_values = []
        top_k_total = []
        
        for k in range(1, top_k + 1):
            top_k_predictions = top_k_classification(predictions_sampled, k)
            top_k_reference = top_k_classification(reference_sampled, k)
            
            matches = np.logical_and(
                top_k_predictions.values == 1,
                top_k_reference.values == 1,
            )
            
            top_ks.append(k)
            top_k_values.append(matches.sum() / k)
            top_k_total.append(matches.sum() / k)
        
        results2 = pd.DataFrame(data=top_k_values)
        means = np.concatenate([np.array([downscale_factor]).astype(np.int32), [results2.sum() / top_k]])
        
        means_priority2.append(means)
        
    means_priority2 = pd.DataFrame(means_priority2, columns=['Downscale', 'Mean']).set_index('Downscale')
    means_priority2.to_excel(output_figures / f'data/priority_cells_results_top_{top_k}.xlsx')
    
    sns.set_theme(rc={'figure.figsize': (14, 5)})
    lplot = sns.lineplot(data=means_priority2, dashes=False)
    fig = lplot.get_figure()
    plt.title(f'Priority Cells (Top {top_k} cells)')
    plt.xlabel('Downscale Factor')
    plt.ylim([0, 1])
    plt.ylabel('Priority Cells Metric')
    plt.legend(title='Downscale')
    sns.move_legend(fig.axes[0], "upper left", bbox_to_anchor=(1, 1))
    fig.savefig(output_figures / f'graphics/priority_cells_top_{top_k}.png')
    plt.close(fig)

def calculate_radius_error(reference, predictions, max_downscale, cell_size, output_figures, top_k):
    buff = 200000
    means_pred_ratios, means_ref_ratios = [], []
    for downscale_factor in trange(1, max_downscale + 1):
        reference_sampled = resample(reference, downscale_factor, cell_size)
        predictions_sampled = resample(predictions, downscale_factor, cell_size)
        pred_ratio_bands, ref_ratio_bands = [], []
        predictions_gpd, reference_gpd = None, None
        predictions_sampled_gpd = geotiff_to_geopandas(predictions_sampled.copy())
        reference_sampled_gpd = geotiff_to_geopandas(reference_sampled.copy())
        
        predictions_sampled_top_gpd = predictions_sampled_gpd.sort_values(by='deforestation', ascending=False).iloc[:top_k]
        reference_sampled_top_gpd = reference_sampled_gpd.sort_values(by='deforestation', ascending=False).iloc[:top_k]
        
        predictions_sampled_buffer_gpd = predictions_sampled_top_gpd.copy()
        predictions_sampled_buffer_gpd.geometry = predictions_sampled_top_gpd.buffer(buff)
        
        reference_sampled_buffer_gpd = reference_sampled_top_gpd.copy()
        reference_sampled_buffer_gpd.geometry = reference_sampled_top_gpd.buffer(buff)
        
        ref_join = gpd.sjoin(predictions_sampled_buffer_gpd, reference_sampled_top_gpd)
        ref_ratio = len(ref_join.groupby('index_right').count()) / top_k
        pred_join = gpd.sjoin(reference_sampled_buffer_gpd, predictions_sampled_top_gpd)
        pred_ratio = len(pred_join.groupby('index_right').count()) / top_k
        
        pred_ratio_bands.append(pred_ratio)
        ref_ratio_bands.append(ref_ratio)
        
        means_pred = np.concatenate([np.array([downscale_factor]).astype(np.int32), pred_ratio_bands])
        means_ref = np.concatenate([np.array([downscale_factor]).astype(np.int32), ref_ratio_bands])

        means_pred_ratios.append(means_pred)
        means_ref_ratios.append(means_ref)
        
    means_pred_ratios = pd.DataFrame(means_pred_ratios, columns=['Downscale', 'Mean']).set_index('Downscale')
    means_pred_ratios.to_excel(output_figures / f'data/radius_precision_top_{top_k}.xlsx')
    
    means_ref_ratios = pd.DataFrame(means_ref_ratios, columns=['Downscale', 'Mean']).set_index('Downscale')
    means_ref_ratios.to_excel(output_figures / f'data/radius_recall_top_{top_k}.xlsx')
    
    sns.set_theme(rc={'figure.figsize': (14, 5)})
    lplot = sns.lineplot(data=means_pred_ratios, dashes=False)
    fig = lplot.get_figure()
    plt.title(f'Precision Radius Accuracy (Top {top_k} cells)')
    plt.xlabel('Downscale Factor')
    plt.ylim([0, 1])
    plt.ylabel('Precision Radius Accuracy')
    plt.legend(title='Downscale')
    sns.move_legend(fig.axes[0], "upper left", bbox_to_anchor=(1, 1))
    fig.savefig(output_figures / f'graphics/radius_precision_top_{top_k}.png')
    plt.close(fig)
    
    sns.set_theme(rc={'figure.figsize': (14, 5)})
    lplot = sns.lineplot(data=means_ref_ratios, dashes=False)
    fig = lplot.get_figure()
    plt.title(f'Recall Radius Accuracy (Top {top_k} cells)')
    plt.xlabel('Downscale Factor')
    plt.ylim([0, 1])
    plt.ylabel('Recall Radius Accuracy')
    plt.legend(title='Downscale')
    sns.move_legend(fig.axes[0], "upper left", bbox_to_anchor=(1, 1))
    fig.savefig(output_figures / f'graphics/radius_recall_top_{top_k}.png')
    plt.close(fig)

def main():
    # Open geotiffs
    reference = open_rasterio(args.reference)
    predictions = open_rasterio(args.predictions)
    
    assert reference.shape == predictions.shape, 'Reference and Predictions must have the same dimensions'
    
    cell_size = args.cell_size
    max_downscale = 10  # args.max_cells
    output_figures = args.output_figures
    if output_figures.exists():
        rmtree(output_figures)
    output_figures.mkdir()
    
    (output_figures / 'graphics').mkdir()
    (output_figures / 'images').mkdir()
    (output_figures / 'data').mkdir()
    
    top_k = 100
    
    real_reference, real_predictions = prepare_data(reference, predictions)
    
    calculate_error(real_reference, real_predictions, max_downscale, cell_size, output_figures)
    
    top_ks = [10, 25, 50, 100]
    for top_k in top_ks:
        calculate_priority_cells(reference, predictions, max_downscale, cell_size, output_figures, top_k)
        calculate_radius_error(reference, predictions, max_downscale, cell_size, output_figures, top_k)

if __name__ == '__main__':
    main()