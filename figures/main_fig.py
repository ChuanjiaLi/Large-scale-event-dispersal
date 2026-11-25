"""Main-text figure panels (Figures 1c-d, 2, 3, 4 and the bottom row of Figure 5).

Run after the analysis steps in src/ (see run_all.sh). Panels are written to figures/output/ as separate PNG files
with transparent backgrounds; the published figures were assembled from these panels. Figure 1a-b (maps) and the
schematics in the top row of Figure 5 use licensed base maps or raw trip records and are not reproduced here.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns
from scipy import stats


def plotscale(ax, bounds, textcolor='black', fontsize=9, loc_scale=None, rate_scale=None, compass=False,
              loc_compass=None, rate_compass=None, accuracy=None, unit="KM", transformers=None):
    """
    :param ax: 绘制轴
    :param bounds: 图像边界
    :param textcolor: 文字颜色
    :param fontsize: 字体大小
    :param loc_scale: 比例尺位置
    :param rate_scale: 比例尺宽高缩放倍率
    :param compass: 是否添加指北针
    :param loc_compass: 指北针位置
    :param rate_compass: 指北针长宽缩放倍率
    :param accuracy: 比例尺长度，单位米（实际为其8倍长度）
    :param unit: 比例尺单位，M/m或KM/km
    :param transformers: 坐标变换
    :return:
    """
    if loc_scale is None:
        loc_scale = [0.1, 0.1]
    if rate_scale is None:
        rate_scale = [1, 1]
    if accuracy is None:
        accuracy = 'auto'
    if loc_compass is None:
        loc_compass = loc_scale.copy()
        loc_compass[1] = 1 - loc_compass[1]
    if rate_compass is None:
        rate_compass = rate_scale

    import math
    lon1 = bounds[0]
    lat1 = bounds[1]
    lon2 = bounds[2]
    lat2 = bounds[3]
    # 划定比例尺栅格所代表的距离
    if accuracy == 'auto':
        accuracy = (int((lon2 - lon1) / 0.0003 / 1000 + 0.5) * 1000)
    # 计算比例尺栅格的经纬度增加量大小Lon和Lat
    deltaLon = accuracy * 360 / (2 * math.pi * 6371004 * math.cos((lat1 + lat2) * math.pi / 360)) * rate_scale[0]
    deltaLat = (lat2 - lat1) * 0.025 * rate_scale[1]
    # 放置位置
    alon, alat = (lon2 - lon1) * loc_scale[0] + lon1, (lat2 - lat1) * loc_scale[1] + lat1  # wdh2
    # 如果有坐标变换，则进行变换
    if transformers is not None:
        deltaLon, deltaLat = transformers.transform(alon + deltaLon, alat + deltaLat)
        alon, alat = transformers.transform(alon, alat)
        deltaLon, deltaLat = deltaLon - alon, deltaLat - alat
    # 加比例尺
    from shapely.geometry import Polygon
    scale = gpd.GeoDataFrame(
        {'color': [(0, 0, 0), (1, 1, 1)],
         'geometry':
             [Polygon(
                 [(alon, alat), (alon + 4 * deltaLon, alat),
                  (alon + 4 * deltaLon, alat + deltaLat),
                  (alon, alat + deltaLat)]),
                 Polygon([(alon + 4 * deltaLon, alat), (alon + 8 * deltaLon, alat),
                          (alon + 8 * deltaLon, alat + deltaLat),
                          (alon + 4 * deltaLon, alat + deltaLat)])]
         })
    scale.plot(ax=ax, edgecolor=(0, 0, 0, 1), facecolor=scale['color'], lw=0.1, zorder=5)

    if (unit == 'KM') | (unit == 'km'):
        unit_length = 1000
    if (unit == 'M') | (unit == 'm'):
        unit_length = 1
    ax.text(alon, alat + deltaLat * 1.2, '0', color=textcolor, fontsize=fontsize, ha='center',
            va='bottom', zorder=6)
    ax.text(alon + 4 * deltaLon, alat + deltaLat * 1.2, str(int(4 * accuracy / unit_length)), color=textcolor,
            fontsize=fontsize, ha='center', va='bottom', zorder=7)
    ax.text(alon + 8 * deltaLon, alat + deltaLat * 1.2, str(int(8 * accuracy / unit_length)), color=textcolor,
            fontsize=fontsize, ha='center', va='bottom', zorder=8)
    ax.text(alon + 8.5 * deltaLon, alat + deltaLat * 1.2, unit, color=textcolor, fontsize=fontsize, ha='left',
            va='top', zorder=9)

    if compass:
        # 加指北针
        # 计算指北针栅格的经纬度增加量大小Lon和Lat
        deltaLon_compass = accuracy * 360 / (2 * math.pi * 6371004 * math.cos((lat1 + lat2) * math.pi / 360)) \
                           * rate_compass[0] * 4
        deltaLat = (lat2 - lat1) * 0.025 * rate_compass[1] * 4
        # 放置位置
        alon, alat = (lon2 - lon1) * loc_compass[0] + lon1, (lat2 - lat1) * loc_compass[1] + lat1  # wdh2
        # 如果有坐标变换，则进行变换
        if transformers is not None:
            deltaLon_compass, deltaLat = transformers.transform(alon + deltaLon_compass, alat + deltaLat)
            alon, alat = transformers.transform(alon, alat)
            deltaLon_compass, deltaLat = deltaLon_compass - alon, deltaLat - alat

        compass = gpd.GeoDataFrame(
            {'color': [(0, 0, 0), (1, 1, 1)],
             'geometry':
                 [Polygon([[alon + 4 * deltaLon, alat], [alon + 4 * deltaLon, alat + deltaLat],
                           [alon + 4 * deltaLon + 0.5 * deltaLon_compass, alat - 0.5 * deltaLat]]),
                  Polygon([[alon + 4 * deltaLon, alat], [alon + 4 * deltaLon, alat + deltaLat],
                           [alon + 4 * deltaLon - 0.5 * deltaLon_compass, alat - 0.5 * deltaLat]])]
             })
        compass.plot(ax=ax, edgecolor=(0, 0, 0, 1), facecolor=scale['color'], lw=0.1, zorder=5)
        ax.text(alon + 4 * deltaLon, alat - 0.5 * deltaLat, 'N', fontsize=fontsize, ha='center', va='top', zorder=9)


def main_fig_1_3():
    # 数据文件（R1：433条记录的城市计数）
    data_path = FD + r'city_tier_features_r1.csv'
    # 设置颜色
    color_list = np.array(['#FC6F68', '#5FACFC', '#AAAAAA'])
    # 一排图片的数量
    figure_number = 2

    data = pd.read_csv(data_path, index_col=[0])

    index = np.argsort(-data['count'].values)
    # 画图
    fig, ax = plt.subplots(1, 1, dpi=DPI)
    plt.subplots_adjust(top=0.95, bottom=0.2, right=0.95, left=0.14, hspace=0, wspace=0)
    # 设置图片大小
    fig.set_size_inches(width_inch / figure_number, width_inch / 1.8 * 0.5)
    # 设置刻度线在内
    plt.tick_params(labelsize=Fontsize, direction='in', width=ax_linewidth, which='both')
    # 修改坐标轴的粗细
    ax.spines['bottom'].set_linewidth(ax_linewidth)
    ax.spines['left'].set_linewidth(ax_linewidth)
    ax.spines['top'].set_linewidth(ax_linewidth)
    ax.spines['right'].set_linewidth(ax_linewidth)
    # 计算回归系数
    slope, intercept = np.polyfit(np.log10(np.arange(len(index)) + 1), np.log10(data['count'].values[index]), 1)
    plt.plot([1, len(index)],
             np.power(10, slope * np.log10(np.array([1, len(index)])) + intercept),
             color='#AA0000', linewidth=1, linestyle='--', zorder=1)
    # 绘制
    plt.scatter(np.arange(len(index)) + 1, data['count'].values[index], marker='o', linewidths=0,
                c=color_list[data['city_tier_ordered'].values][index], zorder=2, s=data['count'].values[index] * 1.5 + 5)
    # 设置对数坐标轴
    plt.xscale('log')
    plt.yscale('log')
    # 设置坐标轴范围
    plt.xlim(0.8, 35)
    # plt.xticks(np.linspace(0, 40000, 5))
    plt.xlabel('City rank', fontsize=Fontsize, labelpad=1.7)
    plt.ylim(0.8, 150)
    # plt.yticks(np.linspace(-1, 1, 5))
    plt.ylabel('Number of events', fontsize=Fontsize)
    # 保存
    plt.savefig(FO + r'MainFig-1-3.png', dpi=DPI, transparent=True)
    plt.close()


def main_fig_1_4():
    # 数据文件
    # R1：主分析433条记录的网格总量（M_b, M_e），原文件event_mag_info_avb.csv来自另一历史版本网格
    data_path = FD + r'event_mag_info_r1.csv'
    # 一排图片的数量
    figure_number = 2

    data = pd.read_csv(data_path)

    x, y = data['baseline_magnitude'].values, data['event_magnitude'].values

    # 画图
    fig, ax = plt.subplots(1, 1, dpi=DPI)
    plt.subplots_adjust(top=0.95, bottom=0.2, right=0.95, left=0.14, hspace=0, wspace=0)
    # 设置图片大小
    fig.set_size_inches(width_inch / figure_number, width_inch / 1.8 * 0.5)
    # 设置刻度线在内
    plt.tick_params(labelsize=Fontsize, direction='in', width=ax_linewidth, which='both')
    # 修改坐标轴的粗细
    ax.spines['bottom'].set_linewidth(ax_linewidth)
    ax.spines['left'].set_linewidth(ax_linewidth)
    ax.spines['top'].set_linewidth(ax_linewidth)
    ax.spines['right'].set_linewidth(ax_linewidth)
    # 计算回归系数
    slope, intercept = np.polyfit(np.log10(x), np.log10(y), 1)

    plt.plot([10, 60000], [10, 60000], color='#AAAAAA', linewidth=1, linestyle='--', zorder=2)
    plt.plot([np.min(x), np.max(x)],
             np.power(10, slope * np.log10(np.array([np.min(x), np.max(x)])) + intercept),
             color='#AA0000', linewidth=1, linestyle='--', zorder=3)
    # 绘制
    plt.scatter(x, y, marker='o', linewidths=0, c="#AAAAAA", alpha=0.5, zorder=1, s=15)
    # 设置对数坐标轴
    plt.xscale('log')
    plt.yscale('log')
    # 设置坐标轴范围
    plt.xlim(10, 60000)
    # plt.xticks(np.linspace(0, 40000, 5))
    plt.xlabel('Baseline OD trips', fontsize=Fontsize, labelpad=1.7)
    plt.ylim(10, 60000)
    # plt.yticks(np.linspace(-1, 1, 5))
    plt.ylabel('Event OD trips', fontsize=Fontsize)
    # 保存
    plt.savefig(FO + r'MainFig-1-4.png', dpi=DPI, transparent=True)
    plt.close()


def main_fig_2_1():
    # R1：主分析433条记录、活跃网格（到达量>1）的Spearman/Pearson（与原图数值相同）
    data_path = FD + r'df_similarity_r1.csv'
    # 排列顺序
    regime_order = ['sss', 'pearson']
    # 颜色列表
    color_list = {'sss': '#7079DE', 'pearson': '#FC6F68'}
    # 一排图片的数量
    figure_number = 2

    data = pd.read_csv(data_path)

    # 构造一个新的dataframe
    data_new = pd.DataFrame(np.vstack([data[col].values.reshape(-1, 1) for col in regime_order]), columns=['values'])
    index = pd.DataFrame(np.array(sum([[col] * len(data) for col in regime_order], [])).reshape(-1, 1), columns=['type'])
    df = pd.concat([index, data_new], axis=1)

    # 画图
    fig, ax = plt.subplots(1, 1, dpi=DPI)
    plt.subplots_adjust(top=0.95, bottom=0.2, right=0.97, left=0.17, hspace=0, wspace=0)
    # 设置图片大小
    fig.set_size_inches(width_inch / figure_number, width_inch / figure_number * 0.5)
    # 设置刻度线在内
    plt.tick_params(labelsize=Fontsize, direction='in', width=ax_linewidth)
    # 修改坐标轴的粗细
    ax.spines['bottom'].set_linewidth(ax_linewidth)
    ax.spines['left'].set_linewidth(ax_linewidth)
    ax.spines['top'].set_linewidth(ax_linewidth)
    ax.spines['right'].set_linewidth(ax_linewidth)
    # 绘图
    sns.violinplot(data=df, x='type', y='values', order=regime_order, hue='type', palette=color_list,
                   linewidth=ax_linewidth * 1.5, linecolor='k', inner='box', legend=False, cut=0, ax=ax, zorder=1)
    # 设置坐标轴范围
    # plt.xlim(-0.5, 6 * 3 + 5 - 0.5)
    plt.xticks([0, 1], ['Spearman', 'Pearson'])
    plt.xlabel("Metric", fontsize=Fontsize, labelpad=1.5)
    plt.ylim(0.0, 1.00)
    plt.yticks(np.arange(0.0, 1.01, 0.25))
    plt.ylabel('Correlation coefficient', fontsize=Fontsize)
    # 保存
    plt.savefig(FO + r'MainFig-2-1.png', dpi=DPI, transparent=True)
    plt.close()


def main_fig_2_2():
    # R1：日常份额分配（主分析433条记录）
    data_path = FD + r'grid_allocation_r1.parquet'
    # 一排图片的数量
    figure_number = 2

    data = pd.read_parquet(data_path, engine="pyarrow")

    x = data['event_density'].values
    y = data['Y_allocated'].values

    # 画图
    fig, ax = plt.subplots(1, 1, dpi=DPI)
    plt.subplots_adjust(top=0.95, bottom=0.2, right=0.95, left=0.18, hspace=0, wspace=0)
    # 设置图片大小
    fig.set_size_inches(width_inch / figure_number, width_inch / figure_number * 0.5)
    # 设置刻度线在内
    plt.tick_params(labelsize=Fontsize, direction='in', width=ax_linewidth)
    # 修改坐标轴的粗细
    ax.spines['bottom'].set_linewidth(ax_linewidth)
    ax.spines['left'].set_linewidth(ax_linewidth)
    ax.spines['top'].set_linewidth(ax_linewidth)
    ax.spines['right'].set_linewidth(ax_linewidth)
    # 设置坐标轴范围
    x_min, x_max = -20, 1000
    y_min, y_max = -30, 1000
    x_ref = np.linspace(x_min, x_max, 200)
    plt.fill_between(x_ref, x_ref, y_max, color='#FAEEF1', zorder=0)
    plt.fill_between(x_ref, y_min, x_ref, color='#EEF4FA', zorder=0)
    plt.plot([0, 1000], [0, 1000], color='k', linewidth=ax_linewidth, linestyle='--',
             label='y = x', zorder=1)
    # 计算回归系数
    slope, intercept = np.polyfit(x, y, 1)
    plt.plot([np.min(x), np.max(x)], slope * np.array([np.min(x), np.max(x)]) + intercept,
             color='#2F4FD0', linewidth=ax_linewidth * 1.5, linestyle='-',
             label='Fitted line\nslope = {:.3f}'.format(slope), zorder=2)
    # 绘制
    plt.scatter(x, y, marker='o', linewidths=ax_linewidth * 0.6,
                facecolors="#9DB7F5", edgecolors="#2F4FD0", alpha=0.6, zorder=3, s=5)
    plt.text(0.08, 0.90, 'Everyday-share allocation', fontsize=Fontsize, fontweight='bold',
             ha='left', va='top', transform=ax.transAxes, zorder=4)
    # 设置坐标轴范围
    plt.xlim(x_min, x_max)
    plt.xticks(np.arange(0, 1001, 200))
    plt.xlabel('Observed arrival volume', fontsize=Fontsize, labelpad=1.5)
    plt.ylim(y_min, y_max)
    plt.yticks(np.arange(0, 1001, 200))
    plt.ylabel('Allocated arrival volume', fontsize=Fontsize)
    # 保存
    plt.savefig(FO + r'MainFig-2-2.png', dpi=DPI, transparent=True)
    plt.close()


def main_fig_2_3():
    import matplotlib.colors as colors

    # R1：主分析433条记录；第三列改为新增出行（活动日减基线）
    data_path = FD + r'grid_allocation_r1.parquet'
    # 一排图片的数量
    figure_number = 3

    data = pd.read_parquet(data_path, engine="pyarrow")

    id_list = [4346, 2916]

    Color = np.hstack((np.array([255, 0, 0]) / 255, 0.5))
    clist = np.vstack((np.array([1, 1, 1, 0.2]), np.array(Color), np.array([0, 0, 0, 0.8])))
    newcmap = colors.LinearSegmentedColormap.from_list('chaos', clist, N=256)
    newcmap = newcmap(np.linspace(0, 1, 256))
    newcmap = colors.ListedColormap(newcmap[0:-40])
    # 三张图共用一个以0为中心的色带：正半段即原色带（白-红-深色），负半段为蓝色（只出现在第三列新增出行的减少处）
    neg = colors.LinearSegmentedColormap.from_list('neg', [(0.1, 0.2, 0.6, 0.8), (0.4, 0.6, 1, 0.5), (1, 1, 1, 0.2)], N=128)
    divcmap = colors.ListedColormap(np.vstack((neg(np.linspace(0, 1, 128)), newcmap(np.linspace(0, 1, 128)))))


    for id_now in range(len(id_list)):
        data_now = data[data['event_id'] == id_list[id_now]].copy().reset_index(drop=True)
        # public data: km offsets from the venue; divided by 111 to keep the degree-based scale bar
        data_now['grid_center_x'] = data_now['x_km'] / 111
        data_now['grid_center_y'] = data_now['y_km'] / 111
        data_now.sort_values(by=['grid_center_x', 'grid_center_y'], inplace=True)
        xy = data_now[['grid_center_x', 'grid_center_y']].values

        x_grid = np.unique(xy[:, 0])
        y_grid = np.unique(xy[:, 1])

        x_grid = np.hstack((2 * x_grid[0] - x_grid[1], x_grid, 2 * x_grid[-1] - x_grid[-2]))
        y_grid = np.hstack((2 * y_grid[0] - y_grid[1], y_grid, 2 * y_grid[-1] - y_grid[-2]))

        # 计算40×40个顶点坐标
        x_vertices = np.zeros((40, 40))
        y_vertices = np.zeros((40, 40))
        for i in range(40):
            for j in range(40):
                # 四个相邻中心点的平均
                x_vertices[i, j] = (x_grid[i] + x_grid[i + 1]) / 2.0
                y_vertices[i, j] = (y_grid[j] + y_grid[j + 1]) / 2.0

        print(x_vertices[0, 0], x_vertices[-1, -1])
        print(y_vertices[0, 0], y_vertices[-1, -1])
        # 103.92213439941406 104.18957653045655
        # 30.59450912475586 30.874187255859376
        # 116.28092956542969 116.60618495941162
        # 39.78352737426757 40.07839956283569

        data_now['extra_trips'] = data_now['event_density'] - data_now['baseline_density']
        vmax = np.max(data_now[['baseline_density', 'event_density']].values)
        vmin = -vmax
        for i in range(3):
            value_now = data_now[['baseline_density', 'event_density', 'extra_trips'][i]].values.reshape((40, 40))

            # 绘制填充图
            # 做大图
            fig, ax = plt.subplots(1, 1, dpi=900)
            # 设置图片大小
            fig.set_size_inches(width_inch / figure_number * 0.9, width_inch / figure_number * 0.9)
            plt.subplots_adjust(top=1, bottom=0.0, right=1, left=0, hspace=0, wspace=0)
            # # 寻找定位点
            # if id_list[id_now] == 2916:
            #     plt.scatter([116.353438, 116.544426, 116.504939, 116.346529],
            #                 [40.023218, 39.90934, 39.819849, 39.84999], color='k', s=4, zorder=2)
            # if id_list[id_now] == 4346:
            #     plt.scatter([103.942581, 103.990118, 104.169153, 104.162979],
            #                 [30.750134, 30.67264, 30.753317, 30.668923], color='k', s=4, zorder=2)
            # 去掉所有线条
            plt.axis('off')
            # 绘图
            mesh = plt.pcolormesh(x_vertices, y_vertices, value_now, cmap=divcmap, vmin=vmin, vmax=vmax, zorder=1)

            # 比例尺
            bounds = np.array([np.min(x_vertices), np.min(y_vertices), np.max(x_vertices), np.max(y_vertices)])
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
            plotscale(ax, bounds, textcolor='black', loc_scale=[0.08, 0.06], fontsize=Fontsize,
                      unit="KM", accuracy=1250, rate_scale=[1, 0.8])

            # 绘制场馆
            plt.scatter(0, 0,
                        c='#ffff00', marker='s', linewidths=0.5, s=10, edgecolors='k', zorder=2)
            # 每个case的三张图共用同一套vmin/vmax，只在第三张图右下角添加小范围colorbar
            if i == 2:
                cax = ax.inset_axes([0.75, 0.07, 0.025, 0.30])
                cb = fig.colorbar(mesh, cax=cax, orientation='vertical')
                cb.ax.tick_params(labelsize=Fontsize, width=0.4, length=1.2, pad=1)
                cb.outline.set_linewidth(0.4)
                cb.set_ticks([vmin, 0, vmax])
                cb.ax.set_yticklabels([int(vmin), 0, int(vmax)])
                cb.ax.yaxis.set_label_position('right')
                cb.ax.set_ylabel('OD trips', fontsize=Fontsize, rotation=270, labelpad=4, va='center')
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
            # 显示图形
            plt.savefig(FO + "MainFig-2-3-%d-%d.png" % (id_list[id_now], i + 1), dpi=900, transparent=True)


def main_fig_3_1():
    # R1：日常份额分配的FVU（主分析433条记录），按M_b四分位分组着色；见prepare_r1_data.py
    df_path = FD + r'df_fvu_r1.csv'
    # 线宽
    linewidth = [2, 1]
    # 点大小
    s = 2
    # 一排图片的数量
    figure_number = 1.6

    # 读取数据
    df_all_results = pd.read_csv(df_path)
    x_all, y_all = df_all_results['baseline_magnitude'].values, df_all_results['fvu'].values
    # 对数-对数最小二乘拟合：log10 FVU = intercept + slope * log10 M_b
    fit = stats.linregress(np.log10(x_all), np.log10(y_all))

    # 画图
    fig, ax = plt.subplots(1, 1, dpi=DPI)
    plt.subplots_adjust(top=0.95, bottom=0.22, right=0.95, left=0.14, hspace=0, wspace=0)
    # 设置图片大小
    fig.set_size_inches(width_inch / figure_number, width_inch / 1.8 * 0.5)
    # 设置刻度线在内
    plt.tick_params(labelsize=Fontsize, direction='in', width=ax_linewidth, which='both')
    # 修改坐标轴的粗细
    ax.spines['bottom'].set_linewidth(ax_linewidth)
    ax.spines['left'].set_linewidth(ax_linewidth)
    ax.spines['top'].set_linewidth(ax_linewidth)
    ax.spines['right'].set_linewidth(ax_linewidth)

    # 绘制散点（按基线流量四分位分组着色，与原图一致；分组见prepare_r1_data.py）
    grp = pd.read_csv(FD + r'df_fig4_r1.csv')[['event_id', 'magnitude_group']]
    df_all_results = df_all_results.merge(grp, on='event_id')
    PALETTE_MAG = {'Small': '#5FACFC', 'Medium': '#FC6F68', 'Large': '#FFB64D', 'Mega': '#5BC39E'}
    for mag in ['Small', 'Medium', 'Large', 'Mega']:
        data_now = df_all_results[df_all_results['magnitude_group'] == mag]
        plt.scatter(data_now['baseline_magnitude'], data_now['fvu'], color=PALETTE_MAG[mag], s=s, zorder=2)

    # 绘制拟合线
    x_range = np.logspace(np.log10(x_all.min()), np.log10(x_all.max()), 200)
    y_fit = np.power(10, fit.intercept) * x_range ** fit.slope
    plt.plot(x_range, y_fit, color="k", linestyle='-', linewidth=linewidth[0], zorder=3)

    # 添加公式标注
    plt.text(0.05, 0.54, r'$y = {:.3f} \cdot x^{{{:.3f}}}$'.format(np.power(10, fit.intercept), fit.slope),
             fontsize=Fontsize, transform=ax.transAxes, color="k", fontweight='bold')
    plt.text(0.81, 0.85, r'$R^2 = {:.3f}$'.format(fit.rvalue ** 2),
             fontsize=Fontsize, transform=ax.transAxes, color="k", fontweight='bold')
    plt.text(0.81, 0.75, r'$p < 0.001$' if fit.pvalue < 0.001 else r'$p = {:.3f}$'.format(fit.pvalue),
             fontsize=Fontsize, transform=ax.transAxes, color="k", fontweight='bold')

    # 原图图例文字位置（色点在PPT中添加）
    plt.text(0.08, 0.22, r'Small', fontsize=Fontsize, transform=ax.transAxes, color="k")
    plt.text(0.22, 0.22, r'Medium', fontsize=Fontsize, transform=ax.transAxes, color="k")
    plt.text(0.4, 0.22, r'Large', fontsize=Fontsize, transform=ax.transAxes, color="k")
    plt.text(0.54, 0.22, r'Mega', fontsize=Fontsize, transform=ax.transAxes, color="k")
    plt.text(0.08, 0.1, r'Power law fit', fontsize=Fontsize, transform=ax.transAxes, color="k")
    plt.xlabel('Baseline traffic magnitude (trips)', fontsize=Fontsize)
    plt.ylabel(r'FVU (1 - $R^2$)', fontsize=Fontsize)
    plt.xscale('log')
    plt.yscale('log')
    # 保存
    plt.savefig(FO + r'MainFig-3-1.png', dpi=DPI, transparent=True)
    plt.close()
    print('Figure 3a: slope = %.3f, R2 = %.3f, n = %d' % (fit.slope, fit.rvalue ** 2, len(x_all)))


def main_fig_3_2():
    # R1：在相对冲击q相近的活动中考察FVU与基线量的关系（替换原四组均值柱状图）
    df_path = FD + r'df_fvu_r1.csv'
    # 排列顺序（相对冲击q分组）
    band_cols = ['1-1.5', '1.5-2', '2-3', '>3']
    # 颜色（与图3a一致）
    PALETTE_BAND = {'1-1.5': '#5FACFC', '1.5-2': '#FC6F68', '2-3': '#FFB64D', '>3': '#5BC39E'}
    LABEL_BAND = {'1-1.5': r'$1 < q \leq 1.5$', '1.5-2': r'$1.5 < q \leq 2$', '2-3': r'$2 < q \leq 3$', '>3': r'$q > 3$'}
    # 一排图片的数量
    figure_number = 1 / (1 - 1 / 1.6)

    # 读取数据
    df_all_results = pd.read_csv(df_path)

    # 画图
    fig, ax = plt.subplots(1, 1, dpi=DPI)
    plt.subplots_adjust(top=0.95, bottom=0.24, right=0.98, left=0.23, hspace=0, wspace=0)
    # 设置图片大小
    fig.set_size_inches(width_inch / figure_number, width_inch / 1.8 * 0.5)
    # 设置刻度线在内
    plt.tick_params(labelsize=Fontsize, direction='in', width=ax_linewidth, which='both')
    # 修改坐标轴的粗细
    ax.spines['bottom'].set_linewidth(ax_linewidth)
    ax.spines['left'].set_linewidth(ax_linewidth)
    ax.spines['top'].set_linewidth(ax_linewidth)
    ax.spines['right'].set_linewidth(ax_linewidth)

    # 每个q分组：散点 + 组内幂律拟合
    for band in band_cols:
        data_now = df_all_results[df_all_results['q_band'] == band].copy()
        x, y = data_now['baseline_magnitude'].values, data_now['fvu'].values
        fit = stats.linregress(np.log10(x), np.log10(y))
        x_range = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
        plt.scatter(x, y, color=PALETTE_BAND[band], s=1, alpha=0.35, linewidths=0, zorder=1)
        plt.plot(x_range, np.power(10, fit.intercept) * x_range ** fit.slope, color=PALETTE_BAND[band],
                 linewidth=1.5, zorder=2, label=LABEL_BAND[band])
        print('Figure 3b: q %s, n = %d, slope = %.3f' % (band, len(x), fit.slope))

    plt.legend(fontsize=Fontsize * 0.75, frameon=False, loc='lower left', handlelength=1, handletextpad=0.4,
               borderaxespad=0.2, labelspacing=0.2)
    plt.xlabel("Baseline traffic magnitude (trips)", fontsize=Fontsize)
    plt.ylabel(r"FVU (1 - $R^2$)", fontsize=Fontsize)
    plt.xscale('log')
    plt.yscale('log')
    # 保存
    plt.savefig(FO + r'MainFig-3-2.png', dpi=DPI, transparent=True)
    plt.close()


def main_fig_3_3():
    from shapely import wkt, ops
    from matplotlib.collections import LineCollection
    import matplotlib.colors as colors
    from pyproj import Transformer

    # 数据位置
    # R1：日常份额分配的FVU（主分析433条记录）；拟合参数由数据计算
    df_path = FD + r'df_fvu_r1.csv'
    
    # 城市顺序
    city_configs = [
        {
            'name': 'Beijing', 'chinese_name': '北京市', 'color': '#7079DE'
        },
        {
            'name': 'Chengdu', 'chinese_name': '成都市', 'color': '#D15B7E'
        },
        {
            'name': 'Qingdao', 'chinese_name': '青岛市', 'color': '#FC6F68'
        },
        {
            'name': 'Shanghai', 'chinese_name': '上海市', 'color': '#FFA06A'
        },
        {
            'name': 'Shenzhen', 'chinese_name': '深圳市', 'color': '#FFB64D'
        },
        {
            'name': 'Zhengzhou', 'chinese_name': '郑州市', 'color': '#FFDA43'
        },
    ]
    # xy范围
    xlim_cols = [[0, 16000], [0, 16000], [0, 8000], [0, 24000], [0, 12000], [0, 12000]]
    ylim_cols = [[-0.4, 1.2], [-2, 1.2], [-2, 1.2], [0.2, 1.0], [0, 1.2], [-1, 1.4]]
    # 线宽
    linewidth = 2
    # 点大小
    s = 4
    # 一排图片的数量
    figure_number = 2.8

    # 读取数据
    df_all_results = pd.read_csv(df_path)
    # data_map = pd.read_excel(data_map)
    # 变为西安80坐标系 https://spatialreference.org/ref/epsg/?page=7
    # map_all = gpd.GeoDataFrame.from_file(data_ground).to_crs("epsg:2343")


    for i, config in enumerate(city_configs):

        # # 画地图
        # # 选出对应的地图数据
        # if i == 0 or i == 3:
        #     index = data_map['Province_amap'] == config['chinese_name']
        # else:
        #     index = data_map['City_amap'] == config['chinese_name']
        # map_new = map_all[np.array(index)].copy().reset_index(drop=True)
        # # 读取OD数据
        # df_od = pd.read_csv(config['file'])
        # # 这里简单取前1000条或随机采样，避免渲染太慢
        # if len(df_od) > 10000:
        #     df_od = df_od.sample(10000, random_state=42)
        # # 创建一个存储列表
        # trips_loc = np.zeros((len(df_od), 4))
        # # 把点的坐标系也转换了
        # transformer = Transformer.from_crs("EPSG:4326", "EPSG:2343", always_xy=True)
        # trips_loc[:, 0], trips_loc[:, 1] = transformer.transform(df_od.iloc[:, 0], df_od.iloc[:, 1])
        # trips_loc[:, 2], trips_loc[:, 3] = transformer.transform(df_od.iloc[:, 2], df_od.iloc[:, 3])

        # # 绘制底图
        # fig, ax = plt.subplots(1, 1)
        # # 设置图片大小
        # fig.set_size_inches(width_inch / figure_number * 0.5, width_inch / figure_number * 0.4)
        # plt.subplots_adjust(top=1, bottom=0.0, right=1, left=0, hspace=0, wspace=0)
        # # 去掉所有线条
        # plt.axis('off')
        # # 创建线段集合（每条线需要两个点：起点和终点）
        # segments = np.stack([trips_loc[:, :2], trips_loc[:, 2:4]], axis=1)
        # # 创建LineCollection对象
        # lc = LineCollection(segments, color=config['color'], linewidths=0.8, alpha=0.1)
        # # 将线段集合添加到坐标轴
        # ax.add_collection(lc)
        # # 绘制底图
        # map_new.plot(ax=ax, column=None,
        #              color="#F0F0F0", edgecolor="grey", linewidth=0.4, zorder=1)
        # # 合并边界
        # Poly = map_new["geometry"].values
        # polygons = []
        # for polygon in Poly:
        #     polygons.append(polygon)
        # polygons = ops.unary_union(polygons)
        # boundary_all = gpd.GeoSeries(polygons)
        # boundary_all.boundary.plot(ax=ax, color='k', linewidth=0.8, zorder=2)
        # # 给出边界极值
        # minx, miny, maxx, maxy = boundary_all.bounds.values[0]
        # # 边界留下一点距离
        # rate_x = 0.01
        # rate_y = 0.01
        # minx_new, maxx_new = minx - rate_x * (maxx - minx), maxx + rate_x * (maxx - minx)
        # miny_new, maxy_new = miny - rate_y * (maxy - miny), maxy + rate_y * (maxy - miny)
        # # 设置坐标轴
        # plt.xlim(minx_new, maxx_new)
        # plt.ylim(miny_new, maxy_new)
        # # 保存
        # plt.savefig(FO + r'MainFig-3-3-map-%s.png' % config['name'], dpi=DPI, transparent=True)
        # plt.close()

        # 画线图
        # 筛选该城市的数据
        city_data = df_all_results[df_all_results['city'] == config['name']]
        city_fit = stats.linregress(np.log10(city_data['baseline_magnitude']), np.log10(city_data['fvu']))
        config['fit_func'] = lambda x, b=city_fit: np.power(10, b.intercept) * x ** b.slope
        config['r2'] = city_fit.rvalue ** 2
        print('Figure 3c: %s, n = %d, slope = %.3f, R2 = %.3f, p = %.2g' % (
            config['name'], len(city_data), city_fit.slope, city_fit.rvalue ** 2, city_fit.pvalue))
        # 画图
        fig, ax = plt.subplots(1, 1, dpi=DPI)
        plt.subplots_adjust(top=0.915, bottom=0.22, right=0.96, left=0.245, hspace=0, wspace=0)
        # 设置图片大小
        fig.set_size_inches(width_inch / figure_number, width_inch / figure_number * 0.9)
        # 设置刻度线在内
        plt.tick_params(labelsize=Fontsize, direction='in', width=ax_linewidth)
        # 修改坐标轴的粗细
        ax.spines['bottom'].set_linewidth(ax_linewidth)
        ax.spines['left'].set_linewidth(ax_linewidth)
        ax.spines['top'].set_linewidth(ax_linewidth)
        ax.spines['right'].set_linewidth(ax_linewidth)
        # 绘制散点
        plt.scatter(city_data['baseline_magnitude'], city_data['fvu'], color='#555555', s=s, zorder=1)
        # 绘制拟合线
        x_range_min, x_range_max = city_data['baseline_magnitude'].min(), city_data['baseline_magnitude'].max()
        x_range = np.linspace(x_range_min, x_range_max * 1.1, 100)
        y_fit = config['fit_func'](x_range)
        plt.plot(x_range, y_fit, color=config['color'], linestyle='--', linewidth=linewidth, zorder=2)
        # # 设置坐标轴范围
        # plt.xlim(xlim_cols[i][0], xlim_cols[i][1])
        # plt.xticks(np.linspace(xlim_cols[i][0], xlim_cols[i][1], 5),
        #            np.round(np.linspace(xlim_cols[i][0] / 1000, xlim_cols[i][1] / 1000, 5), 0).astype(int))
        # 添加R2和p值标注
        plt.text(0.60, 0.85, r'$R^2 = {:.3f}$'.format(config['r2']),
             fontsize=Fontsize, transform=ax.transAxes, color="k", fontweight='bold')
        plt.text(0.60, 0.75, r'$p < 0.001$' if city_fit.pvalue < 0.001 else r'$p = {:.3f}$'.format(city_fit.pvalue),
             fontsize=Fontsize, transform=ax.transAxes, color="k", fontweight='bold')

        if i >= 3:
            plt.xlabel('Baseline traffic magnitude (trips)', fontsize=Fontsize)
        else:
            plt.xlabel('', fontsize=Fontsize)
        # plt.ylim(ylim_cols[i][0], ylim_cols[i][1])
        # plt.yticks(np.linspace(ylim_cols[i][0], ylim_cols[i][1], 5))
        plt.xscale('log')
        plt.yscale('log')
        if i % 3 == 0:
            plt.ylabel(r'FVU (1 - $R^2$)', fontsize=Fontsize)
        else:
            plt.ylabel('')
        plt.title(config['name'], fontsize=Fontsize, pad=4)
        # 保存
        plt.savefig(FO + r'MainFig-3-3-line-%s.png' % config['name'], dpi=DPI, transparent=True)
        plt.close()


def main_fig_4_1():
    # 数据位置
    # R1：主分析433条记录，按网格基线总量M_b四分位分组（见prepare_r1_data.py）
    df_path = FD + r'df_fig4_r1.csv'
    # 排列顺序
    magnitude_cols = ['Small', 'Medium', 'Large', 'Mega']
    mode_cols = ['DRV', 'SUBWAY', 'BUS', 'BYC', 'WLK']
    mode_col_names = ['DRV', 'Metro', 'BUS', 'BYC', 'WLK']
    # 颜色
    color_list = ['#7079DE', '#D15B7E', '#FC6F68', '#FFB64D', '#FFDA43']
    # color_list = ['#5FACFC', '#7079DE', '#FFB64D', '#FA816E', '#D15B7E']
    # 一排图片的数量
    figure_number = 1

    # 读取
    df_mode_freq = pd.read_csv(df_path)
    # 数据处理：宽表转长表
    df_mode_long = pd.melt(
        df_mode_freq,
        id_vars=['magnitude_group'],
        value_vars=mode_cols,
        var_name='Transport Mode',
        value_name='Share'
    )

    # 画图
    fig, ax = plt.subplots(1, 1, dpi=DPI)
    plt.subplots_adjust(top=0.95, bottom=0.13, right=0.99, left=0.08, hspace=0, wspace=0)
    # 设置图片大小
    fig.set_size_inches(width_inch / figure_number, width_inch / figure_number * 0.25)
    # 设置刻度线在内
    plt.tick_params(labelsize=Fontsize, direction='in', width=ax_linewidth)
    # 修改坐标轴的粗细
    ax.spines['bottom'].set_linewidth(ax_linewidth)
    ax.spines['left'].set_linewidth(ax_linewidth)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 提取数据并绘图
    for i in range(4):
        data_now = df_mode_long[df_mode_long['magnitude_group'] == magnitude_cols[i]].copy()
        for j in range(5):
            data_plot = data_now[data_now['Transport Mode'] == mode_cols[j]]['Share'].values
            # 绘图
            plt.bar(j + 6 * i, np.mean(data_plot), color=color_list[j], width=0.8,
                    yerr=np.std(data_plot), capsize=2, error_kw={'elinewidth': 1, 'capthick': 1})
    # 创建虚拟区域来设置legend
    for i in range(5):
        plt.bar(-10, 0, color=color_list[i], width=0.8, label=mode_col_names[i])
    # 显示图例，handlelength：图例标记的长度，默认2.0；handletextpad：图例标记和标签的距离，默认0.8；columnspacing：图例列间距，默认2.0
    plt.legend(ncol=5, fontsize=Fontsize, handlelength=1, handletextpad=0.4, columnspacing=1, loc=(0.015, 0.86))
    # 设置坐标轴范围
    plt.xlim(-0.5, 6 * 3 + 5 - 0.5)
    plt.xticks(6 * np.arange(4) + 5 / 2 - 0.5, magnitude_cols)
    # plt.xlabel("Year", fontsize=Fontsize, labelpad=1)
    plt.ylim(0, 0.6)
    plt.yticks(np.arange(0, 0.61, 0.2))
    plt.ylabel("Mean modal share (%)", fontsize=Fontsize)
    # 保存
    plt.savefig(FO + r'MainFig-4-1.png', dpi=DPI, transparent=True)
    plt.close()


def main_fig_4_2():
    # 数据位置
    # R1：主分析433条记录；形态指标按Methods公式在主数据的D_e上重算
    df_path = FD + r'df_fig4_r1.csv'
    # 排列顺序
    regime_order = ['Small', 'Medium', 'Large', 'Mega']
    metrics_cols = ['iso_area', 'iso_compactness', 'metro_centrality', 'density',
                    'radius_of_gyration', 'concentration_ratio', 'directionality', 'centroid_displacement']
    # 对应文本
    titles_cols = ['15-min Isochrone area', 'Isochrone compactness', 'Metro station count', 'Road network density',
                  'Radius of gyration', 'Spatial gini coefficient', 'Directionality', 'Centroid displacement']
    ylabels_cols = [r'Area ($\mathrm{10}^{2}$ $\mathrm{km}^{-2}$)', 'Isoperimetric quotient (IPQ)', 'Count', r'Density ($\mathrm{km} \cdot \mathrm{km}^{-2}$)',
                   'Radius (km)', 'Gini Index', 'Ratio', 'Distance (km)']
    # 坐标轴范围
    ylim_cols = [[0, 16], [0.6, 1], [0, 24], [0, 8], [4, 16], [0.5, 0.9], [0.6, 3.0], [0, 16]]
    # 因为连线图层在小提琴图上方，会盖住中线，需要单独设置下
    line_connect = [[0.5, 0.5], [0.7, 0.7], [0.5, 0.5], [0.5, 0.65], [0.5, 0.5], [0.5, 0.5], [0.5, 0.4], [0.6, 0.5]]
    # 颜色列表
    color_list = {'Small': '#5FACFC', 'Medium': '#FC6F68', 'Large': '#FFB64D', 'Mega': '#5BC39E'}
    # 重置字体大小
    Fontsize = 7
    # 一排图片的数量
    figure_number = 4

    # 读取数据
    df_all_results = pd.read_csv(df_path)

    # 循环出图
    for i, metric in enumerate(metrics_cols):

        if i == 0:
            df_all_results[metric] = df_all_results[metric] / 100

        # 画图
        fig, ax = plt.subplots(1, 1, dpi=DPI)
        plt.subplots_adjust(top=0.89, bottom=0.24, right=0.97, left=0.25, hspace=0, wspace=0)
        # 设置图片大小
        fig.set_size_inches(width_inch / figure_number, width_inch / figure_number * 1.2)
        # 设置刻度线在内
        plt.tick_params(labelsize=Fontsize, direction='in', width=ax_linewidth)
        # 修改坐标轴的粗细
        ax.spines['bottom'].set_linewidth(ax_linewidth)
        ax.spines['left'].set_linewidth(ax_linewidth)
        ax.spines['top'].set_linewidth(ax_linewidth)
        ax.spines['right'].set_linewidth(ax_linewidth)
        # 绘图
        sns.violinplot(
            data=df_all_results, x='magnitude_group', y=metric, order=regime_order, hue='magnitude_group',
            palette=color_list, linewidth=ax_linewidth * 1.5, linecolor='k', inner='box', legend=False, ax=ax, zorder=1
        )
        # 计算检验
        for j in range(4):
            # 统计检验
            group1 = df_all_results[df_all_results['magnitude_group'] == regime_order[j % 4]][metric]
            group2 = df_all_results[df_all_results['magnitude_group'] == regime_order[(j + 1) % 4]][metric]
            # 简单的 t-test
            t_stat, p_val = stats.ttest_ind(group1.dropna(), group2.dropna())
            # 标注文本
            # if p_val < 0.001:
            #     star = 'p < 0.001'
            # elif p_val < 0.01:
            #     star = 'p < 0.01'
            # elif p_val < 0.05:
            #     star = 'p < 0.05'
            # else:
            #     star = 'ns'
            if p_val < 0.001:
                star = '***'
            elif p_val < 0.01:
                star = '**'
            elif p_val < 0.05:
                star = '*'
            else:
                star = 'ns'

            if i != 1:
                # 绘制检验图
                plt.plot([j % 4 + (1 - j // 3) * 0.05, j % 4 + (1 - j // 3) * 0.05,
                          (j + 1) % 4 - (1 - j // 3) * 0.05, (j + 1) % 4 - (1 - j // 3) * 0.05],
                         [(line_connect[i][0] + j // 3 * 0.05) * (ylim_cols[i][1] - ylim_cols[i][0]) + ylim_cols[i][0],
                          (0.8 + j // 3 * 0.1) * (ylim_cols[i][1] - ylim_cols[i][0]) + ylim_cols[i][0],
                          (0.8 + j // 3 * 0.1) * (ylim_cols[i][1] - ylim_cols[i][0]) + ylim_cols[i][0],
                          (line_connect[i][1] + j // 3 * 0.05) * (ylim_cols[i][1] - ylim_cols[i][0]) + ylim_cols[i][0]],
                         linewidth=ax_linewidth, color='k', zorder=2, linestyle='--'
                         )
                # 用相对位置打上文本
                plt.text((j % 4 + (j + 1) % 4) / 2,
                         (0.8 + j // 3 * 0.1) * (ylim_cols[i][1] - ylim_cols[i][0]) + ylim_cols[i][0],
                         star, fontsize=Fontsize, ha='center', va='bottom', zorder=3)
            else:
                # 绘制检验图
                plt.plot([j % 4 + (1 - j // 3) * 0.05, j % 4 + (1 - j // 3) * 0.05,
                          (j + 1) % 4 - (1 - j // 3) * 0.05, (j + 1) % 4 - (1 - j // 3) * 0.05],
                         [(line_connect[i][0] - j // 3 * 0.05) * (ylim_cols[i][1] - ylim_cols[i][0]) + ylim_cols[i][0],
                          (0.2 - j // 3 * 0.1) * (ylim_cols[i][1] - ylim_cols[i][0]) + ylim_cols[i][0],
                          (0.2 - j // 3 * 0.1) * (ylim_cols[i][1] - ylim_cols[i][0]) + ylim_cols[i][0],
                          (line_connect[i][1] - j // 3 * 0.05) * (ylim_cols[i][1] - ylim_cols[i][0]) + ylim_cols[i][0]],
                         linewidth=ax_linewidth, color='k', zorder=2, linestyle='--'
                         )
                # 用相对位置打上文本
                plt.text((j % 4 + (j + 1) % 4) / 2,
                         (0.2 - j // 3 * 0.1) * (ylim_cols[i][1] - ylim_cols[i][0]) + ylim_cols[i][0],
                         star, fontsize=Fontsize, ha='center', va='bottom', zorder=3)

        # 设置坐标轴范围
        # plt.xlim(-0.5, 6 * 3 + 5 - 0.5)
        if np.any(np.array([2, 3, 6, 7]) - i == 0):
            plt.xticks([0, 1, 2, 3], regime_order, rotation=90)
        else:
            plt.xticks([0, 1, 2, 3], [])
        plt.xlabel("")
        plt.ylim(ylim_cols[i][0], ylim_cols[i][1])
        plt.yticks(np.linspace(ylim_cols[i][0], ylim_cols[i][1], 5))
        plt.ylabel(ylabels_cols[i], fontsize=Fontsize)
        plt.title(titles_cols[i], fontsize=Fontsize, pad=4)
        # 保存
        plt.savefig(FO + r'MainFig-4-2-%d.png' % (i + 1), dpi=DPI, transparent=True)
        plt.close()


def main_fig_4_3():
    # 数据位置
    # R1：5分钟额外流量序列（活动结束后4小时，433条，见prepare_r1_data.py），分组改为主数据的M_b四分位
    df_path = FD + r'df_series_r1.csv'
    # X轴最大范围
    MAX_LEN = 48
    # 坐标轴范围
    ylim_cols = [[0, 3], [0, 3]]
    # ylim_cols = [[0, 2270], [0, 2270]]
    # 线条粗细
    linewidth = [1, 2]
    # 颜色
    PALETTE_MAG = {
        'Small': '#5FACFC',
        'Medium': '#FC6F68',
        'Large': '#FFB64D',
        'Mega': '#5BC39E'
    }
    # 线型
    linestyle_mag = {
        'Small': ':',
        'Medium': '--',
        'Large': '-.',
        'Mega': '-'
    }
    # 一排图片的数量
    figure_number = 2

    # 读取数据
    df_avb_series = pd.read_csv(df_path)
    # df_avb_series['extra_flow'] = df_avb_series['norm_extra_flow']
    df_avb_series['extra_flow'] = np.log10(df_avb_series['extra_flow'] + 1)

    # Plot chart 1
    fig, ax = plt.subplots(1, 1, dpi=DPI)
    plt.subplots_adjust(top=0.95, bottom=0.20, right=0.97, left=0.15, hspace=0, wspace=0)
    # 设置图片大小
    fig.set_size_inches(width_inch / figure_number, width_inch / figure_number * 0.5)
    # 设置刻度线在内
    plt.tick_params(labelsize=Fontsize, direction='in', width=ax_linewidth)
    # 修改坐标轴的粗细
    ax.spines['bottom'].set_linewidth(ax_linewidth)
    ax.spines['left'].set_linewidth(ax_linewidth)
    ax.spines['top'].set_linewidth(ax_linewidth)
    ax.spines['right'].set_linewidth(ax_linewidth)
    # 绘制
    for mag in ['Small', 'Medium', 'Large', 'Mega']:
        # 筛选出属于当前聚类的所有活动
        df_cluster = df_avb_series[df_avb_series['magnitude_group'] == mag]

        # 提取这些活动的时序数据 (不包括event_id, cluster, mode_name等非数据列)
        # 我们需要找到代表时间序列的列
        series_data = df_cluster[['extra_flow', 'extra_magnitude', 'index_col']]

        # 绘制“意大利面” (所有个体曲线)
        for _, series in series_data.groupby('index_col'):
            extra_mag = series_data['extra_magnitude'].mean()
            plt.plot(range(MAX_LEN), series['extra_flow'], color='grey', alpha=0.05,
                     linewidth=linewidth[0], zorder=0)

        # 计算并绘制类簇中心（平均线）
        # 注意：这里我们直接从数据中计算平均值，而不是用tslearn模型中的标准化质心
        # 这样Y轴的单位就是真实的“额外流量”，更具解释性
        if len(series_data) > 0:
            cluster_center = series_data['extra_flow'].values.reshape(-1, 48).mean(axis=0)
            mean_extra_mag = series_data['extra_magnitude'].values.reshape(-1, 48).mean(axis=0)
            plt.plot(range(MAX_LEN), cluster_center, color=PALETTE_MAG[mag], linewidth=linewidth[1],
                     linestyle=linestyle_mag[mag], label=mag, zorder=1)

    # 设置坐标轴范围
    plt.xlim(0, MAX_LEN)
    plt.xticks(np.linspace(0, MAX_LEN, 5), range(0, 5))
    plt.xlabel("Post-event dispersal time (h)", fontsize=Fontsize, labelpad=1)
    plt.ylim(0, 3.5)
    # label设为公式的10^0等等
    plt.yticks(np.linspace(ylim_cols[0][0], ylim_cols[0][1], 4), ['$10^0$', '$10^1$', '$10^2$', '$10^3$'])
    plt.ylabel('Extra population flow', fontsize=Fontsize)
    # 显示图例，handlelength：图例标记的长度，默认2.0；handletextpad：图例标记和标签的距离，默认0.8；columnspacing：图例列间距，默认2.0
    plt.legend(ncol=2, fontsize=Fontsize,
               handlelength=2.5, handletextpad=0.4, columnspacing=1, loc='upper right')
    # 保存
    plt.savefig(FO + r'MainFig-4-3-1.png', dpi=DPI, transparent=True)
    plt.close()

    # Plot chart 2 (R1)：在相同额外出行量下比较峰值集中度（最大5分钟占全部额外出行的比例），替换原单位流量归一化图
    df_fig4 = pd.read_csv(FD + r'df_fig4_r1.csv').dropna(subset=['peak_share'])
    fig, ax = plt.subplots(1, 1, dpi=DPI)
    plt.subplots_adjust(top=0.95, bottom=0.20, right=0.95, left=0.15, hspace=0, wspace=0)
    # 设置图片大小
    fig.set_size_inches(width_inch / figure_number, width_inch / figure_number * 0.5)
    # 设置刻度线在内
    plt.tick_params(labelsize=Fontsize, direction='in', width=ax_linewidth, which='both')
    # 修改坐标轴的粗细
    ax.spines['bottom'].set_linewidth(ax_linewidth)
    ax.spines['left'].set_linewidth(ax_linewidth)
    ax.spines['top'].set_linewidth(ax_linewidth)
    ax.spines['right'].set_linewidth(ax_linewidth)
    # 绘图：散点 + 各组幂律拟合
    for mag in ['Small', 'Medium', 'Large', 'Mega']:
        data_now = df_fig4[df_fig4['magnitude_group'] == mag]
        x, y = data_now['extra_trips'].values, data_now['peak_share'].values
        plt.scatter(x, y, color=PALETTE_MAG[mag], s=3, alpha=0.6, linewidths=0, zorder=1)
        fit = stats.linregress(np.log10(x), np.log10(y))
        x_range = np.logspace(np.log10(x.min()), np.log10(x.max()), 50)
        plt.plot(x_range, np.power(10, fit.intercept) * x_range ** fit.slope, color=PALETTE_MAG[mag],
                 linewidth=linewidth[1], linestyle=linestyle_mag[mag], label=mag, zorder=2)
    plt.xscale('log')
    # 幂律拟合在双对数坐标下为直线，因此纵轴也用对数轴，刻度标为百分比
    plt.yscale('log')
    plt.ylim(0.006, 0.6)
    plt.yticks([0.01, 0.03, 0.1, 0.3], ['1%', '3%', '10%', '30%'])
    plt.minorticks_off()
    # 横轴从100开始（对应原图的0），刻度写成普通数字以保持与原图相同的边距
    plt.xlim(100, 1e5)
    plt.xticks([1e2, 1e3, 1e4], ['100', '1,000', '10,000'])
    plt.tick_params(axis='x', pad=2)
    plt.xlabel("Extra dispersal trips", fontsize=Fontsize, labelpad=0.5)
    plt.ylabel('Peak share', fontsize=Fontsize)
    # 纵轴标签与原图（Extra population flow）位置对齐
    ax.yaxis.set_label_coords(-0.1181, 0.5)
    # 显示图例，handlelength：图例标记的长度，默认2.0；handletextpad：图例标记和标签的距离，默认0.8；columnspacing：图例列间距，默认2.0
    plt.legend(ncol=2, fontsize=Fontsize,
               handlelength=2.5, handletextpad=0.4, columnspacing=1, loc='lower left')
    # 保存
    plt.savefig(FO + r'MainFig-4-3-2.png', dpi=DPI, transparent=True)
    plt.close()



def main_fig_5_1():
    """Figure 5 bottom row: mean modal-shift response of each venue group (grey: 25 threshold combinations)."""
    z = np.load(FD + 'fig5_curves.npz')
    tau = z['tau']
    order = [('H', 'CBD core area'), ('LP', 'Remote suburban area'), ('LC', 'Sub-center area')]
    i_neg, i_pos = int(np.argmin(np.abs(tau + 0.184))), int(np.argmin(np.abs(tau - 0.184)))
    idx = np.array([i for i in range(len(tau)) if i not in (i_neg, i_pos)])
    s, color = 10, ['seismic', '#00ff00', '#ffff00']
    xt, yt = np.arange(-0.1, 0.41, 0.1), np.arange(-0.04, 0.101, 0.02)
    for i, (k, title) in enumerate(order):
        fig, ax = plt.subplots(1, 1, dpi=DPI)
        plt.subplots_adjust(top=0.9, bottom=0.20, right=0.93, left=0.18, hspace=0, wspace=0)
        fig.set_size_inches(width_inch / 3, width_inch / 3)
        plt.tick_params(labelsize=Fontsize, direction='in', width=ax_linewidth)
        for sp in ax.spines.values():
            sp.set_linewidth(ax_linewidth)
        ax.axhline(0, color='0.75', lw=0.4, zorder=0)
        ax.axvline(0, color='0.75', lw=0.4, zorder=0)
        for c in z[k + '_schemes']:
            ax.plot(c[0], c[1], color='0.6', lw=0.3, alpha=0.5, zorder=1)
        d = z[k + '_mean']
        ax.scatter(d[0, idx], d[1, idx], c=tau[idx], cmap=color[0], vmin=-0.5, vmax=0.5, s=s, zorder=3)
        ax.scatter(d[0, i_neg], d[1, i_neg], c=color[1], marker='^', s=s * 2, edgecolor='k', linewidth=0.5, zorder=4)
        ax.scatter(d[0, i_pos], d[1, i_pos], c=color[2], marker='s', s=s * 2, edgecolor='k', linewidth=0.5, zorder=4)
        ax.set_xlim(-0.17, 0.47)
        ax.set_xticks(xt, ['%.0f' % v for v in xt * 100])
        ax.set_ylim(-0.045, 0.11)
        ax.set_yticks(yt, ['%.0f' % v for v in yt * 100])
        ax.set_xlabel('Relative peak intensity change (%)', fontsize=Fontsize)
        ax.set_ylabel('Relative radius change (%)', fontsize=Fontsize)
        ax.set_title(title, fontsize=Fontsize)
        plt.savefig(FO + 'MainFig-5-1-%d.png' % (i + 1), dpi=DPI, transparent=True)
        plt.close()


if __name__ == '__main__':
    import os
    HERE = os.path.dirname(os.path.abspath(__file__))
    FD = os.path.join(HERE, '..', 'results', 'figure_data') + os.sep
    FO = os.path.join(HERE, 'output') + os.sep
    os.makedirs(FO, exist_ok=True)
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Arial'
    plt.rcParams['mathtext.it'] = 'Arial:italic'
    width_inch = (21 - 3.18 * 2) / 2.54     # text width of the journal page (inch)
    Fontsize = 8
    ax_linewidth = 0.5
    DPI = 900
    for fn in [main_fig_1_3, main_fig_1_4, main_fig_2_1, main_fig_2_2, main_fig_2_3, main_fig_3_1, main_fig_3_2,
               main_fig_3_3, main_fig_4_1, main_fig_4_2, main_fig_4_3, main_fig_5_1]:
        fn()
        plt.close('all')
        print('done', fn.__name__)
