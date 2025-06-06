# -*- coding: utf-8 -*-
"""
网约车订单预测系统（严格网格匹配版）
修改版：预测早高峰总订单量（7:00-9:00合并）
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from keras.src.layers import ConvLSTM2D, Reshape, Conv2D
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import warnings
from datetime import time

warnings.filterwarnings("ignore")

# ====================== 参数配置 ======================
GRID_SIZE = 500  # 网格大小(米)
PEAK_HOURS = [7, 8, 9]  # 早晚高峰时段
PEAK_START = time(7, 0)  # 早晚高峰开始时间
PEAK_END = time(9, 0)  # 早晚高峰结束时间
DATA_PATH = '网约车数据2_WGS84.csv'
GEOJSON_PATH = '350200.geojson'
MIN_ORDERS = 0  # 网格最小订单量阈值
TRAIN_DAYS = 6  # 训练天数
PREDICT_DAY = 7  # 预测第7天
# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 1. 数据加载与过滤 ======================
def load_and_filter_data():
    print("正在加载并过滤订单数据...")
    try:
        df = pd.read_csv(
            DATA_PATH,
            parse_dates=['date'],
            usecols=['DEP_LONGITUDE_WGS84', 'DEP_LATITUDE_WGS84', 'date'],
            dtype={'DEP_LONGITUDE_WGS84': float, 'DEP_LATITUDE_WGS84': float}
        )

        # 数据清洗
        df = df.dropna(subset=['DEP_LONGITUDE_WGS84', 'DEP_LATITUDE_WGS84'])

        # 过滤早晚高峰数据
        df = df[(df['date'].dt.time >= PEAK_START) &
                (df['date'].dt.time <= PEAK_END)]

        if len(df) == 0:
            raise ValueError("没有找到该时段的订单数据")

        geometry = [Point(xy) for xy in zip(df['DEP_LONGITUDE_WGS84'], df['DEP_LATITUDE_WGS84'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

        print(f"成功加载 {len(gdf):,} 条早晚高峰订单数据")
        return gdf

    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        raise


# ====================== 2. 创建严格网格（添加行列索引） ======================
def create_strict_grid(admin_gdf):
    print("\n创建严格网格...")
    bounds = admin_gdf.total_bounds

    # 计算网格步长（WGS84坐标）
    x_step = GRID_SIZE / 111320  # 经度方向
    y_step = GRID_SIZE / 110540  # 纬度方向

    # 生成网格
    polygons = []
    x_coords = np.arange(bounds[0], bounds[2] + x_step, x_step)
    y_coords = np.arange(bounds[1], bounds[3] + y_step, y_step)

    # 添加行列索引
    rows = []
    cols = []
    for x_idx, x in enumerate(x_coords[:-1]):
        for y_idx, y in enumerate(y_coords[:-1]):
            polygons.append(Polygon([
                (x, y), (x + x_step, y),
                (x + x_step, y + y_step), (x, y + y_step)
            ]))
            rows.append(y_idx)  # 纬度方向为行
            cols.append(x_idx)  # 经度方向为列

    grid_gdf = gpd.GeoDataFrame({
        'geometry': polygons,
        'row': rows,  # 添加行索引
        'col': cols   # 添加列索引
    }, crs="EPSG:4326")

    grid_gdf['grid_id'] = grid_gdf.index

    # 裁剪到行政区域内
    grid_gdf = gpd.overlay(grid_gdf, admin_gdf, how='intersection')
    print(f"创建完成，网格形状: ({grid_gdf['row'].max()+1}x{grid_gdf['col'].max()+1})")
    return grid_gdf


# ====================== 3. 严格空间连接 ======================
def strict_spatial_join(orders_gdf, grid_gdf):
    print("\n执行严格空间连接...")
    joined = gpd.sjoin(
        orders_gdf,
        grid_gdf,
        how='inner',
        predicate='within'
    )

    # 添加时间特征
    joined['date_day'] = joined['date'].dt.normalize()
    joined['hour'] = joined['date'].dt.hour
    joined['is_peak'] = joined['hour'].isin(PEAK_HOURS)

    print(f"匹配到 {len(joined):,} 条订单（严格在网格内）")
    return joined


# ====================== 可视化历史数据（修正版） ======================
def visualize_history(joined_data, grid_gdf):
    print("\n生成7天早晚高峰订单分布图...")

    # 修正日期处理（确保格式统一）
    joined_data['date_day'] = pd.to_datetime(joined_data['date'].dt.date)

    # 按天统计总订单量
    daily_counts = joined_data.groupby(['date_day', 'grid_id']).size()
    daily_counts = daily_counts.reset_index(name='count')

    # 获取完整的7天日期范围
    date_range = pd.date_range(
        start=joined_data['date_day'].min(),
        periods=7,
        freq='D'
    )

    # 创建可视化
    fig, axes = plt.subplots(7, 1, figsize=(12, 24))
    plt.suptitle("早晚高峰订单分布（7:00-9:00）", y=0.95, fontsize=16)

    # 计算全局颜色标尺（排除异常值）
    global_max = daily_counts['count'].quantile(0.95)

    for i, day in enumerate(date_range):
        ax = axes[i]

        # 处理可能缺失的日期
        day_data = daily_counts[daily_counts['date_day'] == day]
        if len(day_data) == 0:
            day_data = pd.DataFrame({'grid_id': grid_gdf['grid_id'], 'count': 0})

        merged = grid_gdf.merge(day_data, on='grid_id', how='left').fillna({'count': 0})

        # 绘制热力图
        plot = merged.plot(
            column='count',
            cmap='RdYlGn_r',
            ax=ax,
            vmin=0,
            vmax=max(global_max, 1),  # 确保最小值有意义
            legend=False,
            edgecolor='lightgray',
            linewidth=0.2
        )

        # 添加日期和统计信息
        day_str = day.strftime('%Y-%m-%d')
        total = merged['count'].sum()
        ax.set_title(
            f"{day_str} | 总订单: {total:,} | 最大网格: {merged['count'].max()}",
            pad=10, fontsize=12
        )


    # 添加统一图例
    fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0, global_max), cmap='RdYlGn'),
        ax=axes, orientation='horizontal',
        label='订单量', shrink=0.6, aspect=40
    )

    plt.tight_layout()
    plt.savefig('7_days_distribution.png', dpi=300, bbox_inches='tight')
    print("7天订单分布图已保存为 7_days_distribution.png")


# ====================== 5. 准备ConvLSTM数据（修复行列索引访问） ======================
def prepare_conv_lstm_data(joined_data, grid_gdf):
    print("\n准备ConvLSTM训练数据...")

    # 统计每天各网格订单量
    daily_counts = joined_data.groupby(['date_day', 'grid_id']).size().reset_index(name='counts')

    # 获取7天的日期（按日期排序）
    unique_days = sorted(joined_data['date_day'].unique())[:7]
    if len(unique_days) < 7:
        raise ValueError("需要至少7天的数据，当前只有{}天".format(len(unique_days)))

    # 初始化空间矩阵 (时间×行×列×特征)
    max_row = grid_gdf['row'].max() + 1
    max_col = grid_gdf['col'].max() + 1
    spatial_data = np.zeros((7, max_row, max_col, 1))

    # 填充矩阵数据
    for day_idx, day in enumerate(unique_days):
        day_data = daily_counts[daily_counts['date_day'] == day]
        for _, row in day_data.iterrows():
            grid_info = grid_gdf[grid_gdf['grid_id'] == row['grid_id']].iloc[0]
            r, c = int(grid_info['row']), int(grid_info['col'])
            spatial_data[day_idx, r, c, 0] = row['counts']

    # 创建训练样本（前6天预测第7天）
    X_train = np.expand_dims(spatial_data[:6], axis=0)  # 形状 (1, 6, row, col, 1)
    y_train = np.expand_dims(spatial_data[6], axis=0)  # 形状 (1, row, col)

    # 创建有效掩码（基于所有日期的累计订单）
    valid_mask = (spatial_data.sum(axis=0) > MIN_ORDERS).any(axis=2)

    print(f"网格形状: {max_row}x{max_col} | 有效网格: {valid_mask.sum()}/{valid_mask.size}")
    return X_train, y_train, grid_gdf, valid_mask


# ====================== 6. ConvLSTM模型构建 ======================
def build_conv_lstm_model(input_shape):
    model = Sequential([
        # 第一层ConvLSTM（返回序列）
        ConvLSTM2D(
            filters=32,
            kernel_size=(3, 3),
            padding='same',
            return_sequences=True,
            input_shape=input_shape
        ),
        Dropout(0.3),

        # 第二层ConvLSTM（不返回序列）
        ConvLSTM2D(
            filters=16,
            kernel_size=(3, 3),
            padding='same',
            return_sequences=False
        ),
        Dropout(0.3),

        # 过渡层（确保输出形状匹配）
        Reshape((input_shape[1], input_shape[2], 16)),  # 16是上一层的filters数

        # 空间特征提取
        Conv2D(
            filters=8,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        ),

        # 输出层
        Conv2D(
            filters=1,  # 单通道输出
            kernel_size=(1, 1),
            padding='same',
            activation='softplus'
        )
    ])

    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model


# ====================== 7. 可视化预测结果（适配ConvLSTM输出） ======================
def visualize_conv_results(grid_gdf, predictions, actuals, valid_mask):
    """
    最终优化版可视化函数
    参数:
        xiamen_geojson_path: 厦门市GeoJSON文件路径
    """
    # 1. 加载厦门市边界数据
    xiamen_boundary = gpd.read_file(GEOJSON_PATH)

    # 2. 处理预测数据形状
    if predictions.ndim == 4:
        predictions = predictions[0]  # 去除批处理维度

    # 3. 确保实际值是numpy数组
    if not isinstance(actuals, np.ndarray):
        actuals = np.array(actuals)

    # 4. 创建结果DataFrame
    height, width = valid_mask.shape
    rows, cols = np.where(valid_mask)

    result_df = pd.DataFrame({
        'row': rows,
        'col': cols,
        'predicted': predictions[rows, cols, 0].flatten(),
        'actual': actuals[rows, cols].flatten()
    })

    # 5. 合并到原始网格
    merged = grid_gdf.merge(result_df, on=['row', 'col'], how='inner')

    # 6. 计算统计量
    pred_min, pred_max = float(merged['predicted'].min()), float(merged['predicted'].max())
    actual_min, actual_max = float(merged['actual'].min()), float(merged['actual'].max())

    print(f"预测值范围: {pred_min:.1f}~{pred_max:.1f}")
    print(f"实际值范围: {actual_min:.1f}~{actual_max:.1f}")

    # 7. 创建对比可视化
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    plt.suptitle('厦门市网约车早高峰订单量对比 (7:00-9:00)',
                 y=1.02, fontsize=14, fontweight='bold')

    # 统一配色方案
    cmap = 'coolwarm'

    # 动态调整颜色范围（使用98%分位数避免极端值影响）
    vmax = max(
        np.percentile(merged['predicted'], 98),
        np.percentile(merged['actual'], 98)
    )

    # 共用色条参数（隐藏数字）
    cbar_kwargs = {
        'shrink': 0.7,
        'label': '',  # 清空标签
        'ticks': []  # 空刻度列表隐藏数字
    }


    # 预测图
    xiamen_boundary.boundary.plot(
        ax=axes[0],
        color='black',
        linewidth=0.8,
        zorder=2  # 确保边界在最上层
    )
    merged.plot(
        column='predicted',
        cmap=cmap,
        ax=axes[0],
        markersize=12,
        vmin=0,
        vmax=vmax*0.5,
        legend=False,
        legend_kwds=cbar_kwargs
    )
    axes[0].set_title(f'预测订单量\n(总计: {int(merged["predicted"].sum())})',
                      pad=12, fontsize=12)
    axes[0].axis('off')

    # 实际图
    xiamen_boundary.boundary.plot(
        ax=axes[1],
        color='black',
        linewidth=0.8,
        zorder=2
    )
    merged.plot(
        column='actual',
        cmap=cmap,
        ax=axes[1],
        markersize=12,
        vmin=0,
        vmax=vmax*0.5,
        legend=True,
        legend_kwds=cbar_kwargs
    )
    axes[1].set_title(f'实际订单量\n(总计: {int(merged["actual"].sum())})',
                      pad=12, fontsize=12)
    axes[1].axis('off')

    # 调整布局
    plt.tight_layout()

    # 保存为透明背景的PNG
    plt.savefig('xiamen_ridehailing_comparison.png',
                dpi=300,
                bbox_inches='tight',
                transparent=False)
    plt.show()

# ====================== 主程序 ======================
if __name__ == "__main__":
    try:
        # 1. 加载数据
        admin_gdf = gpd.read_file(GEOJSON_PATH)
        orders_gdf = load_and_filter_data()

        # 2. 创建网格
        grid_gdf = create_strict_grid(admin_gdf)

        # 3. 空间连接
        joined_data = strict_spatial_join(orders_gdf, grid_gdf)

        # 4. 准备ConvLSTM数据（替换原prepare_sequences）
        X_train, y_train, grid_gdf, valid_mask = prepare_conv_lstm_data(joined_data, grid_gdf)

        # 5. 训练ConvLSTM模型
        print("\n训练ConvLSTM模型中...")
        model = build_conv_lstm_model(X_train.shape[1:])
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=1,  # 小批量适合ConvLSTM
            validation_split=0,
            verbose=1
        )

        # 6. 预测与可视化
        print("\n生成预测结果...")
        predictions = model.predict(X_train[-1:])  # 用前6天预测第7天
        visualize_conv_results(
            grid_gdf=grid_gdf,
            predictions=model.predict(X_train[-1:]),  # ConvLSTM输出 (1, h, w, 1)
            actuals=y_train[0],  # 实际值 (h, w)
            valid_mask=valid_mask
        )

        # 保存模型（添加在model.fit之后）
        model.save('my_conv_lstm_model.h5')
        print("模型已保存为 my_conv_lstm_model.h5")

    except Exception as e:
        print(f"\n程序出错: {str(e)}")
        import traceback

        traceback.print_exc()