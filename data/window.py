import numpy as np
from numpy.lib.stride_tricks import as_strided

def norm_shape(shape):
    """
    把数组形状统一成元组表示，为了能让shape参与运算
    5 -> (5,)
    [2,1] -> 
    但是实际上numpy.shape就是元组表示
    Args:
        shape (int/tuple): 表示形状的整数或元组

    Raises:
        TypeError: 类型不正确

    Returns:
        tuple: 元组的形状表示格式
    """
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass
    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')


# 传值的时候是sliding_window(data,(窗口大小,传感器通道数),(窗口步长,1))
# 因为滑动窗口是在每个传感器通道上滑动的，如果传入（窗口大小，1）就变成只有一个传感器通道
def sliding_window(data, sliding_window_length, sliding_window_step):
    if sliding_window_step is None:
        # 默认窗口是没有
        sliding_window_length = sliding_window_step

    sliding_window_length = norm_shape(sliding_window_length)  # (sliding_window_length, num_channels)
    sliding_window_step = norm_shape(sliding_window_step)  # (sliding_window_step, 1)

    # 转为np数组，但是这样的话，前面转为元组的操作不就没有卵用了吗？
    # 然而并不是，一个数转np数组，结果还是一个数，但元组(数,)转np数组，结果是一个[]
    # 都转成数组是为了运算方便（因为numpy运算方便）
    sliding_window_length = np.array(sliding_window_length)
    sliding_window_step = np.array(sliding_window_step)
    data_shape = np.array(data.shape)  # np.shape就是元组，直接转n变数组 [num_instance, num_channels]

    # 保证三者的维度相同
    ls = [len(data_shape), len(sliding_window_length), len(sliding_window_step)]
    # 集合只能存不同元素，所以set([1,1,1]) = {1}
    if 1 != len(set(ls)):
        raise ValueError('窗口大小、步长、数据维度必须相同')

    # 保证窗口大小的每一维都小于数据大小
    if np.any(sliding_window_length > data_shape):
        raise ValueError('窗口维度必须小于数据维度')

    # 计算滑动窗口数的公式
    num_windows = ((data_shape - sliding_window_length) // sliding_window_step) + 1
    # newshape=[num_windows,1,sliding_window_length,num_channels]
    newshape = norm_shape(num_windows) + norm_shape(sliding_window_length)

    # np.array(data.strides) = [num_channels * 字节数（一个数据），字节数（一个数据）]
    temp = np.array(data.strides) * sliding_window_step
    # 可能是需要这样的数据格式
    newstrides = norm_shape(temp) + data.strides
    # 1. x是需要划分窗口的数组
    # 2. shape是新数组的形状[滑动窗口数，滑动窗口大小] 类型是tuple
    # 3. stride是步幅，然而是内存上的步幅（数组），单位是字节数 类型是tuple
    # strided=[num_windows,1,sliding_window_length,num_channels]
    # stride是一个四维数组,stride[0]是一个三维数组，但是每个三维数组里都只有一个二维数组
    # 调用stride[i][0]即调用了第i个滑动窗口
    strided = as_strided(x=data, shape=newshape, strides=newstrides)
    # 移除strided大小为1的维度
    if len(strided.shape) > 2:
        return strided.reshape(num_windows[0], -1, data.shape[1])
    else:
        return strided


# 数据分割和重新塑形

def get_sliding_window(data_x, data_y, sliding_window_length, sliding_window_step):
    """
    Returns:
        data_x: [滑动窗口1，滑动窗口2，...] 
        data_y: [[窗口1的标签]，[窗口2的标签]，...] -> [[1], [11]，...] 
    """
    data_x = sliding_window(data_x, (sliding_window_length, data_x.shape[1]), (sliding_window_step, 1))
    # 一个窗口有sliding_window_length个数据和标签
    # 现在把每个窗口的最后一个标签作为整个窗口的标签
    # i=[sliding_window_length,1]
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, sliding_window_length, sliding_window_step)])
    return data_x.astype(np.float64), data_y.astype(np.int64)