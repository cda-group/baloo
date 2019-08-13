import timeit
from collections import OrderedDict

import numpy as np

from baloo import Series
from baloo.weld import WeldObject


def generate_data(scale=1):
    np.random.seed(42)
    n = 1000

    data = OrderedDict((
        ('col1', np.random.randn(n * scale) * 17),
        ('col2', np.random.randn(n * scale) * 29),
        ('col3', np.random.randint(100, size=n * scale, dtype=np.int64)),
        ('col4', np.random.randint(200, size=n * scale, dtype=np.int32))
    ))

    return data


def pandas_pipeline(data):
    from pandas import DataFrame
    from numpy import exp

    df = DataFrame(data)

    df = df[(df['col1'] > 0) & (df['col2'] >= 10) & (df['col3'] < 30)]
    df['col5'] = (df['col1'] + df['col2']) * 0.1
    df['col6'] = df['col5'].apply(exp)

    df.groupby(['col2', 'col4']).var()


def baloo_pipeline(data):
    from baloo import DataFrame, exp

    a1 = WeldObject(None, None)
    a1.weld_code = "a"
    a2 = WeldObject(None, None)
    a2.weld_code = "b"

    data = OrderedDict((
        ('col1', Series(a1, dtype=np.dtype('float32'))),
        ('col2', Series(a2, dtype=np.dtype('int32'))),
    ))

    df = DataFrame(data)['col1']*2
    print(df.generate())

    df = DataFrame(data)['col2']*2
    print(df.generate())

    #  df = DataFrame(data)['col1'] * 2
    #  gen = df.generate()


    # df = DataFrame(data)

    # (df['col3'] * 2).evaluate()
    #
    # df = df[(df['col1'] > 0) & (df['col2'] >= 10) & (df['col3'] < 30)]
    # df['col5'] = (df['col1'] + df['col2']) * 0.1
    # df['col6'] = df['col5'].apply(exp)
    #
    # df.groupby(['col2', 'col4']).var().evaluate()


data_setup = """
from __main__ import generate_data

data = generate_data(scale=20000)"""

pandas_setup = "from __main__ import pandas_pipeline"
baloo_setup = "from __main__ import baloo_pipeline"

# TODO: this is not finished
if __name__ == '__main__':
    number = 1

    # pandas_time = timeit.timeit("pandas_pipeline(data)", setup=pandas_setup + data_setup, number=number)
    # print("Pandas: {time}".format(time=pandas_time / number))

    baloo_time = timeit.timeit("baloo_pipeline(data)", setup=baloo_setup + data_setup, number=number)
    print("Baloo: {time}".format(time=baloo_time / number))
