import time
import cProfile

# from self_play import self_play
from game import State

from dual_network import DN_INPUT_SHAPE
from math import sqrt
from tensorflow.keras.models import load_model
from pathlib import Path
import numpy as np

from CythonPredict import cpredict


def predict(model, state):
    # 推論のための入力データのシェイプの変換
    a, b, c = DN_INPUT_SHAPE
    x = np.array(state.pieces_array())
    x = x.reshape(c, a, b).transpose(1, 2, 0).reshape(1, a, b, c)

    # 推論
    y = model.predict(x, batch_size=1)
    # print(type(y))

    # 方策の取得
    policies = y[0][0][list(state.legal_actions())]  # 合法手のみ
    policies /= sum(policies) if sum(policies) else 1  # 合計1の確率分布に変換
    # print(type(policies))

    # 価値の取得
    value = y[1][0][0]
    # print(type(value))
    return policies, value


def preprepredict(model, state):
    start = time.time()
    for _ in range(300):
        predict(model, state)
    elapsed_time = time.time() - start
    print("Python:elapsed_time:{0}".format(elapsed_time) + "[sec]")

    start = time.time()
    for _ in range(300):
        cpredict(model, state)
    elapsed_time = time.time() - start
    print("Cython:elapsed_time:{0}".format(elapsed_time) + "[sec]")


def cythonPredict(model, state):
    for _ in range(100):
        cpredict(model, state)


if __name__ == "__main__":
    path = sorted(Path("./model").glob("*.h5"))[-1]
    model = load_model(str(path))

    state = State()

    # cProfile.run("preprepredict(model, state)", filename="main.prof")

    # predict(model, state)
    preprepredict(model, state)
