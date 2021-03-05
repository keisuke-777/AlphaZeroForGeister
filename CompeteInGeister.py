# ゲームを管理

import random
import math

# ゲームの状態
class State:
    # 初期化
    def __init__(self, pieces=None, enemy_pieces=None, depth=0):

        self.is_goal = False

        # 駒の配置
        if pieces != None:
            self.pieces = pieces
        else:
            self.pieces = [0] * 36

        if enemy_pieces != None:
            self.enemy_pieces = enemy_pieces
        else:
            self.enemy_pieces = [0] * 36

        # ターンの深さ(ターン数)
        self.depth = depth

        # 駒の初期配置
        if pieces == None or enemy_pieces == None:
            # 青4赤4z
            piece_list = [1, 1, 1, 1, 2, 2, 2, 2]

            random.shuffle(piece_list)  # 配置をランダムに
            self.pieces[25] = piece_list[0]
            self.pieces[26] = piece_list[1]
            self.pieces[27] = piece_list[2]
            self.pieces[28] = piece_list[3]
            self.pieces[31] = piece_list[4]
            self.pieces[32] = piece_list[5]
            self.pieces[33] = piece_list[6]
            self.pieces[34] = piece_list[7]

            random.shuffle(piece_list)  # 配置をランダムに
            self.enemy_pieces[25] = piece_list[0]
            self.enemy_pieces[26] = piece_list[1]
            self.enemy_pieces[27] = piece_list[2]
            self.enemy_pieces[28] = piece_list[3]
            self.enemy_pieces[31] = piece_list[4]
            self.enemy_pieces[32] = piece_list[5]
            self.enemy_pieces[33] = piece_list[6]
            self.enemy_pieces[34] = piece_list[7]

    # 負けかどうか
    def is_lose(self):
        if not any(elem == 1 for elem in self.pieces):  # 自分の青駒が存在しないなら負け
            # print("青喰い")
            return True
        if not any(elem == 2 for elem in self.enemy_pieces):  # 敵の赤駒が存在しない(全部取っちゃった)なら負け
            # print("赤喰い")
            return True
        # 前の手でゴールされてたらis_goalがTrueになってる(ような仕様にする)
        if self.is_goal:
            # print("ゴール")
            return True
        return False

    # 引き分けかどうか
    def is_draw(self):
        return self.depth >= 300  # 300手

    # ゲーム終了かどうか
    def is_done(self):
        return self.is_lose() or self.is_draw()

    # デュアルネットワークの入力の2次元配列の取得
    def pieces_array(self):
        # プレイヤー毎のデュアルネットワークの入力の2次元配列の取得
        def pieces_array_of(pieces):
            table_list = []
            # 青駒(1)→赤駒(2)の順に取得
            for j in range(1, 3):
                table = [0] * 36
                table_list.append(table)
                # appendは参照渡しなのでtable書き換えればtable_listも書き換わる
                for i in range(36):
                    if pieces[i] == j:
                        table[i] = 1

            return table_list

        # デュアルネットワークの入力の2次元配列の取得(自分と敵両方)
        return [pieces_array_of(self.pieces), pieces_array_of(self.enemy_pieces)]

    # position->0~35
    # direction->下:0,左:1,上:2,右:3

    # 駒の移動元と移動方向を行動に変換
    def position_to_action(self, position, direction):
        return position * 4 + direction

    # 行動を駒の移動元と移動方向に変換
    def action_to_position(self, action):
        return (int(action / 4), action % 4)  # position,direction

    # 合法手のリストの取得
    def legal_actions(self):
        actions = []
        for p in range(36):
            # 駒の存在確認
            if self.pieces[p] != 0:
                # 存在するなら駒の位置を渡して、その駒の取れる行動をactionsに追加
                actions.extend(self.legal_actions_pos(p))
        # 青駒のゴール行動は例外的に合法手リストに追加
        if self.pieces[0] == 1:
            actions.extend([2])  # 0*4 + 2
        if self.pieces[5] == 1:
            actions.extend([22])  # 5*4 + 2
        return actions

    # 駒ごと(駒1つに着目した)の合法手のリストの取得
    def legal_actions_pos(self, position):
        actions = []
        x = position % 6
        y = int(position / 6)
        # 下左上右の順に行動できるか検証し、できるならactionに追加
        # ちなみにand演算子は左の値を評価して右の値を返すか決める(左の値がTrue系でなければ右の値は無視する)ので、はみ出し参照してIndexErrorにはならない(&だとなる)
        if y != 5 and self.pieces[position + 6] == 0:  # 下端でない and 下に自分の駒がいない
            actions.append(self.position_to_action(position, 0))
        if x != 0 and self.pieces[position - 1] == 0:  # 左端でない and 左に自分の駒がいない
            actions.append(self.position_to_action(position, 1))
        if y != 0 and self.pieces[position - 6] == 0:  # 上端でない and 上に自分の駒がいない
            actions.append(self.position_to_action(position, 2))
        if x != 5 and self.pieces[position + 1] == 0:  # 右端でない and 右に自分の駒がいない
            actions.append(self.position_to_action(position, 3))
        # 青駒のゴール行動の可否は1ターンに1度だけ判定すれば良いので、例外的にlegal_actionsで処理する(ここでは処理しない)
        return actions

    # 次の状態の取得
    def next(self, action):
        # 次の状態の作成
        state = State(self.pieces.copy(), self.enemy_pieces.copy(), self.depth + 1)

        # position_bef->移動前の駒の位置、position_aft->移動後の駒の位置
        # 行動を(移動元, 移動方向)に変換
        position_bef, direction = self.action_to_position(action)

        # 合法手がくると仮定
        # 駒の移動(後ろに動くことは少ないかな？ + if文そんなに踏ませたくないな と思ったので判定を左右下上の順番にしてるけど意味あるのかは不明)
        if direction == 0:  # 下
            position_aft = position_bef + 6
        elif direction == 1:  # 左
            position_aft = position_bef - 1
        elif direction == 2:  # 上
            if position_bef == 0 or position_bef == 5:  # 0と5の上行動はゴール処理なので先に弾く
                state.is_goal = True
                position_aft = position_bef  # position_befを入れて駒の場所を動かさない(勝敗は決しているので下手に動かさない方が良いと考えた)
            else:
                position_aft = position_bef - 6
        elif direction == 3:  # 右
            position_aft = position_bef + 1
        else:
            print("error関数名:next")

        # 実際に駒移動
        state.pieces[position_aft] = state.pieces[position_bef]
        state.pieces[position_bef] = 0

        # 移動先に敵駒が存在した場合は取る(比較と値入れどっちが早いかあとで調べて最適化したい)
        # piecesとenemy_piecesを対応させるには値をひっくり返す必要がある(要素のインデックスは0~35だから、 n->35-n でひっくり返せる)
        if state.enemy_pieces[35 - position_aft] != 0:
            state.enemy_pieces[35 - position_aft] = 0

        # 駒の交代(ターンプレイヤが切り替わるため)(pieces <-> enemy_pieces)
        tmp = state.pieces
        state.pieces = state.enemy_pieces
        state.enemy_pieces = tmp
        return state

    # 先手かどうか
    def is_first_player(self):
        return self.depth % 2 == 0

    # どちらのプレイヤーが勝利したか判定
    # 勝敗決定時にやるだけだからそんな高速化しないで大丈夫
    def winner_checker(self, win_player):
        if self.is_goal:
            win_player[self.goal_player] += 1
            return

        board = [0] * 36
        if self.depth % 2 == 0:
            my_p = self.pieces.copy()
            rev_ep = list(reversed(self.enemy_pieces))
            for i in range(36):
                board[i] = my_p[i] - rev_ep[i]
        else:
            my_p = list(reversed(self.pieces))
            rev_ep = self.enemy_pieces.copy()
            for i in range(36):
                board[i] = rev_ep[i] - my_p[i]

        if not any(elem == 1 for elem in board):  # 先手の青駒が存在しない
            win_player[1] += 1  # 後手の勝ち
        elif not any(elem == 2 for elem in board):  # 先手の赤駒が存在しない
            win_player[0] += 1  # 先手の勝ち
        elif not any(elem == -1 for elem in board):  # 後手の青駒が存在しない
            win_player[0] += 1  # 先手の勝ち
        elif not any(elem == -2 for elem in board):  # 後手の赤駒が存在しない
            win_player[1] += 1  # 後手の勝ち
        else:
            pass

    # 文字列表示
    def __str__(self):
        row = "|{}|{}|{}|{}|{}|{}|"
        hr = "\n-------------------------------\n"

        # 1つのボードに味方の駒と敵の駒を集める
        board = [0] * 36
        if self.depth % 2 == 0:
            my_p = self.pieces.copy()
            rev_ep = list(reversed(self.enemy_pieces))
            for i in range(36):
                board[i] = my_p[i] - rev_ep[i]
        else:
            my_p = list(reversed(self.pieces))
            rev_ep = self.enemy_pieces.copy()
            for i in range(36):
                board[i] = rev_ep[i] - my_p[i]

        board_essence = []
        for i in board:
            if i == 1:
                board_essence.append("自青")
            elif i == 2:
                board_essence.append("自赤")
            elif i == -1:
                board_essence.append("敵青")
            elif i == -2:
                board_essence.append("敵赤")
            else:
                board_essence.append("　　")

        str = (
            hr + row + hr + row + hr + row + hr + row + hr + row + hr + row + hr
        ).format(*board_essence)
        return str


# ランダムで行動選択
def random_action(state):
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0, len(legal_actions) - 1)]


# 人間に行動を選択させる
def human_player_action(state):
    # 盤面を表示
    print(state)

    # 入力を待つ(受ける)
    before_move_place = int(input("Please enter to move piece (左上~右下にかけて0~35) : "))
    direction = int(input("direction (下0 左1 上2 右3) : "))
    move = state.position_to_action(before_move_place, direction)

    # 合法手か確認
    legal_actions = state.legal_actions()
    if any(elem == move for elem in legal_actions):
        return move

    # エラー処理(デバッグでしか使わんから適当)
    print("非合法手が選択された為、ランダムに行動しました")
    return legal_actions[random.randint(0, len(legal_actions) - 1)]


import GuessEnemyPiece
import numpy as np
import itertools
import time
from pv_mcts import predict, pv_mcts_scores, pv_mcts_action
from dual_network import DN_INPUT_SHAPE
from pathlib import Path
from tensorflow.keras.models import load_model

# 価値の高い行動を愚直に選択し続ける
def predict_action(model, state):
    policies, value = predict(model, state)
    legal_actions = state.legal_actions()
    best_policie_action_index = np.argmax(policies)
    return legal_actions[best_policie_action_index]


# モンテカルロ木探索の行動選択
def mcts_action(state):
    # モンテカルロ木探索のノード
    class node:
        # 初期化
        def __init__(self, state):
            self.state = state  # 状態
            self.w = 0  # 累計価値
            self.n = 0  # 試行回数
            self.child_nodes = None  # 子ノード群

        # 評価
        def evaluate(self):
            # ゲーム終了時
            if self.state.is_done():
                # 勝敗結果で価値を取得
                value = -1 if self.state.is_lose() else 0  # 負けは-1、引き分けは0

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value

            # 子ノードが存在しない時
            if not self.child_nodes:
                # プレイアウトで価値を取得
                value = playout(self.state)

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1

                # 子ノードの展開
                if self.n == 10:
                    self.expand()
                return value

            # 子ノードが存在する時
            else:
                # UCB1が最大の子ノードの評価で価値を取得
                value = -self.next_child_node().evaluate()

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value

        # 子ノードの展開
        def expand(self):
            legal_actions = self.state.legal_actions()
            self.child_nodes = []
            for action in legal_actions:
                self.child_nodes.append(node(self.state.next(action)))

        # UCB1が最大の子ノードを取得
        def next_child_node(self):
            # 試行回数nが0の子ノードを返す
            for child_node in self.child_nodes:
                if child_node.n == 0:
                    return child_node

            # UCB1の計算
            t = 0
            for c in self.child_nodes:
                t += c.n
            ucb1_values = []
            for child_node in self.child_nodes:
                ucb1_values.append(
                    -child_node.w / child_node.n
                    + 2 * (2 * math.log(t) / child_node.n) ** 0.5
                )

            # UCB1が最大の子ノードを返す
            return self.child_nodes[argmax(ucb1_values)]

    # ルートノードの生成
    root_node = node(state)
    root_node.expand()

    # ルートノードを評価 (rangeを変化させると評価回数を変化させられる)
    for _ in range(100):
        root_node.evaluate()

    # 試行回数の最大値を持つ行動を返す
    legal_actions = state.legal_actions()
    n_list = []
    for c in root_node.child_nodes:
        n_list.append(c.n)
    return legal_actions[argmax(n_list)]


# ゲームの終端までシミュレート
def playout(state):
    if state.is_lose():
        return -1
    if state.is_draw():
        return 0
    return -playout(state.next(random_action(state)))


# 最大値のインデックスを返す
def argmax(v_list):
    return v_list.index(max(v_list))


# logのやつの戦闘データ一気にとる
drow_count = 0


def evaluate_GeisterLog():
    global drow_count
    for i in range(10, 20):
        drow_count = 0
        model_pass_str = "./AllLogFile/" + str(i) + ".h5"
        model = load_model(model_pass_str)
        win_player = [0, 0]
        for _ in range(100):
            # 直前の行動を保管
            just_before_action_num = 0
            # 状態の生成
            state = State()

            predict_mcts = pv_mcts_action(model)

            # ゲーム終了までのループ
            while True:
                # ゲーム終了時
                if state.is_done():
                    break

                # 次の状態の取得
                if state.depth % 2 == 0:
                    just_before_action_num = random_action(state)  # ランダム
                    # just_before_action_num = mcts_action(state)  # モンテカルロ木探索
                    # just_before_action_num = no_cheat_mcts_action(
                    #     state
                    # )  # ズルなしモンテカルロ木探索
                    # just_before_action_num = no_cheat_and_fix_mcts_action(
                    #     state
                    # )  # ズルなし固定なしモンテカルロ木探索
                    if just_before_action_num == 2 or just_before_action_num == 22:
                        state.is_goal = True
                        state.goal_player = 0
                        break
                    state = state.next(just_before_action_num)
                else:
                    # just_before_enemy_action_num = just_before_action_num
                    # guess_player_action = GuessEnemyPiece.guess_enemy_piece_player_for_debug(
                    #     model, ii_state, just_before_enemy_action_num
                    # )
                    # just_before_action_num = guess_player_action

                    # just_before_action_num = predict_action(model, state)
                    just_before_action_num = predict_mcts(state)  # 学習データを使ったMCTS
                    if just_before_action_num == 2 or just_before_action_num == 22:
                        state.is_goal = True
                        state.goal_player = 1
                        break
                    state = state.next(just_before_action_num)
            state.winner_checker(win_player)
            print(win_player)
        print("[log_player,mcts]:", win_player)
    K.clear_session()
    del model


def main():
    # 状態の生成
    state = State()

    # GuessEnemyPieceに必要な処理
    path = sorted(Path("./model").glob("*.h5"))[-1]
    model = load_model(str(path))
    ii_state = GuessEnemyPiece.II_State({8, 9, 10, 11})

    # 直前の行動を保管
    just_before_action_num = 0

    # ゲーム終了までのループ
    while True:
        # ゲーム終了時
        if state.is_done():
            # print("ゲーム終了:ターン数", state.depth)

            # if state.is_lose():
            #     if state.depth % 2 == 0:
            #         print("敗北")
            #     else:
            #         print("勝利or引き分け")
            # else:
            #     if state.depth % 2 == 1:
            #         print("勝利or引き分け")
            #     else:
            #         print("敗北")
            break

        # 次の状態の取得
        if state.depth % 2 == 1:
            just_before_enemy_action_num = just_before_action_num
            guess_player_action = GuessEnemyPiece.guess_enemy_piece_player_for_debug(
                model, ii_state, just_before_enemy_action_num
            )
            just_before_action_num = guess_player_action
            # print("自作AIの行動番号", just_before_action_num)
            state = state.next(just_before_action_num)

            just_before_action_num = random_action(state)
            # print("敵(ランダムAI)の行動番号", just_before_action_num)
            state = state.next(just_before_action_num)
        else:
            just_before_action_num = random_action(state)
            # print("ランダムAIの行動番号", just_before_action_num)
            state = state.next(just_before_action_num)

        # 文字列表示
        # print("depth", state.depth)
        print(state)


# 動作確認
if __name__ == "__main__":
    # main()
    evaluate_GeisterLog()
