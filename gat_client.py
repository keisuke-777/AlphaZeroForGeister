import socket
import numpy as np
import itertools
import time
from game import State
from pv_mcts import predict
from dual_network import DN_INPUT_SHAPE
from pathlib import Path
from tensorflow.keras.models import load_model

gamma = 0.9

# おそらく不完全情報ガイスター(のstateのみ？)を定義してそれを更新して管理した方がよさげ
# 不完全情報ガイスターの盤面情報及びそれらの推測値
class II_State:

    # クラス変数で駒順を定義
    piece_name = [
        "h",
        "g",
        "f",
        "e",
        "d",
        "c",
        "b",
        "a",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
    ]

    # 初期化
    def __init__(
        self,
        real_my_piece_blue_set,
        all_piece=None,
        enemy_estimated_num=None,
        enemy_piece_list=None,
        my_piece_list=None,
        living_piece_color=None,
    ):
        #  全ての駒(hgfedcbaABCDEFGHの順になっている)
        # 敵駒0~7,自駒8~15
        if all_piece == None:
            # numpyは基本的に型指定しない方が早い(指定すると裏で余計な処理するっぽい)
            self.all_piece = np.zeros(16, dtype=np.int16)
            # 初期配置を代入(各値は座標を示す)(脱出が88、死亡が99)
            # 0~7は敵駒, 8~15は自駒
            self.all_piece[0] = 1
            self.all_piece[1] = 2
            self.all_piece[2] = 3
            self.all_piece[3] = 4
            self.all_piece[4] = 7
            self.all_piece[5] = 8
            self.all_piece[6] = 9
            self.all_piece[7] = 10

            self.all_piece[8] = 25
            self.all_piece[9] = 26
            self.all_piece[10] = 27
            self.all_piece[11] = 28
            self.all_piece[12] = 31
            self.all_piece[13] = 32
            self.all_piece[14] = 33
            self.all_piece[15] = 34
        else:
            self.all_piece = all_piece

        if enemy_piece_list == None:
            self.enemy_piece_list = [0, 1, 2, 3, 4, 5, 6, 7]
        else:
            self.enemy_piece_list = enemy_piece_list

        if my_piece_list == None:
            self.my_piece_list = [8, 9, 10, 11, 12, 13, 14, 15]
        else:
            self.my_piece_list = my_piece_list

        # real_my_piece_blue_setは自分の青駒のIDのセット(引数必須)
        self.real_my_piece_blue_set = real_my_piece_blue_set
        self.real_my_piece_red_set = (
            set(self.my_piece_list) - self.real_my_piece_blue_set
        )

        # {敵青, 敵赤, 自青,　自赤}
        if living_piece_color == None:
            self.living_piece_color = [4, 4, 4, 4]
        else:
            self.living_piece_color = living_piece_color

        # [[推測値A,(パターンAの青駒のtuple表現)],[推測値B,(パターンBの青駒のtuple表現),...]
        if enemy_estimated_num == None:
            # 盤面の推測値を作成(大きい程青らしく、小さい程赤らしい)
            self.enemy_estimated_num = []
            for enemy_blue in itertools.combinations(
                set(self.enemy_piece_list), self.living_piece_color[0]
            ):
                self.enemy_estimated_num.append([0, enemy_blue])
        else:
            self.enemy_estimated_num = enemy_estimated_num

    # 合法手のリストの取得
    # NNはactionを与えると事前に学習した方策を返す。
    # 赤のゴール(非合法なので知らない手)を与えると、そこを0にして返してくれるはず(エラーは吐かないはず？？？)
    def legal_actions(self):
        actions = []

        # リストに自分の駒を全て追加
        piece_coordinate_array = np.array([0] * 8)
        index = 0
        for i in range(8, 16):
            piece_coordinate_array[index] = self.all_piece[i]
            index += 1
        np.sort(piece_coordinate_array)

        print("my:self.all_piece", piece_coordinate_array)

        for piece_coordinate in piece_coordinate_array:
            # 88以上は行動できないので省く(0~35)
            if piece_coordinate < 36:
                actions.extend(
                    self.piece_coordinate_to_actions(
                        piece_coordinate, piece_coordinate_array
                    )
                )
            # 0と5はゴールの選択肢を追加(赤駒でも問答無用)
            if piece_coordinate == 0:
                actions.extend([2])  # 0*4 + 2
            if piece_coordinate == 5:
                actions.extend([22])  # 5*4 + 2
        return actions

    # 相手目線の合法手のリストを返す
    def enemy_legal_actions(self):
        actions = []
        piece_coordinate_array = np.array([0] * 8)
        index = 0
        for i in range(0, 8):
            if self.all_piece[i] < 36:
                piece_coordinate_array[index] = 35 - self.all_piece[i]
            else:
                piece_coordinate_array[index] = 99
            index += 1
        np.sort(piece_coordinate_array)

        print("enemy:self.all_piece", piece_coordinate_array)

        for piece_coordinate in piece_coordinate_array:
            # 88以上は行動できないので省く(0~35)
            if piece_coordinate < 36:
                actions.extend(
                    self.piece_coordinate_to_actions(
                        piece_coordinate, piece_coordinate_array
                    )
                )
            # 0と5はゴールの選択肢を追加(赤駒でも問答無用)
            if piece_coordinate == 0:
                actions.extend([2])  # 0*4 + 2
            if piece_coordinate == 5:
                actions.extend([22])  # 5*4 + 2
        return actions

    # 駒の移動元と移動方向を行動に変換
    def position_to_action(self, position, direction):
        return position * 4 + direction

    # ある駒の座標(piece_coordinate)から，その駒のとりうる行動を求める
    def piece_coordinate_to_actions(self, piece_coordinate, piece_coordinate_array):
        actions = []
        x = piece_coordinate % 6
        y = int(piece_coordinate / 6)

        if y != 5 and not np.any(piece_coordinate_array == (piece_coordinate + 6)):  # 下
            actions.append(self.position_to_action(piece_coordinate, 0))
        if x != 0 and not np.any(piece_coordinate_array == (piece_coordinate - 1)):  # 左
            actions.append(self.position_to_action(piece_coordinate, 1))
        if y != 0 and not np.any(piece_coordinate_array == (piece_coordinate - 6)):  # 上
            actions.append(self.position_to_action(piece_coordinate, 2))
        if x != 5 and not np.any(piece_coordinate_array == (piece_coordinate + 1)):  # 右
            actions.append(self.position_to_action(piece_coordinate, 3))

        return actions

    # 駒ごと(駒1つに着目した)の合法手のリストの取得
    def legal_actions_pos(self, position, piece_index_list):
        piece_list = []
        for piece_index in piece_index_list:
            piece_list.append(self.all_piece[piece_index])

        actions = []
        x = position % 6
        y = int(position / 6)
        # 下左上右の順に行動できるか検証し、できるならactionに追加
        # ちなみにand演算子は左の値を評価して右の値を返すか決める(左の値がTrue系でなければ右の値は無視する)ので、はみ出し参照してIndexErrorにはならない(&だとなる)
        if y != 5 and (position + 6) not in piece_list:  # 下端でない and 下に自分の駒がいない
            actions.append(self.position_to_action(position, 0))
        if x != 0 and (position - 1) not in piece_list:  # 左端でない and 左に自分の駒がいない
            actions.append(self.position_to_action(position, 1))
        if y != 0 and (position - 6) not in piece_list:  # 上端でない and 上に自分の駒がいない
            actions.append(self.position_to_action(position, 2))
        if x != 5 and (position + 1) not in piece_list:  # 右端でない and 右に自分の駒がいない
            actions.append(self.position_to_action(position, 3))
        # 青駒のゴール行動の可否は1ターンに1度だけ判定すれば良いので、例外的にlegal_actionsで処理する(ここでは処理しない)
        return actions

    # この行動で相手の駒を殺すかどうか(殺すかどうか, 殺す駒のidのインデックス(ii_state.piece_nameに対応))
    def this_action_will_kill(self, action_num):
        coordinate = action_to_coordinate(action_num)

        # 移動後のマスに駒が存在しないならFalse
        if not np.any(self.all_piece == coordinate[1]):
            return False, None

        id_index = np.where(self.all_piece == coordinate[1])[0][0]
        # 移動後のマスに自分の駒が存在する(本来ありえないが，ゴール行動の場合そうなるように実装している)
        if id_index > 7:
            return False, None

        # 移動後のマスに敵駒が存在する
        return True, id_index

    # 自分の行動を受けて、次の状態に遷移(移動先に駒が存在しない場合のみ実行)
    def next(self, action_num):
        coordinate_before, coordinate_after = action_to_coordinate(action_num)
        move_piece_index = np.where(self.all_piece == coordinate_before)[0][0]
        self.all_piece[move_piece_index] = coordinate_after  # 駒の移動

    # 相手の駒を殺す際のnext(移動先に駒が存在する場合のみ実行)
    def kill_next(self, action_num, kill_id_index, now_tcp_str):
        coordinate_before, coordinate_after = action_to_coordinate(action_num)
        move_piece_index = np.where(self.all_piece == coordinate_before)[0][0]

        # 移動先に存在する駒を殺す
        if np.any(self.all_piece == coordinate_after):
            # ex)MOV?14R24R34R44R15B25B35B45B41u31u21u11u40u30u20u10u\r\n
            # 28→4,29→1,30→uとなっており，abcdefghの順で並んでいる
            # uの部分に駒の色が表示されるため，ここを見る
            color_char = now_tcp_str[30 + (7 - kill_id_index) * 3]
            if color_char == "b":
                color_is_blue = True
            elif color_char == "r":
                color_is_blue = False
            else:
                print("error:invalid color_char", color_char)

            reduce_pattern(kill_id_index, color_is_blue, self)
        else:
            print("error:killする駒が見つかりません")

        self.all_piece[move_piece_index] = coordinate_after  # 駒の移動

    # 相手の行動(移動前と移動後の座標)を受けて、ガイスターの盤面を更新(駒が死んだ場合の処理もここで行う)
    def enemy_next(self, before_coordinate, now_coordinate):
        kill = np.any(self.all_piece == now_coordinate)

        # 敵駒がkillしていたら死んだ駒の処理を行う(99は死んだ駒)
        if kill:
            dead_piece_ID = np.where(self.all_piece == now_coordinate)[0][0]
            color_is_blue = np.any(self.real_my_piece_blue_set == dead_piece_ID)
            print(dead_piece_ID, color_is_blue)
            reduce_pattern(dead_piece_ID, color_is_blue, self)

        # 行動前の座標を行動後の座標に変更する
        self.all_piece[
            np.where(self.all_piece == before_coordinate)[0][0]
        ] = now_coordinate

    # ボードの文字列表示
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


# 行動番号を駒の移動元と移動方向に変換
def action_to_position(action_num):
    return (int(action_num / 4), action_num % 4)  # position,direction


# 行動番号を移動前の座標と移動後の座標に変換
def action_to_coordinate(action_num):
    coordinate_before, direction = action_to_position(action_num)
    if direction == 0:  # 下
        coordinate_after = coordinate_before + 6
    elif direction == 1:  # 左
        coordinate_after = coordinate_before - 1
    elif direction == 3:  # 右
        coordinate_after = coordinate_before + 1
    elif direction == 2:  # 上
        if coordinate_before == 0 or coordinate_before == 5:  # 0と5の上行動はゴール処理なので弾く
            coordinate_after = coordinate_before  # coordinate_beforeを入れて駒の場所を動かさない(勝敗は決しているので下手に動かさない方が良い(多分))
        else:
            coordinate_after = coordinate_before - 6
    else:
        print("ERROR:action_to_coordinate(illegal action_num)")
    return coordinate_before, coordinate_after


# 移動前の座標と方向番号から行動番号を算出
def position_to_action(position, direction):
    return position * 4 + direction


# 移動前と移動後の座標から相手の行動番号を算出
def calculate_enemy_action_number_from_coordinate(before_coordinate, now_coordinate):
    enemy_looking_now_coordinate = 35 - now_coordinate
    enemy_looking_before_coordinate = 35 - before_coordinate
    difference = enemy_looking_now_coordinate - enemy_looking_before_coordinate
    if difference == 6:  # 下
        return position_to_action(enemy_looking_before_coordinate, 0)
    elif difference == 1:  # 左
        return position_to_action(enemy_looking_before_coordinate, 1)
    elif difference == -6:  # 上
        return position_to_action(enemy_looking_before_coordinate, 2)
    elif difference == -1:  # 右
        return position_to_action(enemy_looking_before_coordinate, 3)
    else:
        print("ERROR:find_enemy_action_number_from_coordinate(illegal move)")
        return -1


# myの視点で状態を作成
def my_looking_create_state(ii_state, my_blue, my_red, enemy_blue, enemy_red):
    # プレイヤー毎のデュアルネットワークの入力の2次元配列の取得
    def pieces_array_of(blue_piece_list, red_piece_list):
        table_list = []
        blue_table = [0] * 36
        table_list.append(blue_table)  # ちなみにappendは参照渡し
        # blue_piece_listは駒のIDの値なので、ii_state.all_pieceでそのIDを参照してあげると座標が取れる
        for blue_piece in blue_piece_list:
            if ii_state.all_piece[blue_piece] < 36:  # 死駒を除外
                blue_table[ii_state.all_piece[blue_piece]] = 1

        red_table = [0] * 36
        table_list.append(red_table)
        for red_piece in red_piece_list:
            if ii_state.all_piece[red_piece] < 36:
                red_table[ii_state.all_piece[red_piece]] = 1

        return table_list

    # デュアルネットワークの入力の2次元配列の取得(自分と敵両方)
    return [pieces_array_of(my_blue, my_red), pieces_array_of(enemy_blue, enemy_red)]


# 入力の順序はcreate
# enemyの視点から状態を作成
def enemy_looking_create_state(ii_state, my_blue, my_red, enemy_blue, enemy_red):
    # プレイヤー毎のデュアルネットワークの入力の2次元配列の取得
    def pieces_array_of(blue_piece_list, red_piece_list):
        table_list = []
        blue_table = [0] * 36
        # blue_piece_listは駒のIDの値なので、ii_state.all_pieceでそのIDを参照してあげると座標が取れる
        for blue_piece in blue_piece_list:
            if ii_state.all_piece[blue_piece] < 36:  # 死駒を除外
                blue_table[ii_state.all_piece[blue_piece]] = 1
        blue_table.reverse()  # 逆視点にするために要素を反転
        table_list.append(blue_table)

        red_table = [0] * 36
        for red_piece in red_piece_list:
            if ii_state.all_piece[red_piece] < 36:
                red_table[ii_state.all_piece[red_piece]] = 1
        red_table.reverse()  # 逆視点にするために要素を反転
        table_list.append(red_table)

        return table_list

    # デュアルネットワークの入力の2次元配列の取得(自分と敵両方)
    return [pieces_array_of(enemy_blue, enemy_red), pieces_array_of(my_blue, my_red)]


# 行動と推測盤面に対応した行動価値のリストを返す
# 相手の行動前に、相手の目線で各パターンにおける各行動の価値を算出
def enemy_ii_predict(model, ii_state):
    a, b, c = DN_INPUT_SHAPE  # (6, 6, 4)
    my_piece_set = set(ii_state.my_piece_list)
    enemy_piece_set = set(ii_state.enemy_piece_list)
    policies_list = []
    enemy_legal_actions = list(ii_state.enemy_legal_actions())

    my_blue_set = ii_state.real_my_piece_blue_set
    my_red_set = ii_state.real_my_piece_red_set

    for num_and_enemy_blue in ii_state.enemy_estimated_num:  # enemyのパターンの確からしさを求めたい
        # 赤駒のインデックスをセット形式で獲得(my_blueはタプル)
        enemy_red_set = enemy_piece_set - set(num_and_enemy_blue[1])
        sum_np_policies = np.array([0] * len(enemy_legal_actions), dtype="f4")

        # 要修正
        ii_pieces_array = enemy_looking_create_state(
            ii_state, my_blue_set, my_red_set, num_and_enemy_blue[1], enemy_red_set,
        )

        # 方策の算出
        x = np.array(ii_pieces_array)
        x = x.reshape(c, a, b).transpose(1, 2, 0).reshape(1, a, b, c)
        y = model.predict(x, batch_size=1)
        policies = y[0][0][enemy_legal_actions]  # 合法手のみ
        policies /= sum(policies) if sum(policies) else 1  # 合計1の確率分布に変換

        # 行列演算するためにndarrayに変換
        np_policies = np.array(policies, dtype="f4")
        # myのパターンは既存のpoliciesに足すだけ
        sum_np_policies = sum_np_policies + np_policies

        policies_list.extend([sum_np_policies])
    return policies_list


# 相手の行動から推測値を更新
# state, enemy_ii_predictで作成した推測値の行列, 敵の行動番号
def update_predict_num_all(ii_state, beforehand_estimated_num, enemy_action_num):
    print(enemy_action_num)
    enemy_legal_actions = list(ii_state.enemy_legal_actions())
    enemy_action_index = enemy_legal_actions.index(enemy_action_num)

    for index, enemy_estimated_num in enumerate(ii_state.enemy_estimated_num):
        # ii_state.enemy_estimated_num[index][0]
        enemy_estimated_num[0] = (
            enemy_estimated_num[0] * gamma
        ) + beforehand_estimated_num[index][enemy_action_index]


# 駒の死亡処理
# 既存のパターンから推測値を抜き出して新しい推測値を作成
def reduce_pattern(dead_piece_ID, color_is_blue: bool, ii_state):
    if dead_piece_ID < 8 and color_is_blue:  # 敵駒 and 駒が青色
        # dead_piece_IDが含まれているものを削除
        # リストをそのままfor内で削除するとインデックスがバグるのでコピーしたものを参照
        for enemy_estimated_num in ii_state.enemy_estimated_num[:]:
            if dead_piece_ID in enemy_estimated_num[1]:
                ii_state.enemy_estimated_num.remove(enemy_estimated_num)
    elif dead_piece_ID < 8 and not color_is_blue:  # 敵駒 and 駒が赤色
        # dead_piece_IDが含まれていないものを削除
        for enemy_estimated_num in ii_state.enemy_estimated_num[:]:
            if dead_piece_ID not in enemy_estimated_num[1]:
                ii_state.enemy_estimated_num.remove(enemy_estimated_num)

    # all_pieceから削除
    ii_state.all_piece[dead_piece_ID] = 99
    # **_piece_listから削除
    if dead_piece_ID < 8:
        ii_state.enemy_piece_list.remove(dead_piece_ID)
    elif dead_piece_ID < 16:
        ii_state.my_piece_list.remove(dead_piece_ID)
    else:
        print("ERROR:reduce_pattern(**_piece_listから削除)")

    # living_piece_colorから削除
    if dead_piece_ID < 8 and color_is_blue:  # 敵駒 and 駒が青色
        ii_state.living_piece_color[0] -= 1
    elif dead_piece_ID < 8 and not color_is_blue:  # 敵駒 and 駒が赤色
        ii_state.living_piece_color[1] -= 1
    elif dead_piece_ID >= 8 and color_is_blue:  # 自駒 and 駒が青色
        ii_state.living_piece_color[2] -= 1
    elif dead_piece_ID >= 8 and not color_is_blue:  # 自駒 and 駒が赤色
        ii_state.living_piece_color[3] -= 1
    else:
        print("ERROR:reduce_pattern(living_piece_colorから削除)")


# 相手の推測値を使って無難な手を選択
# 価値が最大の行動番号を返す
def action_decision(model, ii_state):
    a, b, c = DN_INPUT_SHAPE  # (6, 6, 4)
    my_piece_set = set(ii_state.my_piece_list)
    enemy_piece_set = set(ii_state.enemy_piece_list)

    # 自分の駒配置を取得(確定)
    real_my_piece_blue_set = ii_state.real_my_piece_blue_set
    real_my_piece_red_set = ii_state.real_my_piece_red_set

    legal_actions = list(ii_state.legal_actions())

    actions_value_sum_list = np.array([0] * len(legal_actions), dtype="f4")

    # 相手の70パターンについてforループ(自分のパターンは確定で計算)
    for num_and_enemy_blue in ii_state.enemy_estimated_num:
        enemy_blue_set = set(num_and_enemy_blue[1])
        enemy_red_set = enemy_piece_set - enemy_blue_set

        # 盤面を6*6*4次元の情報に変換
        ii_pieces_array = my_looking_create_state(
            ii_state,
            real_my_piece_blue_set,
            real_my_piece_red_set,
            enemy_blue_set,
            enemy_red_set,
        )

        x = np.array(ii_pieces_array)
        x = x.reshape(c, a, b).transpose(1, 2, 0).reshape(1, a, b, c)
        y = model.predict(x, batch_size=1)

        # 方策の取得
        policies = y[0][0][legal_actions]  # 合法手のみ
        policies /= sum(policies) if sum(policies) else 1  # 合計1の確率分布に変換

        # 行列演算するためにndarrayに変換
        np_policies = np.array(policies, dtype="f4")

        # パターンごとに「推測値を重みとして掛けた方策」を足し合わせる
        actions_value_sum_list = actions_value_sum_list + (
            np_policies * num_and_enemy_blue[0]
        )

    best_action_index = np.argmax(actions_value_sum_list)  # 最大値のインデックスを取得
    best_action = legal_actions[best_action_index]  # 価値が最大の行動を取得
    return best_action


# 行動をsendall可能なバイナリデータに変換
def action_to_sendall_str(ii_state, action_num):
    # 行動する駒の座標を算出
    position = int(action_num / 4)
    # 駒の座標から駒のIDを特定
    piece_name = ii_state.piece_name[np.where(ii_state.all_piece == position)[0][0]]

    # 行動の方角を算出
    direction_num = action_num % 4
    if direction_num == 0:
        direction = "SOUTH"  # 下
    elif direction_num == 1:
        direction = "WEST"  # 左
    elif direction_num == 2:
        direction = "NORTH"  # 上
    else:  # direction_num == 3
        direction = "EAST"  # 右

    # sendallする形式に整形(ex: MOV:A,NORTH\r\n)
    send_str = "MOV:" + piece_name + "," + direction + "\r\n"

    # エンコードしてから返す
    return send_str.encode(encoding="utf-8")


# # 前に送信したバイナリデータと受け取ったバイナリデータを比較して相手の駒の移動前後の座標を算出
# # なお，ここで求める座標は自分目線になっている
# def recv_str_to_coordinate(before_tcp_str_b, now_tcp_str_b):
#     # バイナリデータをデコード
#     before_tcp_str = before_tcp_str_b.decode(encoding="utf-8")
#     now_tcp_str = now_tcp_str_b.decode(encoding="utf-8")

#     # ex)MOV?14R24R34R44R15B25B35B45B41u31u21u11u40u30u20u10u\r\n
#     # を見ると，28〜51番目に相手の駒の情報が格納されていることがわかる
#     for i in range(28, len(before_tcp_str)):
#         if before_tcp_str[i] != now_tcp_str[i]:
#             # 異なる文字が格納されているインデックスを見つける(移動した駒を見つける)
#             break

#     # iではなく((i-1)//3)*3+1とすることで、座標と駒色(例:41u)の先頭インデックス(3n+1)が取れる
#     beginningOfTheChanged = ((i - 1) // 3) * 3 + 1

#     # 列番号+行番号*6でii_state.all_pieceに格納されているような座標の形式に変換
#     before_coordinate = (
#         int(before_tcp_str[beginningOfTheChanged])
#         + int(before_tcp_str[beginningOfTheChanged + 1]) * 6
#     )
#     now_coordinate = (
#         int(now_tcp_str[beginningOfTheChanged])
#         + int(now_tcp_str[beginningOfTheChanged + 1]) * 6
#     )

#     return before_coordinate, now_coordinate


# ii_stateと受け取ったバイナリデータを比較して相手の駒の移動前と後の座標を算出
# なお，ここで求める座標は自分目線になっている
def recv_str_to_coordinate(ii_state, now_tcp_str_b):
    # バイナリデータをデコード
    now_tcp_str = now_tcp_str_b.decode(encoding="utf-8")

    # ex)MOV?14R24R34R44R15B25B35B45B41u31u21u11u40u30u20u10u\r\n
    # のように，28〜51番目に相手の駒の情報が格納されている
    now_piece = [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(8):
        x = int(now_tcp_str[28 + i * 3])
        y = int(now_tcp_str[28 + i * 3 + 1])
        # now_tcp_strからii_state.all_pieceと対応した座標を入手
        now_piece[7 - i] = x + y * 6

    # all_pieceと比較し，異なる駒を見つける
    for i in range(8):
        if now_piece[i] != ii_state.all_piece[i]:
            break

    before_coordinate = ii_state.all_piece[i]
    now_coordinate = now_piece[i]
    print("移動前：", before_coordinate)
    print("移動後：", now_coordinate)

    return before_coordinate, now_coordinate


# # プロトコルから相手の行動は送られず、更新されたボードが送られてくるそうなので、行動した駒の座標を求める
# def enemy_coordinate_checker(before_board, now_board):
#     for i in range(len(before_board) // 2, len(before_board)):
#         if before_board[i] != now_board[i]:
#             break
#     # iではなく(i//3)*3とすることで、座標と駒色(例:14R)の先頭インデックスが取れる(これしないと2文字目からとってくる恐れがある)
#     beginningOfTheChanged = (i // 3) * 3

#     # 列番号+行番号*6でgame.pyで使ってる表現に直せる
#     before_coordinate = (
#         int(before_board[beginningOfTheChanged])
#         + int(before_board[beginningOfTheChanged + 1]) * 6
#     )
#     now_coordinate = (
#         int(now_board[beginningOfTheChanged])
#         + int(now_board[beginningOfTheChanged + 1]) * 6
#     )

#     # 行動前と行動後の座標を返す
#     return before_coordinate, now_coordinate


# recvで受けたstrから，諸々の処理をした後に行動を決定する
# 諸々の処理：相手の行動からii_stateを更新→推測値の更新→自分の行動からii_stateを更新
def from_recv_to_action_num(model, ii_state, beforehand_estimated_num, now_tcp_str):
    # 前回のstrと受けたstrを並べて相手のaction_numを算出
    before, now = recv_str_to_coordinate(ii_state, now_tcp_str)  # 座標を算出
    enemy_action_num = calculate_enemy_action_number_from_coordinate(
        before, now
    )  # 行動番号を算出

    # 相手の行動からii_stateを更新
    ii_state.enemy_next(before, now)

    # 相手の行動から推測値を更新
    update_predict_num_all(ii_state, beforehand_estimated_num, enemy_action_num)

    # 自分の行動を決定
    action_num = action_decision(model, ii_state)

    return action_num


# 最初は諸々のデータがないので固定された行動を選ぶ
def first_move(model, ii_state):
    # 自分の行動を決定
    action_num = 106  # Bを前進させる手 #適宜変更

    # 自分の行動でii_stateを更新
    ii_state.next(action_num)

    # 自分の行動をsendall可能なバイナリデータに変換
    sendall_str_b = action_to_sendall_str(action_num)

    # sendall予定のバイナリデータを返す
    return sendall_str_b


def main():
    print("Please input a port number that you want to connect.")
    port = int(input())

    path = sorted(Path("./model").glob("*.h5"))[-1]
    model = load_model(str(path))
    ii_state = II_State({8, 9, 10, 11})

    # 相手の盤面から全ての行動の推測値を計算しておく(待ち時間)
    beforehand_estimated_num = enemy_ii_predict(model, ii_state)

    # クライアントの作成
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # サーバを指定
        s.connect(("localhost", port))

        # ネットワークのバッファサイズ。サーバからの文字列を取得(recv)する
        data = s.recv(2048)
        print(repr(data))

        # 赤駒のセット
        s.sendall(b"SET:ABCD\r\n")
        data = s.recv(2048)
        print(repr(data))

        if str(repr(data)) == str(b"OK \r\n"):
            print("駒のセット成功")

        else:
            print("駒のセット失敗")

        now_tcp_str = ""  # 直前に受け取ったtcp

        # 先行
        if port == 10000:
            # 先行の1手目は例外的に決まった行動をとる
            sendall_str_b = first_move(model, ii_state)
            s.sendall(sendall_str_b)

        # 受け取ったdataをstrに変換
        data = s.recv(2048)
        now_tcp_str = data.decode(encoding="utf-8")

        while 1:
            # 相手の行動からii_stateを更新→推測値の更新→action_numの決定
            action_num = from_recv_action_num(
                model, ii_state, beforehand_estimated_num, now_tcp_str
            )

            # 自分の行動をsendall可能なバイナリデータに変換
            sendall_str_b = action_to_sendall_str(action_num)
            print("自分の行動", str(sendall_str_b))

            kill_and_id_index = ii_state.this_action_will_kill(action_num)
            if kill_and_id_index[0]:
                # 相手の駒を殺すなら，recvを待ってからii_stateを更新する必要がある(殺した駒の色を確認するため)

                # 行動を送信
                s.sendall(sendall_str_b)

                # 受け取ったdataをstrに変換
                data = s.recv(2048)
                now_tcp_str = data.decode(encoding="utf-8")

                # 敵の駒を殺す際のnext
                ii_state.kill_next(action_num, kill_and_id_index[1], now_tcp_str)

                # 相手の盤面から全ての行動の推測値を計算する(急ぐ必要がある)
                beforehand_estimated_num = enemy_ii_predict(model, ii_state)

            else:
                # 相手の駒を殺さないなら通常処理

                # 行動を送信
                s.sendall(sendall_str_b)

                # 自分の行動でii_stateを更新
                ii_state.next(action_num)

                # 相手の盤面から全ての行動の推測値を計算しておく(待ち時間)
                beforehand_estimated_num = enemy_ii_predict(model, ii_state)

                # 受け取ったdataをstrに変換
                data = s.recv(2048)
                now_tcp_str = data.decode(encoding="utf-8")

        # s.sendall(b"MOV:A,NORTH\r\n")

        # data = s.recv(2048)
        # print(str(repr(data)))


def test():
    # クライアントの作成
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # サーバを指定
        s.connect(("localhost", 10000))

        # ネットワークのバッファサイズ。サーバからの文字列を取得(recv)する
        data = s.recv(2048)
        print("recv", repr(data))

        # 赤駒のセット
        s.sendall(b"SET:ABCD\r\n")
        data = s.recv(2048)
        print("recv", repr(data))

        if str(repr(data)) == str(b"OK \r\n"):
            print("駒のセット成功")

        else:
            print("駒のセット失敗")

        data = s.recv(2048)
        print("recv", repr(data))

        s.sendall(b"MOV:A,NORTH\r\n")
        data = s.recv(2048)
        print(repr(data))
        data = s.recv(2048)
        print("recv", repr(data))

        s.sendall(b"MOV:B,NORTH\r\n")
        data = s.recv(2048)
        print(repr(data))
        data = s.recv(2048)
        print("recv", repr(data))

        s.sendall(b"MOV:C,NORTH\r\n")
        data = s.recv(2048)
        print(repr(data))
        data = s.recv(2048)
        print("recv", repr(data))

        s.sendall(b"MOV:D,NORTH\r\n")
        data = s.recv(2048)
        print(repr(data))
        data = s.recv(2048)
        print("recv", repr(data))

        s.sendall(b"MOV:E,NORTH\r\n")
        data = s.recv(2048)
        print(repr(data))
        data = s.recv(2048)
        print("recv", repr(data))

        s.sendall(b"MOV:F,NORTH\r\n")
        data = s.recv(2048)
        print(repr(data))
        data = s.recv(2048)
        print("recv", repr(data))

        s.sendall(b"MOV:G,NORTH\r\n")
        data = s.recv(2048)
        print(repr(data))
        data = s.recv(2048)
        print("recv", repr(data))

        s.sendall(b"MOV:H,NORTH\r\n")
        data = s.recv(2048)
        print(repr(data))
        data = s.recv(2048)
        print("recv", repr(data))


if __name__ == "__main__":
    main()
    # test()
