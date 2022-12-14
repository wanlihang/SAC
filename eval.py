import paddle
from flask import Flask, request
from paddle.distribution import Normal

from Actor import Actor

app = Flask(__name__)

import logging

@app.route('/')
def index():
    return "Hello, World!"


def get_action(state):
    # 初始化模型
    actor = Actor(30, 1)
    # 载入模型参数
    actor_state_dict = paddle.load("linear_net.pdparams")
    # 将load后的参数与模型关联起来
    actor.set_state_dict(actor_state_dict)
    state = paddle.to_tensor(state.reshape(1, -1), dtype='float32')
    act_mean, act_log_std = actor(state)
    normal = Normal(act_mean, act_log_std.exp())
    # 重参数化  (mean + std*N(0,1))
    x_t = normal.sample([1])
    action = paddle.tanh(x_t)

    log_prob = normal.log_prob(x_t)
    log_prob -= paddle.log((1 - action.pow(2)) + 1e-6)
    log_prob = paddle.sum(log_prob, axis=-1, keepdim=True)

    action, _ = action[0], log_prob[0]

    action_numpy = action.cpu().numpy()[0]

    action = action_numpy

    # 动作转换
    if action <= - 1.0 / 3:
        action = 0
    elif action <= 1.0 / 3:
        action = 1
    else:
        action = 2

    return action

@app.route('/getAction', methods=['POST'])
def getAction():
    try:
        state =
        print(request.get_json())

        responses = get_action('Node')
    except IndexError as e:
        logging.error(str(e))
        return 'exception:' + str(e)
    except KeyError as e:
        logging.error(str(e))
        return 'exception:' + str(e)
    except ValueError as e:
        logging.error(str(e))
        return 'exception:' + str(e)
    except Exception as e:
        logging.error(str(e))
        return 'exception:' + str(e)
    else:
        return responses


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
