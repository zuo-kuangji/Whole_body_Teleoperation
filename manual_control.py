import time
import zmq

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
)

ctx = zmq.Context()
sock = ctx.socket(zmq.PUB)
sock.bind("tcp://*:5556")

time.sleep(0.3)  # 给订阅端一点连接时间

# 这条就相当于在 Terminal 2 按下 ]
sock.send(build_command_message(start=True, stop=False, planner=True))

while True:
    # mode=2 通常是 WALK
    sock.send(build_planner_message(
        mode=2,
        movement=[1.0, 0.0, 0.0],  # 朝前走
        facing=[1.0, 0.0, 0.0],    # 身体朝前
        speed=0.6,
        height=-1.0,
    ))
    time.sleep(0.02)
