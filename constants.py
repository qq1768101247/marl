# -*- coding: utf-8 -*-
#开发者：Bright Fang
#开发时间：2022/9/17 18:22
import numpy as np
import pygame
import random

import torch

CREATE_ENEMY_EVENT=pygame.USEREVENT

ENEMY_MAKE_TIME=random.randint(500,600)

ENEMY_FLAG=True

ENEMY_AREA=ENEMY_AREA_X,ENEMY_AREA_Y,ENEMY_AREA_WITH,ENEMY_AREA_HEIGHT=50,50,700,600

low = np.array([-1, -1])
high = np.array([1, 1])
action_number=high.shape[0]
max_action = high[0]
min_action = low[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

state_dim_leader = 25  # 领航者obs维度（根据实际修改）
state_dim_follower = 18 # 追随者obs维度
N_Agent=1
M_Enemy=4*N_Agent

SCREEN_SIZE=SCREEN_W,SCREEN_H=1000,800

FPS=60

FONT_CHINESE='华文新魏'

FONT_ENGLISH='arial'

WHITE=(255,255,255)

BLACK=(0,0,0)

RED=(255,0,0)

BLUE=(0,0,255)

GREEN=(0,128,0)

CLICK=False

OPEN_MENU=False

OPEN_MUSIC=True

OPEN_SOUND=True

BULLET_SPEED=5#90/FPS