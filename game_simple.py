import pygame
import time
import pickle
import os
from pygame.locals import *
import numpy as np
import pandas as pd
pygame.font.init()

pygame.init()
# LOAD IMAGE
PLAYER_IMG = pygame.transform.scale(pygame.image.load("player.jpeg"), (40, 40))
ENEMY_IMG = pygame.transform.scale(pygame.image.load("enemy.png"), (40, 40))
WALL_IMG = pygame.transform.scale(pygame.image.load("wall.jpeg"), (30, 30))
TREASURE_IMG = pygame.transform.scale(pygame.image.load("treasure.png"), (40, 30))
data_wall = pd.read_excel('data_wall.xlsx')
STAT_FONT = pygame.font.SysFont("comicsans",25)

WALL_PENALTY = -1
ENEMY_PENALTY = -200
TIME_PENALTY = 0
GOAL_REWARD = 5
Reward = 0
EP_Reward = []
HM_EPISODE = 20

obs = [0,0]
HEIGHT = 480
WIDTH = 480
BATCH_SIZE = 45
DISCRETE_OS_SIZE= [BATCH_SIZE] * len(obs)
env_space_low = np.array([0, 0])
env_space_high = np.array([HEIGHT, WIDTH])
discrete_os_win_size = (env_space_high)/DISCRETE_OS_SIZE
q_table = np.random.uniform(low=0,high = 1, size=DISCRETE_OS_SIZE + [4])
print(q_table)

def get_discrete_state(state):
    goal_marge = np.array([WIDTH-goal.x, HEIGHT-goal.y])
    discrete_state = (state - env_space_high + goal_marge)/discrete_os_win_size
    #print(state)
    return tuple(discrete_state.astype(np.int))

#Q TABLE
HM_EPISODES = 2000
epsilon = 0.9998
EPS_DECAY = 0.998
SHOW_EVERY = 1
start_q_table = None # of filename
LEARNING_RATE = 0.1
DISCOUNT = 0.995



class Player():
    IMG = PLAYER_IMG

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.img = self.IMG
        self.vel = 4
        self.hitbox = (self.x, self.y, 40, 40)
        self.top = self.y
        self.left = self.x
        self.right = self.left + self.hitbox[2]
        self.bottom = self.top + self.hitbox[3]

    def draw(self, win):
        win.blit(self.IMG, (self.x, self.y))
        self.hitbox = (self.x + 4, self.y, 30, 40)
        self.top = self.y
        self.left = self.x
        self.right = self.left + self.hitbox[2]
        self.bottom = self.top + self.hitbox[3]

        # pygame.draw.rect(win,(255,0,0),self.hitbox, 2)

class Wall():
    IMG = WALL_IMG

    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle
        self.img = pygame.transform.rotate(self.IMG, angle)
        self.hitbox = (self.x, self.y, 30, 30)
        self.top = self.y
        self.left = self.x
        self.right = self.left + self.hitbox[2]
        self.bottom = self.top + self.hitbox[3]

    def draw(self, win):
        win.blit(self.img, (self.x, self.y))

class Goal():
    IMG = TREASURE_IMG
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.img = self.IMG
        self.hitbox = (self.x, self.y, 30, 30)
        self.top = self.y
        self.left = self.x
        self.right = self.left + self.hitbox[2]
        self.bottom = self.top + self.hitbox[3]

    def draw(self, win):
        win.blit(self.img, (self.x, self.y))


def draw_window(win, player, walls, goal):
    win.fill((255, 255, 255))
    # walls
    for wall in walls:
        wall.draw(win)
    player.draw(win)
    goal.draw(win)
    text = STAT_FONT.render("Score : "+ str(Reward), 1, (255,255,255))
    text2 = STAT_FONT.render("Episode : "+ str(Episode), 1 , (255, 255, 255))
    text3 = STAT_FONT.render("Epsilon : "+ str(epsilon), 1 , (255, 255, 255))
    win.blit(text, (10, 10))
    win.blit(text2, (150, 10))
    win.blit(text3, (10 , 460))


    pygame.display.update()

def colide():
    colide_test = False
    treasure_test = False
    for wall in walls:
        if player.right < wall.left:
            colide_test = False
        elif player.left > wall.right :
            colide_test = False
        elif  player.top > wall.bottom:
            colide_test = False
        elif player.bottom < wall.top :
            colide_test = False
        else:
            colide_test = True
            break

    if player.right >= goal.left:
        if player.left <= goal.right:
            if player.top <= goal.bottom:
                if player.bottom >= goal.top:
                    treasure_test = True

    return colide_test, treasure_test

window = pygame.display.set_mode((WIDTH, HEIGHT))

def env_reset(player, epsilon, Episode):
    player.x = 50
    player.y = 50
    Episode += 1
    epsilon = epsilon*EPS_DECAY
    EP_Reward.append(Reward)
    return Episode, epsilon


player = Player(50, 50)
goal = Goal(350, 65)
walls = []
for i in range(len(data_wall)):
    x = data_wall['x'].iloc[i]
    y = data_wall['y'].iloc[i]
    angle = data_wall['angle'].iloc[i]
    wall = Wall(x, y, angle)
    walls.append(wall)


run = True
Episode = 1
q_table_init = q_table
print(q_table.shape)
past = ""
count = 0
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            #print(EP_Reward)
            print(q_table)
            with open(f"q_table-{int(time.time())}.pickle", "wb") as f:
                pickle.dump(q_table, f)
            pygame.quit()
            quit()

    colide_test, treasure_test = colide()
    Reward += TIME_PENALTY

    state = np.array([goal.x - player.x, goal.y - player.y])
    discrete_state = get_discrete_state(state)

    if np.random.random() < epsilon:
        action = np.argmax(q_table[discrete_state])
    else:
        action = np.random.randint(0,4)

    count += 1
    reward = TIME_PENALTY
    colide_test, treasure_test = colide()
    if colide_test:
        reward = WALL_PENALTY
        Reward += reward
        if past == 'left':
            player.x += 2*discrete_os_win_size[1]
        if past == 'right':
            player.x -= 2*discrete_os_win_size[1]
        if past == 'top':
            player.y += 2*discrete_os_win_size[0]
        if past == 'bottom':
            player.y -= 2*discrete_os_win_size[0]

    colide_test, treasure_test = colide()
    if action == 0 and not colide_test:
        player.x -= discrete_os_win_size[1]
        past = 'left'
    elif action == 1 and not colide_test:
        player.x += discrete_os_win_size[1]
        past = 'right'
    elif action == 2 and not colide_test:
        player.y -= discrete_os_win_size[0]
        past = 'top'
    elif action == 3 and not colide_test:
        player.y += discrete_os_win_size[0]
        past = 'bottom'

    state = np.array([goal.x - player.x, goal.y - player.y])
    new_discrete_state = get_discrete_state(state)
    max_future_q = np.max(q_table[new_discrete_state])
    current_q = q_table[discrete_state][action]
    colide_test, treasure_test = colide()

    if Reward <= -2000 or count >= 800 or player.x<=0 or player.x>=WIDTH or player.y<=0 or player.y >= HEIGHT:
        Reward = -2000
        Episode, epsilon = env_reset(player, epsilon, Episode)
        Reward = 0
        count = 0
    elif treasure_test == True:
        print(f"made it to episode : {Episode}")
        reward = GOAL_REWARD
        Reward += reward
        new_q = reward
        q_table[discrete_state][action] = new_q
        Episode, epsilon = env_reset(player, epsilon, Episode)
        Reward = 0
        count = 0
    elif colide_test:
        reward = WALL_PENALTY
        new_q = reward
        q_table[discrete_state][action] = new_q

    else:
        reward = 0
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[discrete_state][action] = new_q

    draw_window(window, player, walls, goal)
