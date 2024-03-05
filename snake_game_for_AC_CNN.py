
#MIT License
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#The code is slightly modified, and the original code can be found at https://github.com/patrickloeber/snake-ai-pytorch/blob/main/snake_game_human.py

import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import torch
import cv2

pygame.init()
font = pygame.font.Font(None, 25)

class Direction(Enum):
    RIGHT = 4
    LEFT = 2
    UP = 1
    DOWN = 3

class REWARD(Enum):
    DEAD = -5e-2
    MOVE_TO_FOOD = 1e-1
    EAT_FOOD = 1
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
GREEN = (0, 255, 0)

BLOCK_SIZE = 20
SPEED = 5

class SnakeGame:
    
    def __init__(self, w=240, mode='player',speed = SPEED):
        self.w = w
        self.h = w
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.speed = speed
        self.clock = pygame.time.Clock()
        # init game state
        self.time_not_eat = 0
        self._gnerate_snake()
        self.boundary = [Point(0, 0)     , Point(0, self.h), 
                         Point(self.w, 0), Point(self.w, self.h)]
        self.score = 0
        self.food = None
        self.mode = mode
        self._place_food()
        
    def reset(self):
        self.direction = Direction.RIGHT
        self.time_not_eat = 0
        self._gnerate_snake()
        self.score = 0
        self.food = None
        self._place_food()
        self._update_ui()
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def _gnerate_snake(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.head = Point(x, y)
        self.old_head = self.head
        self.snake = [self.head]
        self.direction = random.choice(list(Direction))
        if self.direction == Direction.RIGHT:
            direction_vector = Point(-1, 0)
        elif self.direction == Direction.LEFT:
            direction_vector = Point(1, 0)
        elif self.direction == Direction.UP:
            direction_vector = Point(0, 1)
        elif self.direction == Direction.DOWN:
            direction_vector = Point(0, -1)
        for i in range(3):
            self.snake.append(Point(x + direction_vector.x * BLOCK_SIZE * (i+1),
                                    y + direction_vector.y * BLOCK_SIZE * (i+1)))
    
    def _close_to_food(self):
        return abs(self.head.x - self.food.x) < abs(self.old_head.x - self.food.x) or abs(self.head.y - self.food.y) < abs(self.old_head.y - self.food.y)
        
    def play_step(self):
        # 1. collect user input
        #player mode
        eatfood = False
        if self.mode == 'player':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.direction = Direction.LEFT
                    elif event.key == pygame.K_RIGHT:
                        self.direction = Direction.RIGHT
                    elif event.key == pygame.K_UP:
                        self.direction = Direction.UP
                    elif event.key == pygame.K_DOWN:
                        self.direction = Direction.DOWN

        elif self.mode == 'machine':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            
        # 2. move
        self._move(self.direction) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        game_over = False
        if self._is_collision():
            game_over = True

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            self._place_food()
            eatfood = True
            self.time_not_eat = 0
        else:
            self.snake.pop()
            self.time_not_eat += 1
            if self.time_not_eat > len(self.snake)*30:
                game_over = True

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(self.speed)

        # 6. calculate the reward
        if game_over:
            reward = REWARD.DEAD.value
        elif eatfood:
            reward = REWARD.EAT_FOOD.value
        #else if the snake is getting closer to the food, give a small reward
        elif self._close_to_food():
            reward = REWARD.MOVE_TO_FOOD.value #*np.exp(-self.time_not_eat/30)
        else:
            reward = -REWARD.MOVE_TO_FOOD.value #*np.exp(-self.time_not_eat/40)
        # 7. return game over and score
        return game_over, self.score, reward
    
    def _is_collision(self):
        return self.position_is_collision(self.head)
    
    def position_is_collision(self,position):
        # hits boundary
        if position.x > self.w - BLOCK_SIZE or position.x < 0 or \
           position.y > self.h - BLOCK_SIZE or position.y < 0:
            return True
        # hits itself
        if position in self.snake[1:]:
            return True
        
        return False

        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        
        #draw head as green
        pygame.draw.rect(self.display, GREEN, pygame.Rect(self.snake[0].x, self.snake[0].y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        if self.mode == 'player':
            text = font.render("Score: " + str(self.score), True, WHITE)
            self.display.blit(text, [0, 0])

        pygame.display.flip()
        
    def _move(self, direction):
        x = self.head.x
        y = self.head.y

        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        
        self.old_head = self.head
        self.head = Point(x, y)

    
    def state(self):
        #build the grid state of the whole game
        grid_w = np.zeros(((self.w)//BLOCK_SIZE+2,(self.h)//BLOCK_SIZE+2))
        grid_f = np.zeros(((self.w)//BLOCK_SIZE+2,(self.h)//BLOCK_SIZE+2))
        grid_s = np.zeros(((self.w)//BLOCK_SIZE+2,(self.h)//BLOCK_SIZE+2))

        
        for i in range(len(self.snake)):
            if self.snake[i].x//BLOCK_SIZE < (self.w)//BLOCK_SIZE and self.snake[i].y//BLOCK_SIZE < (self.h)//BLOCK_SIZE:
                grid_s[self.snake[i].x//BLOCK_SIZE+1,self.snake[i].y//BLOCK_SIZE+1] = (len(self.snake)-i)/len(self.snake)
                grid_f[self.snake[i].x//BLOCK_SIZE+1,self.snake[i].y//BLOCK_SIZE+1] = .1
                grid_w[self.snake[i].x//BLOCK_SIZE+1,self.snake[i].y//BLOCK_SIZE+1] = .1                
        

        grid_f[self.food.x//BLOCK_SIZE+1,self.food.y//BLOCK_SIZE+1] = 1
        grid_f[self.head.x//BLOCK_SIZE+1,self.head.y//BLOCK_SIZE+1] = .8
        grid_w[self.head.x//BLOCK_SIZE+1,self.head.y//BLOCK_SIZE+1] = .8


        grid_w[0,:] = 1
        grid_w[-1,:] = 1
        grid_w[:,0] = 1
        grid_w[:,-1] = 1
        grid_f[0,:] = 0
        grid_f[-1,:] = 0
        grid_f[:,0] = 0
        grid_f[:,-1] = 0
        grid_s[0,:] = 0
        grid_s[-1,:] = 0
        grid_s[:,0] = 0
        grid_s[:,-1] = 0
        grid = np.stack((grid_w,grid_f,grid_s))
        # print(grid.shape)
        grid = torch.tensor(grid, dtype=torch.float32)

        #use cv2 to save the image
        img = grid.numpy()
        img = np.transpose(img, (1,2,0))
        cv2.imwrite('snake.png',img*255)

        return grid.unsqueeze(0)

    
    def action(self,action):
        if (action == 0) and (self.direction != Direction.RIGHT):
            self.direction = Direction.LEFT
        elif (action == 1) and (self.direction != Direction.LEFT):
            self.direction = Direction.RIGHT
        elif (action == 2) and (self.direction != Direction.DOWN):
            self.direction = Direction.UP
        elif (action == 3) and (self.direction != Direction.UP):
            self.direction = Direction.DOWN
        else:
            return False
        return True
        

            

if __name__ == '__main__':
    game = SnakeGame()
    
    # game loop
    while True: 
        game_over, score ,reward = game.play_step()
        a = game.state()
        #format the print "reward: XXXX"
        print(f'reward:{reward:1.2f}',end='\r')
        if game_over == True:
            break
        
    print('Final Score', score)
    print('Reward', reward)
                
    pygame.quit()