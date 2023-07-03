import pygame
import numpy as np
import tensorflow as tf 
from collections import deque
import random

'''

before chatgpt was capable of creating this 03/07/2023. I made this from scrach

'''

pygame.init()

SNAKE_STEP = 30
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 700
NO_IN_GRID_BLOCKS  = 600 / 30
EXPERIENCES_CAPACITY = 1000
SCORE = 0
ALPHA = 1
GAMMA = 0.5
SCORES = []
STATE = []
EXPERIENCES = deque(maxlen=EXPERIENCES_CAPACITY)
DECISIONS = ["up","down","left","right"]  
QTABLE_VS = []
QTABLE = []

window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("My Pygame Window")


def softmax(x):
    e_x = np.exp(x - np.max(x))  #Subtracting the maximum value for numerical stability
    return e_x / np.sum(e_x)


def epsilon_greedy_action(q_values, epsilon):
    if random.uniform(0, 1) < epsilon:
        # Explore: Choose a random action uniformly
        action = random.choice(DECISIONS)
    else:
        # Exploit: Choose the action with the highest Q-value
        action = DECISIONS[q_values.index(max(q_values))]
        
    return action


def q_table_update(index,decision, origin_table, next_step_table):
    
    if STATE[-1][0] >= 0 and STATE[-1][1] >= 0 and STATE[-1][0] != 210 and STATE[-1][1] != 210:
        im_reward = -1
    elif STATE[-1][0] < 0 and STATE[-1][1] < 0:
        im_reward = -100
    else:
        im_reward = 100
    #print("state[-1] is: ", STATE[-1])
    #print("im_reward is: ", im_reward)    

    #DECISIONS = ["up","down","right","left"]  
    '''one_deeper = [(STATE[-1][0], STATE[-1][1] - 30),(STATE[-1][0], STATE[-1][1] + 30), (STATE[-1][0] + 30, STATE[-1][1]), (STATE[-1][0] - 30, STATE[-1][1])]
    tup = one_deeper[next_step_table.index(max(next_step_table))]
    if tup[0] or tup[1] < 0:
        tup = (30,30)
    

    if STATE[-1][-1:] == (180,210) or STATE[-1][-1:] == (210,180) or STATE[-1][-1:] == (240,210) or STATE[-1][-1:] == (210,240) or STATE[-1][-1:] == (210,210):
        print(f"reached: {STATE[-1][-1:]} im_reward is {}")'''

    #ind = round(((tup[1] / SNAKE_STEP) * NO_IN_GRID_BLOCKS ) + (tup[0] / SNAKE_STEP))    
    if origin_table[DECISIONS.index(decision)] == 1000:
        pass
    else:
        new_q_value = origin_table[DECISIONS.index(decision)]  + ALPHA * (im_reward + (GAMMA * max(next_step_table))) - origin_table[DECISIONS.index(decision)]
        QTABLE_VS[index][DECISIONS.index(decision)] = new_q_value

    


class agent:
    def __init__(self):
        self.w1 = np.random.rand(4,4)
        self.w2 = np.random.rand(4,4)
        self.w3 = np.random.rand(4,4)
        self.weights = [self.w1, self.w2, self.w3]


    def feed_forward(self,state):
        inputs = 0    

        for i, weights in enumerate(self.weights):
            if i == 0: 

                output = np.dot(weights,state)
                output = np.maximum(-1,output)
                inputs = output

            elif i == len(self.weights) - 1:
                  output = np.dot(weights,inputs)  
                  out = softmax(output)

            else:
                output = np.dot(weights,inputs) 
                inputs = output

        return np.argmax(out) 

                

class Score:

    def __init__(self):
        self.score = 0
        self.text = f"SCORE: {self.score}"
        self.text_colour = (0, 0, 0)
        self.font = pygame.font.Font(None, 36)

        self.original_score = self.score
        self.last_score = self.score

    def update(self,*args):

        try:
            if args[0] == "norm":
                self.score -= 1

        except:
            pass

        try:
            if args[0] == "out":
                self.score -= 100
                SCORES.append(self.score)
                self.score = self.original_score

        except:
            pass    

        try:
            if args[0] == "goal": 
                self.score += 100
                SCORES.append(self.score)
                self.score = self.original_score

        except:
            pass

        self.text = f"SCORE: {self.score}"        
        

    def draw(self):
        text_surface = self.font.render(self.text, True, self.text_colour)
        text_rect = text_surface.get_rect()
        text_rect.topleft = (20, 620)
        window.blit(text_surface, text_rect)


    def draw_prev(self):

        if len(SCORES) > 0:
            self.last_score = SCORES[-1]
        else:
             self.last_score = self.score

        text = f"PREV SCORE: {self.last_score}"
        text_surface = self.font.render(text, True, self.text_colour)
        text_rect = text_surface.get_rect()
        text_rect.topleft = (350, 620)
        window.blit(text_surface, text_rect)


class Snake:
    
    def __init__(self, x,y,WINDOW_WIDTH,WINDOW_HEIGHT):
        self.x = x
        self.y = y
        self.WINDOW_WIDTH = WINDOW_WIDTH
        self.WINDOW_HEIGHT = WINDOW_HEIGHT
        STATE.append([self.x, self.y])

        self.original_x = self.x
        self.original_y = self.y
        self.score = Score()
        self.episode_no = 0

    def draw(self):
        pygame.draw.rect(window, (0,0,255),(self.x,self.y,30,30))
        self.score.draw()
        self.score.draw_prev()

    def move(self,up = True, down = False, right = False, left = False):
        
        oldx = self.x
        oldy = self.y

        if STATE[0] == [210,210]:
            self.reset()

        if left == True:
            action = "left"
            self.x -= SNAKE_STEP
            self.score.update("norm")
            STATE[0] = [self.x, self.y]

                
        elif right == True:
            action = "right"
            self.x += SNAKE_STEP  
            self.score.update("norm")  
            STATE[0] = [self.x, self.y]
            

        elif up == True: 
            action = "up"
            self.y -= SNAKE_STEP 
            self.score.update("norm")
            STATE[0] = [self.x, self.y]

        elif down == True:
            action = "down"
            self.y += SNAKE_STEP  
            self.score.update("norm")
            STATE[0] = [self.x, self.y] 
           

        if self.x == 210 and self.y == 210:
            self.score.update("goal")
            STATE[0] = [self.x, self.y] 
   
        if self.x < 0 or self.x >= WINDOW_WIDTH or self.y < 0 or self.y >= 600:
            self.score.update("out")
            EXPERIENCES.append(((oldx, oldy),action,SCORES[-1],(self.x, self.y),self.episode_no,"done"))
            self.reset()

        else:
            EXPERIENCES.append(((oldx, oldy),action,self.score.score,(self.x, self.y),self.episode_no))        



    def reset(self):
        self.x = self.original_x
        self.y = self.original_y
        STATE[0] = [self.x, self.y]
        self.episode_no += 1


def generate_q_table():
    for w in range(0,600,30):
        for h in range(0,600,30):
            QTABLE_VS.append([])
            for action in DECISIONS:
                if h == 210 and w == 210:
                     QTABLE_VS[-1].append(1000)
                else:
                    QTABLE_VS[-1].append(0)
            QTABLE_VS[-1].append((0+w,0+h))    


def draw_grid():
    for w in range(0,600,30):
        for h in range(0,600,30):
            pygame.draw.rect(window, (0,0,0),(0,0+h,600,2))
            pygame.draw.rect(window, (0,0,0),(0+w,0,2,600))
            

    pygame.draw.rect(window, (0,0,0),(0,600,600,2))


def draw_food():
    pygame.draw.rect(window, (0, 255, 0), (210, 210, 30, 30))


def handle_agent_decision(decision, snake):
    if decision == "down":
        snake.move(down = True, up = False, left = False, right = False)
    elif decision == "up":
        snake.move(down = False, up = True, left = False, right = False)
    elif decision == "right":
        snake.move(down = False, up = False, left = False, right = True)
    elif decision == "left":
        snake.move(down = False, up = False, left = True, right = False)      


def handle_snake_movement(keys, snake):
    if keys[pygame.K_DOWN]:
        snake.move(down = True, up = False, left = False, right = False)
    elif keys[pygame.K_UP]:
        snake.move(down = False, up = True, left = False, right = False)
    elif keys[pygame.K_RIGHT]:
        snake.move(down = False, up = False, left = False, right = True)
    elif keys[pygame.K_LEFT]:
        snake.move(down = False, up = False, left = True, right = False)      

def draw_exploration_rate(epsilon_rate):
    font = pygame.font.Font(None, 25)
    text_surface = font.render(f"exploration rate: {epsilon_rate}", True,(255, 0, 0))
    text_rect = text_surface.get_rect()
    text_rect.topleft = (20, 660)
    window.blit(text_surface, text_rect)


def draw(epsilon_rate,snake):
    window.fill((255, 255, 255))
    draw_food()
    draw_exploration_rate(epsilon_rate) 
    snake.draw()
    draw_grid()
    pygame.display.update()


#STATE = snake x coord, snake y coord, food x coord, food y coord.
#4 x 4 x 4
  
def main():
    running = True
    clock = pygame.time.Clock()
    snake = Snake(30,30,30,30)
    jamesbond = agent()
    generate_q_table()
    itter = 0
    print("original state is: ", [30,30])
    epsilon_rate = 1
    
    
    while running:
        itter += 1
        
        if epsilon_rate <= 0.001:
            epsilon_rate = 0.001
        else:  
            epsilon_rate -= 0.0001 

        clock.tick(10)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                print(QTABLE_VS)
                print(epsilon_rate)

        
        keys = pygame.key.get_pressed()
        draw(epsilon_rate,snake) 

        #AGENT DECISION.
        index = round(((STATE[-1][0] / SNAKE_STEP) * NO_IN_GRID_BLOCKS ) + (STATE[-1][1] / SNAKE_STEP))
        #print("index is: ", index)
        #print("table entry is: ", QTABLE_VS[index])
        #decision = jamesbond.feed_forward(np.array(STATE[0]).reshape(4,1))   ---------------> policy gradients.    
        decision = epsilon_greedy_action(QTABLE_VS[index][:-1],epsilon_rate)    #-----------------> greedy epsilon.
        #handle_snake_movement(keys, snake)     ---------------------> move by myself         
                
        #MOVEMENT. 
        handle_agent_decision(decision,snake)
        #print("decision is ", decision)
        #print("new coordinate is: ", STATE[-1])
        

        #REWARD CALCULATION.
        q_table_update(index,decision,QTABLE_VS[index][:-1], QTABLE_VS[round(((STATE[-1][0] / SNAKE_STEP) * NO_IN_GRID_BLOCKS) + (STATE[-1][1] / SNAKE_STEP))][:-1])
        '''if itter >= 5: 
            print(QTABLE_VS)
            return'''


        
        #Q-LEARNING.


        
        
        
             
    pygame.quit()
    quit()


main()    


generate_q_table()
print(QTABLE_VS[84])




