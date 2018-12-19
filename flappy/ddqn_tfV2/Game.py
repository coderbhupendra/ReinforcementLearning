# 1 - Import library
import pygame
import random
import numpy as np
from ddqn_tfV2.Hyperparameters import *
import pylab
keys = [False, False, False, False]
width, height = 640, 480

# 3 - Load images
bird = pygame.image.load("/home/bhupendra/RL/flappy/resources/images/bird.jpeg")
bar = pygame.image.load("/home/bhupendra/RL/flappy/resources/images/bar.png")
grass= pygame.image.load("/home/bhupendra/RL/flappy/resources/images/grass.png")
coin=pygame.image.load("/home/bhupendra/RL/flappy/resources/images/coin.png")
barw,barh=bar.get_width(),bar.get_height()
baruprect = pygame.Rect(bar.get_rect())
bardownrect = pygame.Rect(bar.get_rect())
birdrect = pygame.Rect(bird.get_rect())
coinrect = pygame.Rect(coin.get_rect())

# to load sound only onces in constructor
flag=1
gameno=0
score_plot=[]
class Game:
    def __init__(self,render):
        self.barno=0
        self.barsup=[]
        self.barsdown=[]
        self.coins=[]
        self.popcoin=True
        self.bartime=1
        self.playerpos=[100,100]
        self.total_score=0
        self.reward=0
        self.action=0
        self.state_size=state_size
        self.action_size=action_size
        self.initial_state = [ 100,100,100, 100]
        self.done=False
        self.render=render
        self.stack_frames=[]
        for i in range(4):self.stack_frames.append(self.initial_state)
        global flag
        if(self.render and flag):
            # 2 - Initialize the game
            flag=0
            pygame.init()
            pygame.mixer.init()
            # 3.1 - Load audiow
            hit = pygame.mixer.Sound("../resources/audio/explode.wav")
            hit.set_volume(0.05)
            pygame.mixer.music.load('../resources/audio/moonlight.wav')
            pygame.mixer.music.play(-1, 0.0)
            pygame.mixer.music.set_volume(0.25)
            self.screen=pygame.display.set_mode((width, height))

    def reset(self,render,plotscore=True):
        if plotscore:
            score_plot.append(self.total_score)
            if gameno % 5 == 0:
                pylab.plot(score_plot[1:], 'b')
                pylab.savefig(score_plot_path)
        self.__init__(render)

    def getObservation(self):
        upy, downy, xdiff = 0, 0, 0
        if (self.playerpos[0] < self.barsdown[0][0]+barw):
            downy = self.barsdown[0][1]
            upy=barh+ self.barsup[0][1]
            xdiff = self.barsdown[0][0] - self.playerpos[0]-birdrect[2]
        else:
            downy = self.barsdown[1][1]
            upy = barh + self.barsup[1][1]
            xdiff = self.barsdown[1][0] - self.playerpos[0]-birdrect[2]

        state = [self.playerpos[1], downy,upy, xdiff]

        return state

    def play(self,action,game_no=1):
        gameno=game_no
        self.reward=0
        # self.total_score+=self.reward
        self.bartime-=1
        self.playerpos[1] += 2.5

        if action == 1:
            self.playerpos[1] -= 5

        if(self.bartime==0):
            #enter new barw

            self.barno+=1
            self.bartime=70
            r = random.randint(0, 5)
            up_start=-25*r
            self.barsup.append([width,up_start])
            down_start=barh+up_start+120
            self.barsdown.append([width,down_start])
            coin_start=barh+up_start+45
            self.coins.append([width+barw/2,coin_start])
            self.popcoin = True


        for up,down in zip(self.barsup,self.barsdown):
            up[0]-=5
            down[0]-=5

        for co in self.coins:
            co[0]-=5

        if self.barsup[0][0] < -barw:
            self.barsup.pop(0)
            self.barsdown.pop(0)

            # print("deleted old bar")
        # 7 - update the screen

        if self.playerpos[1]>height or self.playerpos[1]<0:
            self.reward=-1000
            self.total_score+=self.reward
            self.done=True

        birdrect.left = self.playerpos[0]
        birdrect.top = self.playerpos[1]

        coinrect.left = self.coins[0][0]
        coinrect.top = self.coins[0][1]
        if self.popcoin and birdrect.colliderect(coinrect):
            self.reward = 10
            self.total_score += self.reward
            self.coins.pop(0)
            self.popcoin=False

        for up, down in zip(self.barsup, self.barsdown):
            baruprect.left = up[0]
            baruprect.top = up[1]
            bardownrect.left = down[0]
            bardownrect.top = down[1]

            if birdrect.colliderect(baruprect) or birdrect.colliderect(bardownrect):
                self.reward=-1000
                self.total_score+=self.reward
                self.done = True
                break

        if self.render:
            # 5 - clear the screen before drawing it again
            self.screen.fill(0)
            # 6 - draw the screen elements

            # for x in range(int(width/grass.get_width())+1):
            #     for y in range(int(height/grass.get_height())+1):
            #         self.screen.blit(grass,(x*100,y*100))

            for up, down in zip(self.barsup, self.barsdown):
                self.screen.blit(bar,up)
                self.screen.blit(bar,down)

            for co in zip(self.coins):
                self.screen.blit(coin,co)

            self.screen.blit(bird, self.playerpos)
                # 6.4 - Draw clock
            font = pygame.font.Font(None, 24)
            scoretxt = font.render("Episode "+str(gameno)+" score "+str(int(self.total_score)).zfill(2), True, (255,255,255))
            textRect = scoretxt.get_rect()
            textRect.topright = [width/2, 5]
            self.screen.blit(scoretxt, textRect)

            pygame.display.flip()

        old_state_frame=self.stack_frames.copy()
        state=self.getObservation()
        self.stack_frames.pop(0)
        self.stack_frames.append(state)
        current_state_frame=self.stack_frames.copy()
        env=(np.reshape(old_state_frame, [self.state_size]),action, self.reward, np.reshape(current_state_frame, [self.state_size]), self.done)

        return env

if __name__ == "__main__":
    render=True
    game=Game(render)
    for i in range(100):
        #start new game
        done=False
        action=0
        while done!=True:
            for event in pygame.event.get():
                # check if the event is the X button
                if event.type==pygame.QUIT:
                    # if it is quit the game
                    pygame.quit()
                    exit(0)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        keys[0] = True
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_w:
                        keys[0] = False

            # 9 - Move player
            if keys[0]:
                action=1
            else:
                action=0

            (state_frames, action, reward, next_state_frames, done)=game.play(action,i)
            # print(state, action, reward, next_state, done)

        game.reset(render)

