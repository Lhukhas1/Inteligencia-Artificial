#Aluno: Lucas Severino da Silva  -> RA: 2412152
# -*- coding: utf-8 -*-
"""
Recriação do Jogo da Velha

@author: Prof. Daniel Cavalcanti Jeronymo
"""

import pygame
    
import sys
import os
import traceback
import random
import numpy as np
import copy

class GameConstants:
    #                  R    G    B
    ColorWhite     = (255, 255, 255)
    ColorBlack     = (  0,   0,   0)
    ColorRed       = (255,   0,   0)
    ColorGreen     = (  0, 255,   0)
    ColorBlue     = (  0, 0,   255)
    ColorDarkGreen = (  0, 155,   0)
    ColorDarkGray  = ( 40,  40,  40)
    BackgroundColor = ColorBlack
    
    screenScale = 1
    screenWidth = screenScale*600
    screenHeight = screenScale*600
    
    # grid size in units
    gridWidth = 3
    gridHeight = 3
    
    # grid size in pixels
    gridMarginSize = 5
    gridCellWidth = screenWidth//gridWidth - 2*gridMarginSize
    gridCellHeight = screenHeight//gridHeight - 2*gridMarginSize
    
    randomSeed = 0
    
    FPS = 30
    
    fontSize = 20

class Game:
    class GameState:
        # 0 empty, 1 X, 2 O
        grid = np.zeros((GameConstants.gridHeight, GameConstants.gridWidth))
        currentPlayer = 0
    
    def __init__(self, expectUserInputs=True):
        self.expectUserInputs = expectUserInputs
        
        # Game state list - stores a state for each time step (initial state)
        gs = Game.GameState()
        self.states = [gs]
        
        # Determines if simulation is active or not
        self.alive = True
        
        self.currentPlayer = 1
        
        # Journal of inputs by users (stack)
        self.eventJournal = []
        
        
    def checkObjectiveState(self, gs):
        # Complete line?
        for i in range(3):
            s = set(gs.grid[i, :])
            if len(s) == 1 and min(s) != 0:
                return s.pop()
            
        # Complete column?
        for i in range(3):
            s = set(gs.grid[:, i])
            if len(s) == 1 and min(s) != 0:
                return s.pop()
            
        # Complete diagonal (main)?
        s = set([gs.grid[i, i] for i in range(3)])
        if len(s) == 1 and min(s) != 0:
            return s.pop()
        
        # Complete diagonal (opposite)?
        s = set([gs.grid[-i-1, i] for i in range(3)])
        if len(s) == 1 and min(s) != 0:
            return s.pop()
            
        # nope, not an objective state
        return 0
    
    
    # Implements a game tick
    # Each call simulates a world step
    def update(self):  
        # If the game is done or there is no event, do nothing
        if not self.alive or not self.eventJournal:
            return
        
        # Get the current (last) game state
        gs = copy.copy(self.states[-1])
        
        # Jogador 1
        if gs.currentPlayer == 0 or gs.currentPlayer == 1:
            if not self.eventJournal:
                return  
            x, y = self.eventJournal.pop()
            if gs.grid[x][y] != 0:
                return  
            gs.grid[x][y] = 1
            gs.currentPlayer = 2

        # Jogador 2 
        elif gs.currentPlayer == 2:
            move = self.get_bestMove()
            if move:
                x, y = move
                gs.grid[x][y] = 2
            gs.currentPlayer = 1

        # Check if end of game
        if self.checkObjectiveState(gs):
            self.alive = False
                        
        # Add the new modified state
        self.states += [gs]


    # MiniMax
    def miniMax(self, state, depth, is_maximizing):
        winner = self.checkObjectiveState(state)
        if winner == 1:
            return -5 + depth
        elif winner == 2:
            return 5 - depth
        elif winner == -1:
            return 0

        if is_maximizing:
            best_score = -float('inf')
            for row in range(3):
                for col in range(3):
                    if state.grid[row][col] == 0:
                        state.grid[row][col] = 2
                        score = self.miniMax(state, depth + 1, False)
                        state.grid[row][col] = 0
                        best_score = max(best_score, score)
            return best_score
        else:
            best_score = float('inf')
            for row in range(3):
                for col in range(3):
                    if state.grid[row][col] == 0:
                        state.grid[row][col] = 1
                        score = self.miniMax(state, depth + 1, True)
                        state.grid[row][col] = 0
                        best_score = min(best_score, score)
            return best_score

    def get_bestMove(self):
        best_score = -float('inf')
        best_move = None
        gs = copy.deepcopy(self.states[-1])

        for row in range(3):
            for col in range(3):
                if gs.grid[row][col] == 0:
                    gs.grid[row][col] = 2 
                    score = self.miniMax(gs, 0, False)
                    gs.grid[row][col] = 0

                    if score > best_score:
                        best_score = score
                        best_move = (row, col)

        return best_move


def drawGrid(screen, game):
    rects = []

    rects = [screen.fill(GameConstants.BackgroundColor)]
    
    # Get the current game state
    gs = game.states[-1]
    grid = gs.grid
 
    # Draw the grid
    for row in range(GameConstants.gridHeight):
        for column in range(GameConstants.gridWidth):
            color = GameConstants.ColorWhite
            
            if grid[row][column] == 1:
                color = GameConstants.ColorRed
            elif grid[row][column] == 2:
                color = GameConstants.ColorBlue
            
            m = GameConstants.gridMarginSize
            w = GameConstants.gridCellWidth
            h = GameConstants.gridCellHeight
            rects += [pygame.draw.rect(screen, color, [(2*m+w) * column + m, (2*m+h) * row + m, w, h])]    
    
    return rects


def draw(screen, font, game):
    rects = []
            
    rects += drawGrid(screen, game)

    return rects


def initialize():
    random.seed(GameConstants.randomSeed)
    pygame.init()
    game = Game()
    font = pygame.font.SysFont('Courier', GameConstants.fontSize)
    fpsClock = pygame.time.Clock()

    # Create display surface
    screen = pygame.display.set_mode((GameConstants.screenWidth, GameConstants.screenHeight), pygame.DOUBLEBUF)
    screen.fill(GameConstants.BackgroundColor)
        
    return screen, font, game, fpsClock


def handleEvents(game):
    #gs = game.states[-1]
    
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONUP:
            pos = pygame.mouse.get_pos()
            
            col = pos[0] // (GameConstants.screenWidth // GameConstants.gridWidth)
            row = pos[1] // (GameConstants.screenHeight // GameConstants.gridHeight)
            #print('clicked cell: {}, {}'.format(cellX, cellY))
            
            # send player action to game
            game.eventJournal.append((row, col))
            
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            pygame.quit()
            sys.exit()

            
def mainGamePlayer():
    try:
        # Initialize pygame and etc.
        screen, font, game, fpsClock = initialize()
              
        # Main game loop
        while game.alive:
            # Handle events
            handleEvents(game)
                    
            # Update world
            game.update()
            
            # Draw this world frame
            rects = draw(screen, font, game)     
            pygame.display.update(rects)
            
            # Delay for required FPS
            fpsClock.tick(GameConstants.FPS)
            
        # close up shop
        pygame.quit()
    except SystemExit:
        pass
    except Exception as e:
        #print("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
        pygame.quit()
        #raise Exception from e
    
    
if __name__ == "__main__":
    # Set the working directory (where we expect to find files) to the same
    # directory this .py file is in. You can leave this out of your own
    # code, but it is needed to easily run the examples using "python -m"
    file_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(file_path)

    mainGamePlayer()