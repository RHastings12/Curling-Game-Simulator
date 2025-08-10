# Ryan Hastings

import pygame, sys
import numpy as np
import random
import copy
from scipy.integrate import ode

# Set up the colors
BLACK = (1, 1, 1)
GREY = (150, 150, 150)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Top left corner is (0,0)
TEXT_AREA_HEIGHT = 40
WIN_WIDTH = 1800
WIN_HEIGHT = 200 + TEXT_AREA_HEIGHT

# Store default rock positions
rock_positions_red = [(10, 10 + TEXT_AREA_HEIGHT), (26, 10 + TEXT_AREA_HEIGHT), (42, 10 + TEXT_AREA_HEIGHT), (58, 10 + TEXT_AREA_HEIGHT),
                      (10, 26 + TEXT_AREA_HEIGHT), (26, 26 + TEXT_AREA_HEIGHT), (42, 26 + TEXT_AREA_HEIGHT), (58, 26 + TEXT_AREA_HEIGHT)]
rock_positions_yellow = [(10, WIN_HEIGHT - 10), (26, WIN_HEIGHT - 10), (42, WIN_HEIGHT - 10), (58, WIN_HEIGHT - 10),
                         (10, WIN_HEIGHT - 26), (26, WIN_HEIGHT - 26), (42, WIN_HEIGHT - 26), (58, WIN_HEIGHT - 26)]

# Lists of rocks to check for collisions
rocks = []
rocks_in_play = []
rocks_out_soon = []
rocks_moving = []
rocks_moving_soon = []
rocks_fgz = []    # "Free Guard Zone" rocks (cannot be taken out of play until after the 5th rock)

# Easy to know when end is over
rocks_to_go = []

# Keep track of the score
game_score = {'r': 0, 'y': 0}

NUM_ENDS = 8
rocks_thrown = 0    # Keeps track of how many rocks have been thrown in the current end

class Line:
    def __init__(self, start_pos, end_pos):
        self.start_pos = start_pos
        self.end_pos = end_pos

class Image(pygame.sprite.Sprite):
    def __init__(self, pos, imgfile):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.image.load(imgfile)
        self.rect = self.image.get_rect()
        self.pos = pos
        self.image_rot = self.image

    def rotate(self, angle):
        self.image_rot = pygame.transform.rotate(self.image, angle)

    def move(self, x, y):
        self.pos = (x, y)
        self.rect.center = self.pos
    
    def draw(self, surface):
        rect = self.image_rot.get_rect()
        rect.center = self.pos
        # image = pygame.Surface([rect.width, rect.height])
        # surface.blit(image, rect)
        surface.blit(self.image_rot, rect)

class Circle(pygame.sprite.Sprite):
    def __init__(self, colour, diameter, pos):
        pygame.sprite.Sprite.__init__(self)
        
        self.radius = diameter // 2
        self.image = pygame.Surface([diameter, diameter])
        self.image.set_colorkey((0, 0, 0, 0))
        self.rect = self.image.get_rect()

        pygame.draw.circle(self.image, colour, self.rect.center, diameter//2)
        self.move(pos[0], pos[1])

    def move(self, x, y):
        self.pos = (x, y)
        self.rect.center = self.pos

class Rect(pygame.sprite.Sprite):
    def __init__(self, colour, width, height, pos):
        pygame.sprite.Sprite.__init__(self)
        
        self.colour = colour
        self.image_orig = pygame.Surface([width, height])
        self.image_orig.set_colorkey((0, 0, 0, 0))
        self.image = self.image_orig
        self.rect = self.image.get_rect()

        pygame.draw.rect(self.image, colour, self.rect)
        self.move(pos[0], pos[1])
    
    def rotate(self, angle):
        self.image = pygame.transform.rotozoom(self.image_orig, angle, 1.0)
        self.rect = self.image.get_rect()

    def move(self, x, y):
        self.pos = (x, y)
        self.rect.center = self.pos

class RoundRect(pygame.sprite.Sprite):
    def __init__(self, colour, width, height, pos, border_rad):
        pygame.sprite.Sprite.__init__(self)
        
        self.colour = colour
        self.image_orig = pygame.Surface([width, height])
        self.image_orig.set_colorkey((0, 0, 0, 0))
        self.image = self.image_orig
        self.rect = self.image.get_rect()

        pygame.draw.rect(self.image, colour, self.rect, border_radius = border_rad)
        self.move(pos[0], pos[1])
    
    def rotate(self, angle):
        self.image = pygame.transform.rotate(self.image_orig, angle)

    def move(self, x, y):
        self.pos = (x, y)
        self.rect.center = self.pos

class Text(pygame.sprite.Sprite):
    def __init__(self, color, background=BLACK, antialias=True, fontname="calibri", fontsize=12):
        pygame.font.init()
        self.font = pygame.font.SysFont(fontname, fontsize)
        self.color = color
        self.background = background
        self.antialias = antialias
    
    def draw(self, str1, screen, pos):
        text = self.font.render(str1, self.antialias, self.color, self.background)
        screen.blit(text, pos)

class Bar:
    def __init__(self, width, pos):
        self.bar_outline = Rect(BLUE, width, 30, pos)
        self.bar_back = Rect(GREY, width - 10, 20, pos)
        self.bar = Rect(GREY, width - 10, 20, pos)

        self.sprites = pygame.sprite.Group([self.bar_outline, self.bar_back, self.bar])

class House:
    def __init__(self, pos):
        self.twelve_foot = Circle(BLUE, np.floor((WIN_HEIGHT - TEXT_AREA_HEIGHT) * 0.8), pos)
        self.eight_foot = Circle(WHITE, np.floor((WIN_HEIGHT - TEXT_AREA_HEIGHT) * 0.6), pos)
        self.four_foot = Circle(RED, np.floor((WIN_HEIGHT - TEXT_AREA_HEIGHT) * 0.4), pos)
        self.button = Circle(WHITE, np.floor((WIN_HEIGHT - TEXT_AREA_HEIGHT) * 0.2), pos)

        self.sprites = pygame.sprite.Group([self.twelve_foot, self.eight_foot, self.four_foot, self.button])

class Ice:
    def __init__(self):
        self.curl = random.randint(1, 5)
        self.speed = random.randint(1, 5)

class Rock:
    def __init__(self, pos, sprite, rock_num, rock_colour):
        self.rock_num = rock_num
        self.rock_colour = rock_colour
        self.sprite = sprite
        self.pos = pos
        self.prev_pos = pos
        self.mass = 18
        self.diameter = 16
        self.distance = 0
        self.power = 0
        self.rotation = 0
        self.direction_dangle = 0
        self.Ibody = np.identity(3)           # inertia tensor
        self.IbodyInv = np.linalg.inv(self.Ibody)  # inverse of inertia tensor
        self._R = np.identity(3)
        self.Iinv = np.dot(self._R, np.dot(self.IbodyInv, self._R.T))
        self.I = np.linalg.inv(self.Iinv)
        self.v = np.zeros(3)       # linear velocity
        self.omega = np.zeros(3)   # angular velocity
        self.t = 0.0
        self.dt = 0.05

        self.state = np.zeros(19)
        self.state[0:3] = np.array(pos)            # position (C.O.M.)
        self.state[3:12] = self._R.reshape([1,9])  # rotation
        self.state[12:15] = self.mass * self.v    # linear momentum
        self.state[15:18] = np.zeros(3)                   # angular momentum

        # Setting up the solver
        self.solver = ode(self.f)
        self.solver.set_integrator('dop853')
    
    def f(self, t, state):
        rate = np.zeros(19)
        rate[0:3] = self.v

        speed = np.sqrt(self.v[0]**2 + self.v[1]**2 + self.v[2]**2)

        # Update direction of rock after hitting "break point" (rough estimate)
        if t >= 9 + (abs(self.rotation) * 0.2):
            direction_rotation_matrix = np.array([[np.cos(np.deg2rad(self.direction_dangle)), -np.sin(np.deg2rad(self.direction_dangle)), 0.0],
                                                    [np.sin(np.deg2rad(self.direction_dangle)), np.cos(np.deg2rad(self.direction_dangle)), 0.0],
                                                    [0.0, 0.0, 1.0]])
            if speed != 0:
                direction_axis = np.dot(self.v / speed, direction_rotation_matrix)
            else:
                direction_axis = np.zeros(3)
        else:
            if speed != 0:
                direction_axis = self.v / speed
            else:
                direction_axis = np.zeros(3)

        _R = state[3:12].reshape([3,3])
        _R = self.orthonormalize(_R)
        IbodyInv = np.identity(3)
        Iinv = np.dot(_R, np.dot(IbodyInv, _R.T))
        _L = state[15:18]
        omega = np.dot(Iinv, _L)

        force = ((-45 - (10 / self.ice_conditions.speed) + (abs(self.omega[2]) * 1.5))) * direction_axis      # Higher rotation makes rock travel farther

        torque = np.array([0, 0, -omega[2] * (1 / (30 + (speed * 0.1)))])

        rate[3:12] = np.dot(self.star(omega), _R).reshape([1,9])
        rate[12:15] = force
        rate[15:18] = torque
        return rate
    
    def star(self, v):
        vs = np.zeros([3,3])
        vs[0][0] = 0
        vs[1][0] = v[2]
        vs[2][0] = -v[1]
        vs[0][1] = -v[2]
        vs[1][1] = 0
        vs[2][1] = v[0]
        vs[0][2] = v[1] 
        vs[1][2] = -v[0]
        vs[2][2] = 0
        return vs;       

    def orthonormalize(self, m):
        mo = np.zeros([3,3])
        r0 = m[0,:]
        r1 = m[1,:]
        r2 = m[2,:]
        
        r0new = r0 / np.linalg.norm(r0)
        
        r2new = np.cross(r0new, r1)
        r2new = r2new / np.linalg.norm(r2new)

        r1new = np.cross(r2new, r0new)
        r1new = r1new / np.linalg.norm(r1new)

        mo[0,:] = r0new
        mo[1,:] = r1new
        mo[2,:] = r2new
        return mo

    def update(self):

        rock_dt = self.dt

        # Lower dt if velocity is too small
        if abs(self.v[0]) < 0.5:
            rock_dt = self.dt / 5
        elif abs(self.v[0]) < 0.1:
            rock_dt = self.dt / 10
        
        new_state = self.solver.integrate(self.t + rock_dt)
        collision = False

        # Collision detection
        for played_rock in rocks_in_play:
            if played_rock != self:
                played_rock_dt = played_rock.dt
                new_played_rock_state = played_rock.solver.integrate(played_rock.t + played_rock_dt)

                n = new_played_rock_state[0:3] - new_state[0:3]
                distance = np.sqrt(n[0]**2 + n[1]**2 + n[2]**2)

                if self.is_collision(distance, played_rock):
                    collision = True

                    # Get rock states at exact collision time
                    state, played_rock_state = self.respond_to_collision(new_state, new_played_rock_state, played_rock, self.t + rock_dt, played_rock.t + played_rock_dt)

                    # Collision response
                    n = played_rock_state[0:3] - state[0:3]
                    distance = np.sqrt(n[0]**2 + n[1]**2 + n[2]**2)

                    n_hat = n / distance
                    v_a1 = played_rock_state[12:15] / played_rock.mass
                    v_b1 = state[12:15] / self.mass
                    omega_a1 = np.dot(played_rock.Iinv, played_rock_state[15:18])
                    omega_b1 = np.dot(self.Iinv, state[15:18])

                    v_ab = v_a1 - v_b1

                    e = 1
                    j = -((1 + e) * np.dot(v_ab, n_hat)) / ((1 / played_rock.mass) + (1 / self.mass))
                    
                    # Calculate final velocity of first rock
                    v_a2 = (v_a1 + ((j * n_hat) / played_rock.mass)) * 0.85
                    omega_a2 = copy.deepcopy(omega_a1)

                    # Calculate final velocity of second rock
                    v_b2 = (v_b1 - ((j * n_hat) / self.mass)) * 0.85
                    omega_b2 = copy.deepcopy(omega_b1)

                    v_b1_dist = np.sqrt(v_b1[0]**2 + v_b1[1]**2 + v_b1[2]**2)
                    v_b2_dist = np.sqrt(v_b2[0]**2 + v_b2[1]**2 + v_b2[2]**2)

                    # Angular velocity scenarios
                    if omega_b1[2] < 0:
                        if np.dot(v_b1, rotate_vec(v_b2, -90)) <= 0:
                            omega_a2[2] += (-omega_b1[2] * 0.5) - (omega_b1[2] * (v_b2_dist / v_b1_dist))
                            omega_b2 *= -0.8 * (v_b2_dist / v_b1_dist)
                        elif np.dot(v_b1, rotate_vec(v_b2, 90)) <= 0:
                            omega_a2[2] += (-omega_b1[2] * 0.5) + (omega_b1[2] * (v_b2_dist / v_b1_dist))
                            omega_b2 *= 2 * (v_b2_dist / v_b1_dist)
                    elif omega_b1[2] > 0:
                        if np.dot(v_b1, rotate_vec(v_b2, 90)) <= 0:
                            omega_a2[2] += (-omega_b1[2] * 0.5) - (omega_b1[2] * (v_b2_dist / v_b1_dist))
                            omega_b2 *= -0.8 * (v_b2_dist / v_b1_dist)
                        elif np.dot(v_b1, rotate_vec(v_b2, -90)) <= 0:
                            omega_a2[2] += (-omega_b1[2] * 0.5) + (omega_b1[2] * (v_b2_dist / v_b1_dist))
                            omega_b2 *= 2 * (v_b2_dist / v_b1_dist)

                    self.update_state(state, 0)
                    played_rock.update_state(played_rock_state, 0)

                    # Reinitialize velocities
                    played_rock.init_velocity(v_a2, omega_a2)
                    self.init_velocity(v_b2, omega_b2)

                    self.solver.set_initial_value(state, 0)
                    played_rock.solver.set_initial_value(played_rock_state, 0)

                    rocks_moving_soon.append(played_rock)
        
        if not collision:
            self.update_state(new_state, self.t + rock_dt)
    
    def update_state(self, state, t):
        self.state = state
        self.t = t
        self.pos = state[0:3]
        self.v = state[12:15] / self.mass
        self.omega = np.dot(self.Iinv, state[15:18])
    
    def is_collision(self, distance, played_rock):
        return distance < played_rock.diameter / 2 + self.diameter / 2
    
    def respond_to_collision(self, state, played_rock_state, played_rock, rock_t, played_rock_t):
        rock_dt = self.dt / 2    # Decrease time step size
        played_rock_dt = played_rock.dt / 2

        dstate = state
        dplayed_rock_state = played_rock_state

        while True:

            n = dplayed_rock_state[0:3] - dstate[0:3]
            distance = np.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
            difference = abs(distance - ((played_rock.diameter / 2) + (self.diameter / 2)))

            # Break when desired accuracy is achieved
            if difference <= 0.000001:
                break
            
            if distance > (played_rock.diameter / 2) + (self.diameter / 2):
                # Advance by smaller time step
                self.solver.set_initial_value(dstate, rock_t)
                played_rock.solver.set_initial_value(dplayed_rock_state, played_rock_t)

                dstate = self.solver.integrate(rock_t + rock_dt)
                if played_rock in rocks_moving:
                    dplayed_rock_state = played_rock.solver.integrate(played_rock_t + played_rock_dt)

                rock_t += rock_dt
                played_rock_t += played_rock_dt
            elif distance < (played_rock.diameter / 2) + (self.diameter / 2):
                # Backup by smaller time step
                self.solver.set_initial_value(dstate, rock_t)
                played_rock.solver.set_initial_value(dplayed_rock_state, played_rock_t)

                dstate = self.solver.integrate(rock_t - rock_dt)
                if played_rock in rocks_moving:
                    dplayed_rock_state = played_rock.solver.integrate(played_rock_t - played_rock_dt)

                rock_t -= rock_dt
                played_rock_t -= played_rock_dt

            rock_dt = rock_dt / 2    # Decrease time step size
            played_rock_dt = played_rock_dt / 2
        
        return dstate, dplayed_rock_state
    
    def init_velocity(self, v, omega):
        self.v = v
        self.omega = omega

        self.I = np.linalg.inv(self.Iinv)
        self.state[12:15] = self.mass * self.v
        self.state[15:18] = np.dot(self.I, omega)
    
    # Remove rock from play
    def remove_from_play(self):
        self.v = np.zeros(3)
        self.state[12:15] = np.zeros(3)
        self.omega = np.zeros(3)
        self.state[15:18] = np.zeros(3)
        if self in rocks_in_play:
            rocks_in_play.remove(self)
        
        if self in rocks_moving:
            rocks_moving.remove(self)

        self.sprite.image_rot = self.sprite.image
        if self.rock_colour == 'r':
            self.sprite.move(WIN_WIDTH - rock_positions_red[self.rock_num][0], rock_positions_red[self.rock_num][1])
            self.pos = (WIN_WIDTH - rock_positions_red[self.rock_num][0], rock_positions_red[self.rock_num][1])
            self.prev_pos = self.pos
        elif self.rock_colour == 'y':
            self.sprite.move(WIN_WIDTH - rock_positions_yellow[self.rock_num][0], rock_positions_yellow[self.rock_num][1])
            self.pos = (WIN_WIDTH - rock_positions_yellow[self.rock_num][0], rock_positions_yellow[self.rock_num][1])
            self.prev_pos = self.pos
        else:
            print("Invalid rock colour")
    
    # Reset the rock to be ready for the next end
    def reset(self):
        self.sprite.image_rot = self.sprite.image
        self.distance = 0
        self.power = 0
        self.rotation = 0
        self.direction_dangle = 0
        self.Ibody = np.identity(3)           # inertia tensor
        self.IbodyInv = np.linalg.inv(self.Ibody)  # inverse of inertia tensor
        self._R = np.identity(3)
        self.Iinv = np.dot(self._R, np.dot(self.IbodyInv, self._R.T))
        self.I = np.linalg.inv(self.Iinv)
        self.v = np.zeros(3)       # linear velocity
        self.omega = np.zeros(3)   # angular velocity
        self.t = 0

        self.state = np.zeros(19)
        self.state[0:3] = np.array(self.pos)            # position (C.O.M.)
        self.state[3:12] = self._R.reshape([1,9])  # rotation
        self.state[12:15] = self.mass * self.v    # linear momentum
        self.state[15:18] = np.zeros(3)                   # angular momentum

        self.solver.set_initial_value(self.state, 0)
    
    def get_angle_2d(self):
        v1 = [1,0,0]
        v2 = np.dot(self.state[3:12].reshape([3,3]), v1)
        cosang = np.dot(v1, v2)
        axis = np.cross(v1, v2)
        return np.degrees(np.arccos(cosang)), axis

    def prn_state(self):
        print('Pos', self.state[0:3])
        print('Rot', self.state[3:12].reshape([3,3]))
        print('P', self.state[12:15])
        print('L', self.state[15:18])

class Simulation:
    def __init__(self, rocks):
        self.paused = True         # Starting in paused mode
        self.rocks = rocks
        self.ice_conditions = Ice()
        self.active_rock = 0       # 0-7: red rocks, 8-15: yellow rocks
        self.active = False

        for rock in rocks:
            rock.ice_conditions = self.ice_conditions

    def update(self):
        # Update rock positions
        for rock in rocks_moving:
            rock.update()
            rock.sprite.move(rock.pos[0], rock.pos[1])
        
        for new_rock in rocks_moving_soon:
            rocks_moving.append(new_rock)
            rocks_moving_soon.remove(new_rock)
    
    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

def rotate_vec(v, angle):
    r = np.array([[np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0.0],
                  [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 0.0],
                  [0.0, 0.0, 1.0]])
    return np.dot(v, r)

# Calculate the score after an end
def calculate_score(house):
    # Center and radius of the house
    center = house.twelve_foot.pos
    radius = house.twelve_foot.radius

    rock_order = []    # Ascending-ordered list of all rocks in the house
                       # by distance from the pin

    # Get list of all rocks that finished the end in the house
    for rock in rocks_in_play:
        rock_pos = (rock.pos[0], rock.pos[1])
        distance_vector = np.array(rock_pos) - np.array(center)
        rock.distance = np.sqrt(distance_vector[0]**2 + distance_vector[1]**2) - (rock.diameter / 2)
        if rock.distance <= radius:
            rock_order.append(rock)
    
    # Sort list of rocks ascending by distance from pin and count the score
    if len(rock_order) > 0:
        rock_order.sort(key = lambda rock : rock.distance)
        colour_scored = rock_order[0].rock_colour
        for rock in rock_order:
            if rock.rock_colour == colour_scored:
                game_score[colour_scored] += 1
            else:
                break
        return colour_scored
    
    return ''

def reset_fgz():
    for rock in rocks_in_play:
        rock.pos = rock.prev_pos
        rock.state[0:3] = rock.pos
        rock.sprite.move(rock.pos[0], rock.pos[1])

        rock.v = np.zeros(3)
        rock.state[12:15] = np.zeros(3)
        rock.omega = np.zeros(3)
        rock.state[15:18] = np.zeros(3)
        
        if rock in rocks_moving:
            rocks_moving.remove(rock)

def main():
    pygame.init()

    clock = pygame.time.Clock()

    screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption('Curling Game')
    
    # Starting rock position
    rock_start_pos = (60, (WIN_HEIGHT // 2) + (TEXT_AREA_HEIGHT // 2))

    # Create rocks
    for i in range(16):
        if i < 8:
            rock_sprite = Image((rock_positions_red[i][0], rock_positions_red[i][1]), 'images/red_rock.png')
            rock = Rock([rock_positions_red[i][0], rock_positions_red[i][1], 0], rock_sprite, i, 'r')
            rocks.append(rock)
            rocks_to_go.append(rock)
        else:
            rock_sprite = Image((rock_positions_yellow[i - 8][0], rock_positions_yellow[i - 8][1]), 'images/yellow_rock.png')
            rock = Rock([rock_positions_yellow[i - 8][0], rock_positions_yellow[i - 8][1], 0], rock_sprite, i - 8, 'y')
            rocks.append(rock)
            rocks_to_go.append(rock)
    
    # Create houses
    houses = [House((150, (WIN_HEIGHT // 2) + (TEXT_AREA_HEIGHT // 2))), House((WIN_WIDTH - 150, (WIN_HEIGHT // 2) + (TEXT_AREA_HEIGHT // 2)))]
    
    # Create hacks
    hack0 = Rect(BLACK, 20, 12, (30, ((WIN_HEIGHT // 2) + (TEXT_AREA_HEIGHT // 2)) - 20))
    hack1 = Rect(BLACK, 20, 12, (30, ((WIN_HEIGHT // 2) + (TEXT_AREA_HEIGHT // 2)) + 20))
    hack2 = Rect(BLACK, 20, 12, (WIN_WIDTH - 30, (WIN_HEIGHT // 2) + (TEXT_AREA_HEIGHT // 2) - 20))
    hack3 = Rect(BLACK, 20, 12, (WIN_WIDTH - 30, (WIN_HEIGHT // 2) + (TEXT_AREA_HEIGHT // 2) + 20))
    hacks = pygame.sprite.Group([hack0, hack1, hack2, hack3])
    
    # Create lines
    back_line0 = Rect(BLACK, 1, WIN_HEIGHT - TEXT_AREA_HEIGHT, (70, (WIN_HEIGHT // 2) + (TEXT_AREA_HEIGHT // 2)))
    back_line1 = Rect(BLACK, 1, WIN_HEIGHT - TEXT_AREA_HEIGHT, (WIN_WIDTH - 70, (WIN_HEIGHT // 2) + (TEXT_AREA_HEIGHT // 2)))
    t_line0 = Rect(BLACK, 1, WIN_HEIGHT - TEXT_AREA_HEIGHT, (150, (WIN_HEIGHT // 2) + (TEXT_AREA_HEIGHT // 2)))
    t_line1 = Rect(BLACK, 1, WIN_HEIGHT - TEXT_AREA_HEIGHT, (WIN_WIDTH - 150, (WIN_HEIGHT // 2) + (TEXT_AREA_HEIGHT // 2)))
    center_line = Rect(BLACK, WIN_WIDTH, 1, (WIN_WIDTH // 2, (WIN_HEIGHT // 2) + (TEXT_AREA_HEIGHT // 2)))
    hog_line0 = Rect(BLACK, 10, WIN_HEIGHT - TEXT_AREA_HEIGHT, (450, (WIN_HEIGHT // 2) + (TEXT_AREA_HEIGHT // 2)))
    hog_line1 = Rect(BLACK, 10, WIN_HEIGHT - TEXT_AREA_HEIGHT, (WIN_WIDTH - 450, (WIN_HEIGHT // 2) + (TEXT_AREA_HEIGHT // 2)))
    lines = pygame.sprite.Group([back_line0, back_line1, t_line0, t_line1, center_line, hog_line0, hog_line1])

    # Create text
    text_area = Rect((50, 50, 50), WIN_WIDTH, TEXT_AREA_HEIGHT, (0, 0))
    score = Text(WHITE, (50, 50, 50), fontsize = 18)
    end = Text(WHITE, (50, 50, 50), fontsize = 18)
    sim_controls = Text(WHITE, (50, 50, 50), fontsize = 18)
    controls = Text(WHITE, (50, 50, 50), fontsize = 30)
    rock_controls = Text(WHITE, (50, 50, 50), fontsize = 35)
    variables = Text(WHITE, (50, 50, 50), fontsize = 18)
    next_colour = Text(WHITE, (50, 50, 50), fontsize = 40)
    paused = Text(WHITE, (50, 50, 50), fontsize = 40)
    rotation = Text(WHITE, (50, 50, 50), fontsize = 14)

    # Create bars
    power_bar = Bar(160, ((WIN_WIDTH // 2) - 310, TEXT_AREA_HEIGHT // 2))
    rotation_bar = Bar(70, ((WIN_WIDTH // 2) + 280, TEXT_AREA_HEIGHT // 2))

    rotation_bar.bar.image.fill(RED, (rotation_bar.bar.image.get_width() // 2 - 10, 0, 20, 20))

    bars = [power_bar, rotation_bar]

    # Create aim line
    start_pos = pygame.math.Vector2(rock_start_pos)
    end_pos = pygame.math.Vector2(2000, rock_start_pos[1])
    aim_vector = end_pos - start_pos
    aim = Line(start_pos, end_pos)

    sim = Simulation(rocks)
    end_num = 1

    controls_show = False
    
    first_colour = 'r'
    coin_flip = random.randint(0, 1)
    if coin_flip == 1:
        first_colour = 'y'
        sim.active_rock = 8
    

    # Start screen
    title = Text(WHITE, (50, 50, 50), fontsize = 50)
    resume = Text(WHITE, (50, 50, 50), fontsize = 30)
    while True:
        screen.fill((50, 50, 50))

        title.draw("CURLING SIMULATOR", screen, (WIN_WIDTH // 2 - 200, WIN_HEIGHT // 2 - 50))
        resume.draw("Press 'R' to start", screen, (WIN_WIDTH // 2 - 80, WIN_HEIGHT // 2 + 30))
        pygame.display.flip()

        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)

        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            screen.fill((50, 50, 50))
            if first_colour == 'r':
                next_colour.draw("RED START ", screen, (WIN_WIDTH // 2 - 100, WIN_HEIGHT // 2 - 30))
            elif first_colour == 'y':
                next_colour.draw("YELLOW START ", screen, (WIN_WIDTH // 2 - 120, WIN_HEIGHT // 2 - 30))
            pygame.display.flip()
            pygame.time.wait(1500)

            sim.resume()
            break
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            pygame.quit()
            sys.exit(0)

    angle = 0
    dangle = 0.025
    
    global rocks_thrown

    # Main game loop
    while True:
        # 60 fps
        clock.tick(60)

        # for rock in rocks_in_play:
        #     print(rock.pos)
        rock = sim.rocks[sim.active_rock]

        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)

        if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
            sim.pause()
            continue
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            sim.resume()
            controls_show = False
            continue
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
            controls_show = True

        # Pause screen
        if sim.paused:
            screen.fill((50, 50, 50))
            if controls_show == True:
                rock_controls.draw("UP/DOWN  - Adjust POWER   |   LEFT/RIGHT  - Adjust AIM", screen, (525, WIN_HEIGHT // 2 - 100))
                rock_controls.draw("W/S  - Adjust ROTATION", screen, (770, WIN_HEIGHT // 2 - 30))
                rotation.draw("Slider left = CLOCKWISE  |  Slider right = COUNTERCLOCKWISE  |  Slider middle = ZERO", screen, (690, WIN_HEIGHT // 2 + 20))
                resume.draw("Press 'R' to return to game", screen, (WIN_WIDTH // 2 - 125, WIN_HEIGHT // 2 + 70))
            else:
                paused.draw("PAUSED", screen, (WIN_WIDTH // 2 - 52, WIN_HEIGHT // 2 - 50))
                controls.draw("Press 'C' for controls   |   Press 'R' to resume", screen, (WIN_WIDTH // 2 - 260, WIN_HEIGHT // 2 + 30))
                
            pygame.display.flip()
            continue

        if not sim.active and not sim.paused:

            rock.direction_axis = np.array([1, 0, 0])

            keys = pygame.key.get_pressed()

            # Increase power
            if keys[pygame.K_UP]:
                if rock.power < 150:
                    rock.power += 1
                    power_bar.bar.image.fill(YELLOW, (0, 0, rock.power, 20))
            
            # Decrease power
            if keys[pygame.K_DOWN]:
                if rock.power > 0:
                    rock.power -= 1
                    power_bar.bar.image.fill(GREY)
                    power_bar.bar.image.fill(YELLOW, (0, 0, rock.power, 20))
            
            # Increase clockwise rotation
            if keys[pygame.K_w]:
                if rock.rotation > -20:
                    rock.rotation -= 1
                    rotation_bar.bar.image.fill(GREY)
                    if rock.rotation >= 0:
                        rotation_bar.bar.image.fill(RED, (rotation_bar.bar.image.get_width() // 2 - 10 + abs(rock.rotation), 0, 20, 20))
                    else:
                        rotation_bar.bar.image.fill(RED, (rotation_bar.bar.image.get_width() // 2 - 10 - abs(rock.rotation), 0, 20, 20))
            
            # Increase counter-clockwise rotation
            if keys[pygame.K_s]:
                if rock.rotation < 20:
                    rock.rotation += 1
                    rotation_bar.bar.image.fill(GREY)
                    if rock.rotation >= 0:
                        rotation_bar.bar.image.fill(RED, (rotation_bar.bar.image.get_width() // 2 - 10 + abs(rock.rotation), 0, 20, 20))
                    else:
                        rotation_bar.bar.image.fill(RED, (rotation_bar.bar.image.get_width() // 2 - 10 - abs(rock.rotation), 0, 20, 20))

            # Aim left
            if keys[pygame.K_LEFT]:
                if angle < 30:
                    angle += dangle
                    aim_vector = aim.end_pos - aim.start_pos
                    aim_vector = aim_vector.rotate(-angle)

            # Aim right
            if keys[pygame.K_RIGHT]:
                if angle > -30:
                    angle -= dangle
                    aim_vector = aim.end_pos - aim.start_pos
                    aim_vector = aim_vector.rotate(-angle)

            # Throw the rock
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                sim.active = True
                rocks_moving.append(rock)      # Add the rock to the list of currently moving rocks
                rocks_in_play.append(rock)     # Add the rock to the list of currently in-play rocks
                rocks_thrown += 1

                # Control the direction and magnitude of curl
                if rock.rotation < 0:
                    rock.direction_dangle = (sim.ice_conditions.curl + 8) * 32 / (abs(rock.rotation) + 20)
                    omega = np.array([0, 0, rock.rotation * 0.2])
                elif rock.rotation > 0:
                    rock.direction_dangle = -1 * (sim.ice_conditions.curl + 8) * 32 / (abs(rock.rotation) + 20)
                    omega = np.array([0, 0, rock.rotation * 0.2])
                else:
                    rock.direction_dangle = random.randint(-50, 50)
                    omega = np.array([0, 0, 0])

                # Initialize velocity and get the ode ready
                init_rot_matrix = np.array([[np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0.0],
                                            [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 0.0],
                                            [0.0, 0.0, 1.0]])
                v = np.dot(np.array([(rock.power + 300) * 0.26, 0, 0]), init_rot_matrix)
                rock.init_velocity(v, omega)
                rock.solver.set_initial_value(rock.state, 0)
                rocks_to_go.remove(rock)

        # Position next rock
        if not sim.active and not sim.paused:
            rock.sprite.move(rock_start_pos[0], rock_start_pos[1])
            rock.state[0:3] = np.array([rock.sprite.pos[0], rock.sprite.pos[1], 0])
            rock.pos = rock.state[0:3]
        
        # Clear screen
        screen.fill(WHITE)

        # Draw houses
        for house in houses:
            house.sprites.draw(screen)
        hacks.draw(screen)
        lines.draw(screen)

        # Draw aim line
        if not sim.active and not sim.paused:
            pygame.draw.aaline(screen, GREEN, rock_start_pos, aim.start_pos + aim_vector)

        # Draw rock sprites
        for rocksp in rocks:
            rock_angle, axis = rocksp.get_angle_2d()
            if axis[2] < 0:
                rock_angle *= -1.

            rocksp.sprite.rotate(rock_angle)

            rocksp.sprite.draw(screen)
        
        # Draw text
        screen.blit(text_area.image, (0, 0))
        variables.draw("POWER", screen, (WIN_WIDTH // 2 - 460, 10))
        variables.draw("ROTATION", screen, (WIN_WIDTH // 2 + 330, 10))
        score.draw(f"RED  {game_score['r']}  |  {game_score['y']}  YELLOW", screen, (WIN_WIDTH - 200, 10))
        end.draw(f"END {end_num}", screen, (WIN_WIDTH // 2 - 25, 10))
        sim_controls.draw("PAUSE  - P | QUIT  - Q", screen, (10, 10))
        
        # Draw bars
        for bar in bars:
            bar.sprites.draw(screen)

        pygame.display.flip()

        # Update after throwing the rock
        if sim.active and not sim.paused:
            sim.update()

            if rocks_thrown > 5:
                rocks_fgz.clear()

            # Check for any rocks that are still moving
            for rock in rocks_moving:
                # Check if rock is past the back line
                if rock.pos[0] - (rock.diameter // 2) > WIN_WIDTH - 70:
                    if rock in rocks_fgz and rocks[sim.active_rock].rock_colour != rock.rock_colour:
                        reset_fgz()
                        rocks[sim.active_rock].remove_from_play()
                        break
                    else:
                        rocks_out_soon.append(rock)
                
                # Check if rock has hit the sides
                if rock.pos[1] - (rock.diameter // 2) <= TEXT_AREA_HEIGHT or rock.pos[1] + (rock.diameter // 2) >= WIN_HEIGHT:
                    if rock in rocks_fgz and rocks[sim.active_rock].rock_colour != rock.rock_colour:
                        reset_fgz()
                        rocks[sim.active_rock].remove_from_play()
                        break
                    else:
                        rocks_out_soon.append(rock)

                # Check if rock has stopped
                if abs(rock.v[0]) <= 0.05 and abs(rock.v[1]) <= 0.05:
                    rocks_moving.remove(rock)
                    rock.v = np.zeros(3)
                    rock.state[12:15] = np.zeros(3)
                    rock.omega = np.zeros(3)
                    rock.state[15:18] = np.zeros(3)

                    # Center and radius of the house
                    center = house.twelve_foot.pos
                    radius = house.twelve_foot.radius
                    rock_pos = (rock.pos[0], rock.pos[1])
                    distance_vector = np.array(rock_pos) - np.array(center)
                    rock.distance = np.sqrt(distance_vector[0]**2 + distance_vector[1]**2) - (rock.diameter / 2)

                    # Check if rock stopped before the hog line
                    if rock.pos[0] - (rock.diameter // 2) <= WIN_WIDTH - 445:
                        rocks_out_soon.append(rock)

                    elif rock.distance > radius and rock not in rocks_fgz:
                        rock.prev_pos = rock.pos
                        rocks_fgz.append(rock)

                    else:
                        if rock in rocks_fgz:
                            rocks_fgz.remove(rock)

                        rock.prev_pos = rock.pos
            
            for rock in rocks_out_soon:
                rock.remove_from_play()
                rocks_out_soon.remove(rock)
            
            # Change active rock
            if len(rocks_moving) == 0:
                sim.active = False
                angle = 0

                rock = rocks[sim.active_rock]
                rock.power = 0
                rock.rotation = 0
                power_bar.bar.image.fill(GREY)
                rotation_bar.bar.image.fill(GREY)
                rotation_bar.bar.image.fill(RED, (rotation_bar.bar.image.get_width() // 2 - 10, 0, 20, 20))
                aim_vector = end_pos - start_pos

                # Determine the next rock
                if first_colour == 'r':
                    if sim.active_rock < 8:
                        sim.active_rock += 8
                    else:
                        sim.active_rock -= 7
                elif first_colour == 'y':
                    if sim.active_rock >= 8:
                        sim.active_rock -= 8
                    else:
                        sim.active_rock += 9
                
                # Calculate score and reset rocks
                if len(rocks_to_go) == 0:

                    colour_scored = calculate_score(houses[1])   # Calculate score

                    # Change colour that goes first
                    if colour_scored == '':
                        if first_colour == 'r':
                            sim.active_rock = 0
                        elif first_colour == 'y':
                            sim.active_rock = 8
                    else:
                        first_colour = colour_scored
                        if colour_scored == 'r':
                            sim.active_rock = 0
                        elif colour_scored == 'y':
                            sim.active_rock = 8

                    # Reset rocks
                    for i in range(len(rocks)):
                        if i < 8:
                            rocks[i].pos = [rock_positions_red[i][0], rock_positions_red[i][1], 0]
                            rocks[i].sprite.move(rock_positions_red[i][0], rock_positions_red[i][1])
                        else:
                            rocks[i].pos = [rock_positions_yellow[i - 8][0], rock_positions_yellow[i - 8][1], 0]
                            rocks[i].sprite.move(rock_positions_yellow[i - 8][0], rock_positions_yellow[i - 8][1])
                        
                        rocks[i].reset()
                        rocks_to_go.append(rocks[i])

                    rocks_in_play.clear()
                    rocks_moving.clear()

                    end_num += 1
                    pygame.time.wait(500)

                    # Declare winner, or continue playing if tied
                    if end_num > NUM_ENDS:
                        if game_score['r'] > game_score['y']:
                            screen.fill((50, 50, 50))
                            title.draw("RED WINS", screen, (WIN_WIDTH // 2 - 100, WIN_HEIGHT // 2 - 20))
                            score.draw(f"RED  {game_score['r']}  |  {game_score['y']}  YELLOW", screen, (WIN_WIDTH // 2 - 62, WIN_HEIGHT // 2 + 50))
                            pygame.display.flip()
                            pygame.time.wait(5000)
                            pygame.quit()
                            break
                        elif game_score['r'] < game_score['y']:
                            screen.fill((50, 50, 50))
                            title.draw("YELLOW WINS", screen, (WIN_WIDTH // 2 - 120, WIN_HEIGHT // 2 - 20))
                            score.draw(f"RED  {game_score['r']}  |  {game_score['y']}  YELLOW", screen, (WIN_WIDTH // 2 - 62, WIN_HEIGHT // 2 + 50))
                            pygame.display.flip()
                            pygame.time.wait(5000)
                            pygame.quit()
                            break
                    
                    # Display starting colour
                    rock = rocks[sim.active_rock]
                    screen.fill((50, 50, 50))
                    if rock.rock_colour == 'r':
                        next_colour.draw("RED START ", screen, (WIN_WIDTH // 2 - 100, WIN_HEIGHT // 2 - 30))
                    elif rock.rock_colour == 'y':
                        next_colour.draw("YELLOW START ", screen, (WIN_WIDTH // 2 - 120, WIN_HEIGHT // 2 - 30))
                    pygame.display.flip()
                    pygame.time.wait(1500)

if __name__ == '__main__':
    main()