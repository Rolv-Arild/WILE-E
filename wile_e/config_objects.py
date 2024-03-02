import math
from typing import List, Dict, Any

import numpy as np
from rlgym.api import RewardFunction, AgentID, StateType, RewardType
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import CAR_MAX_SPEED, BACK_WALL_Y, GOAL_HEIGHT, GOAL_CENTER_TO_POST, BALL_RADIUS
from rlgym.rocket_league.math import cosine_similarity

''
GOAL_THRESHOLD = 5215.5  # Tested in-game with BakkesMod


class VelocityPlayerToBallReward(RewardFunction[str, GameState, float]):
    def reset(self, initial_state: StateType, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: StateType, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, RewardType]:
        rewards = {}
        for agent in agents:
            vel = state.cars[agent].physics.linear_velocity
            player_ball = state.ball.position - state.cars[agent].position
            rewards[agent] = vel.dot(player_ball) / CAR_MAX_SPEED
        return rewards


def closest_point_in_goal(ball_pos):
    # Find the closest point on each goal to the ball
    x = math.copysign(1, ball_pos[0]) * min(abs(ball_pos[0]), GOAL_CENTER_TO_POST - BALL_RADIUS)
    y = BACK_WALL_Y + BALL_RADIUS
    z = min(ball_pos[2], GOAL_HEIGHT - BALL_RADIUS)
    return np.array([x, y, z])


def solid_angle_eriksson(O, A, B, C):
    # Calculate the solid angle of a triangle
    a = A - O
    b = B - O
    c = C - O
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    c /= np.linalg.norm(c)
    numerator = np.linalg.norm(np.dot(np.cross(a, b), c))  # noqa numpy complains about cross for no reason
    denominator = 1 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a)
    E = 2 * math.atan2(numerator, denominator)
    return E


def view_goal_ratio(pos, goal_y):
    # Calculate the percent of the field of view that the goal takes up
    bl = np.array([-GOAL_CENTER_TO_POST, goal_y, 0])
    br = np.array([GOAL_CENTER_TO_POST, goal_y, 0])
    tl = np.array([-GOAL_CENTER_TO_POST, goal_y, GOAL_HEIGHT])
    tr = np.array([GOAL_CENTER_TO_POST, goal_y, GOAL_HEIGHT])
    solid_angle_1 = solid_angle_eriksson(pos, bl, br, tl)
    solid_angle_2 = solid_angle_eriksson(pos, br, tr, tl)
    return (solid_angle_1 + solid_angle_2) / (4 * math.pi)


def solid_angle_ball(pos, ball_pos, ball_radius=BALL_RADIUS):
    # Calculate the solid angle of the ball
    d = np.linalg.norm(pos - ball_pos)
    r_sphere = math.sqrt(d ** 2 - ball_radius ** 2)
    E = 2 * math.pi * (1 - r_sphere / d)
    return E


def view_ball_ratio(pos, ball_pos):
    # Calculate the percent of the field of view that the ball takes up
    solid_angle = solid_angle_ball(pos, ball_pos)
    return solid_angle / (4 * math.pi)


class GoalViewReward(RewardFunction[str, GameState, float]):

    def __init__(self, transform="default"):
        self.current_quality = 0.
        if transform == "default":
            transform = lambda x: max(math.log2(x), -20)
        elif transform == "identity":
            transform = lambda x: x
        self.transform = transform

    def calculate_quality(self, state: StateType):
        ball_pos = state.ball.position
        blue_goal_view = view_goal_ratio(ball_pos, -GOAL_THRESHOLD)
        orange_goal_view = view_goal_ratio(ball_pos, +GOAL_THRESHOLD)
        quality = self.transform(blue_goal_view) - self.transform(orange_goal_view)
        return quality

    def reset(self, initial_state: StateType, shared_info: Dict[str, Any]) -> None:
        self.current_quality = self.calculate_quality(initial_state)

    def get_rewards(self, agents: List[AgentID], state: StateType, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, RewardType]:
        current_quality = self.calculate_quality(state)
        reward = current_quality - self.current_quality
        self.current_quality = current_quality
        return {agent: reward for agent in agents}


class StateQualityReward(RewardFunction[str, GameState, float]):
    def __init__(self):
        self.state_quality = 0.
        self.player_qualities = {}

    def calculate_quality(self, state: StateType):
        state_quality = 0
        player_qualities = {a: 0 for a in state.cars.keys()}

        # Goal view
        ball_pos = state.ball.position
        view_blue_goal = view_goal_ratio(ball_pos, -GOAL_THRESHOLD)
        view_orange_goal = view_goal_ratio(ball_pos, +GOAL_THRESHOLD)
        state_quality += max(np.log2(view_blue_goal), -20) - max(np.log2(view_orange_goal), -20)

        for car in state.cars.values():
            # Bepis
            car_ball = state.ball.position - car.position
            car_blue_goal = closest_point_in_goal(ball_pos)
            car_orange_goal = car_blue_goal * np.array([1, -1, 1])

            if car.is_blue:
                car_own_goal = car_blue_goal
                car_opp_goal = car_orange_goal
            else:
                car_own_goal = car_orange_goal
                car_opp_goal = car_blue_goal

            bepis = cosine_similarity(car_ball, car_opp_goal) - cosine_similarity(car_ball, car_own_goal)
            player_qualities[car.car_id] += bepis

        diff_state_quality = state_quality - self.state_quality
        self.state_quality = state_quality
        diff_player_qualities = {a: player_qualities[a] - self.player_qualities.get(a, 0) for a in player_qualities}
        self.player_qualities = player_qualities
        return diff_state_quality, diff_player_qualities

    def reset(self, initial_state: StateType, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: StateType, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, RewardType]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            rewards[agent] = -((car.position[0] ** 2 + car.position[1] ** 2) ** 0.5)
        return rewards
