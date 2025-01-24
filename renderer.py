import pygame
import math

import torch

class Renderer():
    def __init__(self, max_shapes, width, height):
        self.max_shapes = max_shapes
        self.width = width
        self.height = height
        self.shape_mapping = self.create_shape_mapping()

    def create_shape_mapping(self):
        shapes = []
        for i in range(3, 3 + self.max_shapes):
            shapes.append(i)  # Polygons with i sides
        return shapes

    def render(self, history, epoch):
        pygame.init()
        self.font = pygame.font.SysFont(None, 24)

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"Episode {epoch}")
        
        clock = pygame.time.Clock()

        # Render initial state
        self.screen.fill((255, 255, 255))

        initial_agents = history["initial_agents"]
        initial_landmarks = history["initial_landmarks"]

        # Render initial agents
        agent_positions = initial_agents["positions"][0].cpu().numpy()
        agent_colors = initial_agents["colors"][0].cpu().numpy()
        agent_shapes = initial_agents["shapes"][0].cpu().numpy().astype(int)
        agent_gaze = initial_agents["gaze"][0].cpu().numpy()

        for i, pos in enumerate(agent_positions):
            color = agent_colors[i]
            shape = self.shape_mapping[agent_shapes[i]]
            self.draw_shape(shape, color, pos)
            self.draw_gaze(color, pos, agent_gaze[i])

        # Render initial landmarks
        landmark_positions = initial_landmarks["positions"][0].cpu().numpy()
        landmark_colors = initial_landmarks["colors"][0].cpu().numpy()
        landmark_shapes = initial_landmarks["shapes"][0].cpu().numpy().astype(int)

        for i, pos in enumerate(landmark_positions):
            color = landmark_colors[i]
            shape = self.shape_mapping[landmark_shapes[i]]
            self.draw_shape(shape, color, pos)

        pygame.display.flip()

        # Animate the rest of the episode
        timesteps = len(history["agents"][0]["positions"])

        for t in range(timesteps):
            self.screen.fill((255, 255, 255))

            # Render agents
            for agent_idx in range(len(history["agents"])):
                pos = history["agents"][agent_idx]["positions"][t][0].cpu().detach().numpy()
                color = history["initial_agents"]["colors"][0][agent_idx].cpu().detach().numpy()
                shape = self.shape_mapping[history["initial_agents"]["shapes"][0][agent_idx].cpu().detach().numpy().astype(int)]
                gaze = history["agents"][agent_idx]["gaze"][t][0].cpu().detach().numpy()
                utterance = history["agents"][agent_idx]["utterances"][t][0].cpu().detach().numpy()

                self.draw_shape(shape, color, pos)
                self.draw_gaze(color, pos, gaze)
                self.draw_utterance(utterance, pos)

            # Render landmarks
            for i, pos in enumerate(landmark_positions):
                color = landmark_colors[i]
                shape = self.shape_mapping[landmark_shapes[i]]
                self.draw_shape(shape, color, pos)

            pygame.display.flip()
            clock.tick(5)  # Adjust the speed of the animation

        pygame.quit()

    def draw_shape(self, sides, color, pos):
        size = 10  # Increase the size of the shapes
        if sides == 3:
            points = [(int(pos[0]), int(pos[1]) - size), (int(pos[0]) - size, int(pos[1]) + size), (int(pos[0]) + size, int(pos[1]) + size)]
            pygame.draw.polygon(self.screen, color, points)
        elif sides == 4:
            pygame.draw.rect(self.screen, color, pygame.Rect(int(pos[0]) - size, int(pos[1]) - size, 2 * size, 2 * size))
        else:
            points = []
            for i in range(sides):
                angle = 2 * 3.14159 * i / sides
                x = int(pos[0]) + size * math.cos(angle)
                y = int(pos[1]) + size * math.sin(angle)
                points.append((x, y))
            pygame.draw.polygon(self.screen, color, points)

    def draw_gaze(self, color, pos, gaze):
        start_pos = (int(pos[0]), int(pos[1]))
        end_pos = (int(gaze[0]), int(gaze[1]))
        num_dots = 20
        for i in range(num_dots):
            t = i / num_dots
            x = int(start_pos[0] * (1 - t) + end_pos[0] * t)
            y = int(start_pos[1] * (1 - t) + end_pos[1] * t)
            pygame.draw.circle(self.screen, color, (x, y), 2)

    def draw_utterance(self, utterance, pos):
        utterance_index = torch.argmax(torch.tensor(utterance)).item()
        utterance_text = str(utterance_index)
        text_surface = self.font.render(utterance_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (int(pos[0]), int(pos[1]) - 20))