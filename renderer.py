import pygame
import random
import math

class Renderer():
    def __init__(self, max_shapes, width, height):
        self.max_shapes = max_shapes
        self.width = width
        self.height = height
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Initial State')
        self.shape_mapping = self.create_shape_mapping()

    def create_shape_mapping(self):
        shapes = []
        for i in range(3, 3 + self.max_shapes):
            shapes.append(i)  # Polygons with i sides
        return shapes

    def render(self, history):
        # Wait for a few seconds to view the rendered state
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            initial_agents = history["initial_agents"]
            initial_landmarks = history["initial_landmarks"]

            self.screen.fill((255, 255, 255))

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
                j = landmark_shapes[i]
                shape = self.shape_mapping[j]
                self.draw_shape(shape, color, pos)

            pygame.display.flip()

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