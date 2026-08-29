import nnlib_py as nn
import numpy as np
import pygame
import math
import random
from utils import center_drawing, array_to_cords, cords_to_array, softmax

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 410

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("AI digit recognition")
clock = pygame.time.Clock()

network = nn.Network(nn.Losses.softmax_cross_entropy)
network.add(nn.PreTrainedLayer("../models/v2-9.95%/layer_0_weights.npy", "../models/v2-9.95%/layer_0_biases.npy", nn.Activations.relu))
network.add(nn.PreTrainedLayer("../models/v2-9.95%/layer_1_weights.npy", "../models/v2-9.95%/layer_1_biases.npy", nn.Activations.relu))
network.add(nn.PreTrainedLayer("../models/v2-9.95%/layer_2_weights.npy", "../models/v2-9.95%/layer_2_biases.npy", nn.Activations.linear))

font = pygame.font.Font(None, 32)
big_font = pygame.font.Font(None, 62)

pixelmatrix = np.zeros(784)
virtual_pixel_size = 12
results = {}
empty = True
amount_predictions_shown = 5

button_width = 120
button_height = 45
button_hor_gap = ((28 * virtual_pixel_size) - 2 * button_width) / 3
button_y = ((WINDOW_HEIGHT - virtual_pixel_size * 28) - button_height) / 2 + virtual_pixel_size * 28 + 2
clear_button_rect = pygame.Rect(button_hor_gap, button_y, button_width, button_height)
center_button_rect = pygame.Rect(button_hor_gap * 2 + button_width, button_y, button_width, button_height)

confidence_bar_max_width = WINDOW_WIDTH - 28 * virtual_pixel_size - 3
confidence_bar_height = 28 * virtual_pixel_size / amount_predictions_shown
confidence_bar_x = 28 * virtual_pixel_size + 3

running = True

while running:
    # event loop
    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if clear_button_rect.collidepoint(pygame.mouse.get_pos()):
                    pixelmatrix = np.zeros(784)
                    empty = True
                elif center_button_rect.collidepoint(pygame.mouse.get_pos()):
                    if not empty:
                        pixelmatrix = center_drawing(pixelmatrix)

    # drawing logic
    if pygame.mouse.get_pressed()[0] and pygame.mouse.get_pos()[0] < virtual_pixel_size * 28 - 1 and pygame.mouse.get_pos()[1] < virtual_pixel_size * 28 - 1 and pygame.mouse.get_pos()[0] > 1 and pygame.mouse.get_pos()[1] > 1:
        empty = False
        matrix_location = (round(pygame.mouse.get_pos()[0] / virtual_pixel_size) - 1, round(pygame.mouse.get_pos()[1] / virtual_pixel_size) - 1)
        center_pix = cords_to_array(matrix_location)
        top_pix = cords_to_array((matrix_location[0], matrix_location[1] - 1))
        right_pix = cords_to_array((matrix_location[0] + 1, matrix_location[1]))
        bottom_pix = cords_to_array((matrix_location[0], matrix_location[1] + 1))
        left_pix = cords_to_array((matrix_location[0] - 1, matrix_location[1]))
        if center_pix >= 0 and center_pix < 784:
            pixelmatrix[center_pix] = 1.0
        if top_pix >= 0 and top_pix < 784:
            if pixelmatrix[top_pix] == 0.0:
                pixelmatrix[top_pix] = random.uniform(0.31, 0.78)
        if right_pix >= 0 and right_pix < 784:
            if pixelmatrix[right_pix] == 0.0:
                pixelmatrix[right_pix] = random.uniform(0.31, 0.78)
        if bottom_pix >= 0 and bottom_pix < 784:
            if pixelmatrix[bottom_pix] == 0.0:
                pixelmatrix[bottom_pix] = random.uniform(0.31, 0.78)
        if left_pix >= 0 and left_pix < 784:
            if pixelmatrix[left_pix] == 0.0:
                pixelmatrix[left_pix] = random.uniform(0.31, 0.78)

    # ai logic
    if not empty:
        prediction_array = network.forward(pixelmatrix)[0]
        softmaxed_prediction_array = softmax(prediction_array)
        for i in range(10):
            results[i] = softmaxed_prediction_array[i]
        sorted_results = [
            {"number": key, "confidence": value}
            for key, value in sorted(results.items(), key=lambda item: item[1], reverse=True)
        ]

    # displaying
    screen.fill((2, 56, 41))
    pygame.draw.rect(screen, (0,0,0), (28 * virtual_pixel_size + 1, 1, 1, WINDOW_HEIGHT))
    pygame.draw.rect(screen, (0,0,0), (1, 28 * virtual_pixel_size + 1, WINDOW_WIDTH, 1))
    pygame.draw.rect(screen, (0,0,0), clear_button_rect, width=1, border_radius=3)
    clear_text = font.render("CLEAR", True, (0,0,0))
    clear_text_surface = clear_text.get_rect()
    clear_text_surface.center = clear_button_rect.center
    screen.blit(clear_text, clear_text_surface)
    pygame.draw.rect(screen, (0,0,0), center_button_rect, width=1, border_radius=3)
    center_text = font.render("CENTER", True, (0,0,0))
    center_text_surface = center_text.get_rect()
    center_text_surface.center = center_button_rect.center
    screen.blit(center_text, center_text_surface)
    for i, virtual_pixel in enumerate(pixelmatrix):
        x,y = array_to_cords(i)
        pygame.draw.rect(screen, (virtual_pixel * 255, virtual_pixel * 255, virtual_pixel * 255), (x * virtual_pixel_size, y * virtual_pixel_size, virtual_pixel_size, virtual_pixel_size))
    if not empty:
        for i in range(amount_predictions_shown):
            pygame.draw.rect(screen, (0,0,0), (confidence_bar_x, confidence_bar_height * i, confidence_bar_max_width * sorted_results[i]["confidence"], confidence_bar_height))
            number_text = font.render(str(sorted_results[i]["number"]), True, (255,255,255))
            screen.blit(number_text, (confidence_bar_x + 8, (confidence_bar_height * i) + confidence_bar_height / 2 - number_text.get_height() / 2))
        big_number_text = big_font.render(str(sorted_results[0]["number"]), True, (255,255,255))
        screen.blit(big_number_text, (((WINDOW_WIDTH - 28 * virtual_pixel_size) / 2 - big_number_text.get_width() / 2) + 28 * virtual_pixel_size, ((WINDOW_HEIGHT - 28 * virtual_pixel_size) / 2 - big_number_text.get_height() / 2) + 3 + 28 * virtual_pixel_size))
    pygame.display.update()