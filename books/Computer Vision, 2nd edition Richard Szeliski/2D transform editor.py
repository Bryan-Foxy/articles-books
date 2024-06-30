import pygame

def draw_button(screen, color, rect, text):
    pygame.draw.rect(screen, color, rect)
    font = pygame.font.Font(None, 36)
    text_surf = font.render(text, True, (0, 0, 0))
    text_rect = text_surf.get_rect(center=rect.center)
    screen.blit(text_surf, text_rect)

def main():
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)

    pygame.init()
    pygame.display.set_caption('2D transform editor')
    screen = pygame.display.set_mode((800, 600))

    run = True
    drawing_mode = False

    # Define rectangles variables
    rect_start = None
    rect_end = None
    drawing = False
    rectangles = []

    # Define button
    button_rect = pygame.Rect(0, 5, 200, 50)
    button_color = RED
    button_text = "Draw Rectangles"

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_pos = event.pos
                    if button_rect.collidepoint(mouse_pos):
                        drawing_mode = not drawing_mode
                        button_color = GREEN if drawing_mode else RED
                        button_text = "Stop Drawing" if drawing_mode else "Draw Rectangles"
                    elif drawing_mode:
                        rect_start = event.pos
                        drawing = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and drawing:
                    rect_end = event.pos
                    rectangles.append(pygame.Rect(rect_start, (rect_end[0] - rect_start[0], rect_end[1] - rect_start[1])))
                    drawing = False

            elif event.type == pygame.MOUSEMOTION:
                if drawing:
                    rect_end = event.pos

        # Fill the screen with white
        screen.fill(WHITE)

        # Draw all rectangles
        for rect in rectangles:
            pygame.draw.rect(screen, BLACK, rect, 2)

        # Draw the current rectangle being drawn
        if drawing and rect_start and rect_end:
            temp_rect = pygame.Rect(rect_start, (rect_end[0] - rect_start[0], rect_end[1] - rect_start[1]))
            pygame.draw.rect(screen, BLACK, temp_rect, 2)

        # Draw the button
        draw_button(screen, button_color, button_rect, button_text)

        # Update the screen
        pygame.display.flip()

    pygame.quit()
    print('Exit')

if __name__ == '__main__':
    main()