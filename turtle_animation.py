import turtle
import math
import os
import imageio
from PIL import Image
import random

turtle.speed(10)
turtle.title("Heart Curve")
turtle.bgcolor("white")

frame_number = 0

def save_frame():
    global frame_number
    canvas = turtle.getcanvas()
    canvas.postscript(file=f"frame_{frame_number:04}.eps", colormode='color')

    img = Image.open(f"frame_{frame_number:04}.eps")
    img.save(f"frame_{frame_number:04}.png", 'png')
    os.remove(f"frame_{frame_number:04}.eps")
    frame_number += 1



# turtle.shape("turtle")   
# turtle.shapesize(1.5)


def draw_heart(iterations, initial_chaos, chaos_increment):
    chaos_factor = initial_chaos  # Starting chaos factor

    for iteration in range(iterations):
        prev_x, prev_y = None, None

        for i in range(361):
            t = math.radians(i)

            # Apply the chaotic adjustment
            random_adjustment = 1 + random.uniform(-chaos_factor, chaos_factor)

            x = random_adjustment * 10 * (16 * math.sin(t) ** 3)
            y = random_adjustment * 10 * (13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))

            if prev_x is not None and prev_y is not None:
                angle = math.atan2(y - prev_y, x - prev_x)
                turtle.setheading(math.degrees(angle))

            # Transition color to white as the iterations progress
            overall_progress = (iteration * 361 + i) / (iterations * 361)
            turtle.pencolor(1, overall_progress, overall_progress)
            turtle.fillcolor(1, overall_progress, overall_progress)

            turtle.goto(x, y)
            prev_x, prev_y = x, y
            save_frame()

        # Continuously increase the chaos factor
        chaos_factor += chaos_increment

draw_heart(iterations=5, initial_chaos=0.1, chaos_increment=0.1)
turtle.done()


