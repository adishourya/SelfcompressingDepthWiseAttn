import manim

CAMERA_FILE_PATH = "./camera.svg"
# EAGLE_FILE_PATH = "./gemini_bird_camera.png"
EAGLE_FILE_PATH = "./eagle.svg"

print(manim.config)
print("===========")
print(manim.config.frame_height , manim.config.frame_width)


class PhotoCaptureScene(manim.Scene):
    def construct(self):
        # Set white background
        self.camera.background_color = manim.WHITE

        # Load main image (eagle)
        eagle = manim.SVGMobject(EAGLE_FILE_PATH)
        # eagle = manim.ImageMobject(EAGLE_FILE_PATH)
        eagle.scale(4)
        # eagle.set(height=6)  # scale to reasonable size
        # eagle.set(width=6)  # scale to reasonable size
        self.add(eagle)
        # eagle.to_edge(manim.LEFT)
        self.wait(1)

        # Add small camera icon at bottom-right
        camera_icon = manim.SVGMobject(CAMERA_FILE_PATH)
        camera_icon.set(height=0.8)
        camera_icon.next_to(eagle,manim.DOWN)
        # camera_icon.to_corner(manim.DOWN + manim.RIGHT, buff=0.1)
        self.add(camera_icon)

        # Simulate camera flash: whitish overlay fade
        flash = manim.Rectangle(
            width=manim.config.frame_width,
            height=manim.config.frame_height,
            fill_color=manim.WHITE,
            fill_opacity=0,
        )
        self.add(flash)

        # Fade in the flash quickly, then fade out
        self.play(flash.animate.set_opacity(0.8), run_time=0.15)
        self.play(flash.animate.set_opacity(0), run_time=0.15)


        # Duplicate the image to show the captured photo
        captured_eagle = eagle.copy()
        captured_eagle.scale(0.3)
        captured_eagle.to_corner(manim.DOWN + manim.LEFT, buff=0.1)
        captured_eagle.shift(manim.RIGHT * 1.5)
        self.play(manim.FadeIn(captured_eagle, shift=manim.DOWN * 1.5), run_time=0.5)

        self.wait(2)

if __name__ == "__main__":
    import os
    os.system("manim -qh --resolution 1000,1000 wildlife_photography.py PhotoCaptureScene")
