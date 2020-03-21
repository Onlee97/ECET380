#==================
#Import Dependency
#==================
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from tensorflow import keras

# To get smooth animations
import matplotlib as mpl
import matplotlib.animation as animation
mpl.rc('animation', html='jshtml')

from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver

#==================
#Display Graph and Image
#==================
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rl"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
	path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
	print("Saving figure", fig_id)
	if tight_layout:
		plt.tight_layout()
	plt.savefig(path, format=fig_extension, dpi=resolution)

def update_scene(num, frames, patch):
	patch.set_data(frames[num])
	return patch,

def plot_animation(frames, repeat=False, interval=40):
	fig = plt.figure()
	patch = plt.imshow(frames[0])
	plt.axis('off')
	anim = animation.FuncAnimation(
		fig, update_scene, fargs=(frames, patch),
		frames=len(frames), repeat=repeat, interval=interval)
	plt.close()
	return anim

def plot_observation(obs):
	# Since there are only 3 color channels, you cannot display 4 frames
	# with one primary color per frame. So this code computes the delta between
	# the current frame and the mean of the other frames, and it adds this delta
	# to the red and blue channels to get a pink color for the current frame.
	obs = obs.astype(np.float32)
	img = obs[..., :3]
	current_frame_delta = np.maximum(obs[..., 3] - obs[..., :3].mean(axis=-1), 0.)
	img[..., 0] += current_frame_delta
	img[..., 2] += current_frame_delta
	img = np.clip(img / 150, 0, 1)
	plt.imshow(img)
	plt.axis("off")

class ShowProgress:
	def __init__(self, total):
		self.counter = 0
		self.total = total
	def __call__(self, trajectory):
		if not trajectory.is_boundary():
			self.counter += 1
		if self.counter % 100 == 0:
			print("\r{}/{}".format(self.counter, self.total), end="")

#==================
#Initialize env
#==================
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4

max_episode_steps = 27000 # <=> 108k ALE frames since 1 step = 4 frames
# environment_name = "BreakoutNoFrameskip-v4"
environment_name = "AssaultNoFrameskip-v0"

env = suite_atari.load(
	environment_name,
	max_episode_steps=max_episode_steps,
	gym_env_wrappers=[AtariPreprocessing, FrameStack4])

from tf_agents.environments.tf_py_environment import TFPyEnvironment
tf_env = TFPyEnvironment(env)

def show_env(env):
	print("Possible Action: ", env.gym.get_action_meanings())
	img = env.render(mode="rgb_array")
	plt.figure(figsize=(6, 8))
	plt.imshow(img)
	plt.axis("off")
	save_fig("breakout_plot")
	plt.show()

# show_env(env)

#==================
#Load Policy  
#==================
saved_policy = tf.compat.v2.saved_model.load("savedPolicy")

#==================
#Test run  
#==================
frames = []
def save_frames(trajectory):
    global frames
    frames.append(tf_env.pyenv.envs[0].render(mode="rgb_array"))

prev_lives = tf_env.pyenv.envs[0].ale.lives()
def reset_and_fire_on_life_lost(trajectory):
    global prev_lives
    lives = tf_env.pyenv.envs[0].ale.lives()
    if prev_lives != lives:
        tf_env.reset()
        tf_env.pyenv.envs[0].step(1)
        prev_lives = lives

watch_driver = DynamicStepDriver(
    tf_env,
    saved_policy,
    observers=[save_frames, reset_and_fire_on_life_lost, ShowProgress(1000)],
    num_steps=1000)
final_time_step, final_policy_state = watch_driver.run()
plot_animation(frames)

import PIL
image_path = os.path.join("images", "rl", "assault.gif")
frame_images = [PIL.Image.fromarray(frame) for frame in frames[:500]]
frame_images[0].save(image_path, format='GIF',
                     append_images=frame_images[1:],
                     save_all=True,
                     duration=30,
                     loop=0)