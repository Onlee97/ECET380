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
#Creating DQN 
#==================
from tf_agents.networks.q_network import QNetwork

preprocessing_layer = keras.layers.Lambda(
						  lambda obs: tf.cast(obs, np.float32) / 255.)
conv_layer_params=[(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
fc_layer_params=[512]

q_net = QNetwork(
	tf_env.observation_spec(),
	tf_env.action_spec(),
	preprocessing_layers=preprocessing_layer,
	conv_layer_params=conv_layer_params,
	fc_layer_params=fc_layer_params)

#==================
#Creating DQN Agent
#==================
from tf_agents.agents.dqn.dqn_agent import DqnAgent

# see TF-agents issue #113
#optimizer = keras.optimizers.RMSprop(lr=2.5e-4, rho=0.95, momentum=0.0,
#                                     epsilon=0.00001, centered=True)

train_step = tf.Variable(0)
update_period = 4 # run a training step every 4 collect steps
optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=2.5e-4, decay=0.95, momentum=0.0,
									 epsilon=0.00001, centered=True)
epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
	initial_learning_rate=1.0, # initial ε
	decay_steps=250000 // update_period, # <=> 1,000,000 ALE frames
	end_learning_rate=0.01) # final ε
agent = DqnAgent(tf_env.time_step_spec(),
				 tf_env.action_spec(),
				 q_network=q_net,
				 optimizer=optimizer,
				 target_update_period=2000, # <=> 32,000 ALE frames
				 td_errors_loss_fn=keras.losses.Huber(reduction="none"),
				 gamma=0.99, # discount factor
				 train_step_counter=train_step,
				 epsilon_greedy=lambda: epsilon_fn(train_step))
agent.initialize()

#==================
#Creating replay Buffer
#==================

from tf_agents.replay_buffers import tf_uniform_replay_buffer

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
	data_spec=agent.collect_data_spec,
	batch_size=tf_env.batch_size,
	max_length=1000000)

replay_buffer_observer = replay_buffer.add_batch


#Add trainning Metrics
from tf_agents.metrics import tf_metrics
train_metrics = [
	tf_metrics.NumberOfEpisodes(),
	tf_metrics.EnvironmentSteps(),
	tf_metrics.AverageReturnMetric(),
	tf_metrics.AverageEpisodeLengthMetric(),
]

from tf_agents.eval.metric_utils import log_metrics
import logging
logging.getLogger().setLevel(logging.INFO)
log_metrics(train_metrics)

#==================
#Creating Collect Driver
#==================
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver

collect_driver = DynamicStepDriver(
	tf_env,
	agent.collect_policy,
	observers=[replay_buffer_observer] + train_metrics,
	num_steps=update_period) # collect 4 steps for each training iteration

from tf_agents.policies.random_tf_policy import RandomTFPolicy

initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(),
										tf_env.action_spec())

num_steps = 100
# num_steps = 20000
init_driver = DynamicStepDriver(
	tf_env,
	initial_collect_policy,
	observers=[replay_buffer.add_batch, ShowProgress(num_steps)],
	num_steps=num_steps) # <=> 80,000 ALE frames
final_time_step, final_policy_state = init_driver.run()

#Trajectories
tf.random.set_seed(888) # chosen to show an example of trajectory at the end of an episode
trajectories, buffer_info = replay_buffer.get_next(
	sample_batch_size=2, num_steps=3)

from tf_agents.trajectories.trajectory import to_transition
time_steps, action_steps, next_time_steps = to_transition(trajectories)

def show_episode():
	plt.figure(figsize=(10, 6.8))
	for row in range(2):
		for col in range(3):
			plt.subplot(2, 3, row * 3 + col + 1)
			plot_observation(trajectories.observation[row, col].numpy())
	plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0.02)
	save_fig("sub_episodes_plot")
	plt.show()

#==================
#Creating Dataset 
#==================
dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    num_parallel_calls=3).prefetch(3)

from tf_agents.utils.common import function
collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)

#==================
#Training Loop 
#==================
def train_agent(n_iterations):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)
    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        print("\r{} loss:{:.5f}".format(
            iteration, train_loss.loss.numpy()), end="")
        if iteration % 1000 == 0:
            log_metrics(train_metrics)

# train_agent(n_iterations=10000)
train_agent(n_iterations=100)

#==================
#Save Policy
#==================
from tf_agents.policies.policy_saver import PolicySaver
policy = agent.policy
saver = PolicySaver(policy, batch_size=None)
saver.save("savedPolicy")

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
    agent.policy,
    observers=[save_frames, reset_and_fire_on_life_lost, ShowProgress(1000)],
    num_steps=1000)
final_time_step, final_policy_state = watch_driver.run()
plot_animation(frames)

import PIL
image_path = os.path.join("images", "rl", "assault.gif")
frame_images = [PIL.Image.fromarray(frame) for frame in frames[:150]]
frame_images[0].save(image_path, format='GIF',
                     append_images=frame_images[1:],
                     save_all=True,
                     duration=30,
                     loop=0)